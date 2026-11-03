from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class BlocksworldGTOHPGemini3FlashHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    Bottleneck Analysis:
    Stack clearing is the primary bottleneck. To move or stack a block, every block currently 
    above it in a tower must be unstacked first. This is a recursive process.

    Admissibility Sketches:
    1. sum_task_costs: Sum of the minimum actions required to satisfy each task in the 
       HTN task network. Since tasks are mostly serial in Blocksworld GTOHP, this is a strong lower bound.
    2. goal_stability_penalty: Every 'on(x,y)' goal that is not currently satisfied and stable 
       requires at least 3 actions (e.g., pick-up, stack, plus clearing costs).
    3. clear_cost: For a block to be clear, every block above it must be unstacked and put down, 
       costing 2 actions per block above plus a base expansion cost.
    """
    def __init__(self, blocksworld_gemini3flash_heuristic_2="blocksworld_gemini3flash_heuristic_2"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_2)
        self.on_ids = {}            # (x, y) -> global_id
        self.ontable_ids = {}       # x -> global_id
        self.clear_ids = {}         # x -> global_id
        self.handempty_id = -1
        self.on_with_bottom = defaultdict(list) # y -> [(x, global_id)]
        self.on_with_top = defaultdict(list)    # x -> [(y, global_id)]
        self.goal_on = {}           # x -> y (x should be on y)
        self.goal_ontable = set()   # x should be on table
        self.all_blocks = set()
        self.num_blocks = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.on_with_bottom[args[1]].append((args[0], f.global_id))
                self.on_with_top[args[0]].append((args[1], f.global_id))
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
                self.all_blocks.add(args[0])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id

        self.num_blocks = len(self.all_blocks)
        
        # Parse goal state
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goal_ontable.add(args[0])

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        well_placed = self._compute_well_placed(state)
        h_tasks = 0
        
        for task in task_network:
            name = task.name
            args = self._parse_task_args(name)
            if "do_put_on" in name and len(args) == 2:
                x, y = args[0], args[1]
                on_fid = self.on_ids.get((x, y))
                if on_fid is not None and (state >> on_fid) & 1 and y in well_placed:
                    continue
                # Non-trivial do_put_on cost: clear(x) + clear(y) + 3
                h_tasks += self._clear_cost(x, state) + self._clear_cost(y, state) + 3
            elif "do_clear" in name and len(args) == 1:
                h_tasks += self._clear_cost(args[0], state)
            elif "do_on_table" in name and len(args) == 1:
                x = args[0]
                fid = self.ontable_ids.get(x, -1)
                if fid == -1 or not ((state >> fid) & 1):
                    h_tasks += self._clear_cost(x, state) + 2
            elif "do_move" in name:
                h_tasks += 2
            elif any(p in name for p in ["pick-up", "put-down", "stack", "unstack"]):
                h_tasks += 1

        # Displacement penalty: blocks not well-placed must move
        displaced_count = 0
        for b in self.all_blocks:
            if b not in well_placed:
                displaced_count += 1
        
        # goal-stability / unsatisfied goals
        unsatisfied_goals = 0
        for x, y in self.goal_on.items():
            on_fid = self.on_ids.get((x, y))
            if on_fid is None or not ((state >> on_fid) & 1) or y not in well_placed:
                unsatisfied_goals += 1

        return max(h_tasks, unsatisfied_goals * 2) + int(0.001 * displaced_count)

    def _clear_cost(self, x, state):
        h = 0
        curr = x
        for _ in range(self.num_blocks):
            above = None
            for top_block, fid in self.on_with_bottom.get(curr, []):
                if (state >> fid) & 1:
                    above = top_block
                    break
            if above:
                h += 1
                curr = above
            else:
                break
        if h == 0: return 0
        return 1 + 2 * h

    def _compute_well_placed(self, state):
        well_placed = set()
        on_map = {}
        for bot, tops in self.on_with_bottom.items():
            for top, fid in tops:
                if (state >> fid) & 1:
                    on_map[bot] = top
                    break
        
        # Blocks on table
        for b, fid in self.ontable_ids.items():
            if (state >> fid) & 1:
                # Is it the goal position?
                if b in self.goal_ontable or (b not in self.goal_on and b not in self.goal_ontable):
                    well_placed.add(b)
                    # Propagate up
                    curr = b
                    while curr in on_map:
                        top = on_map[curr]
                        if self.goal_on.get(top) == curr:
                            well_placed.add(top)
                            curr = top
                        else:
                            break
        return well_placed

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_args(self, name):
        start = name.find('[')
        end = name.find(']')
        if start != -1 and end != -1:
            return name[start+1:end].split(',')
        return []
