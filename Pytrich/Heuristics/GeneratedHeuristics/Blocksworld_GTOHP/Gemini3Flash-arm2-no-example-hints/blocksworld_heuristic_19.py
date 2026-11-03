from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic19(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.
    
    BOTTLENECK: Stack clearing. Moving a buried block requires clearing every block above it first.
    Each unstack/put-down pair costs 2 actions. To move x to y, we must clear x (2*H(x)),
    clear y (2*H(y)), potentially put down y if it's held, and then pick and stack x (2).
    
    ADMISSIBILITY SKETCHES:
    1. Task Cost Sum: Each abstract task in the network has a minimum number of primitive actions 
       required to decompose based on the current stack heights. Summing these is admissible 
       assuming tasks represent disjoint work.
    2. Misplaced Blocks: Each block not in its final goal position (or resting on a misplaced block)
       must be moved at least once (unstack + stack/putdown), costing at least 2 actions.
    """
    
    def __init__(self, blocksworld_gemini3flash_heuristic_19="BlocksworldGTOHPGemini3FlashHeuristic19"):
        super().__init__(blocksworld_gemini3flash_heuristic_19)
        self.on_fact_id = {}      # (top, bottom) -> global_id
        self.ontable_fact_id = {} # block -> global_id
        self.holding_fact_id = {} # block -> global_id
        self.clear_fact_id = {}   # block -> global_id
        self.handempty_id = None
        self.blocks = []
        self.goal_on = {}         # top -> bottom
        self.goal_ontable = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to map predicates to IDs
        temp_blocks = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_id[(args[0], args[1])] = fact.global_id
                temp_blocks.add(args[0])
                temp_blocks.add(args[1])
            elif pred == "ontable":
                self.ontable_fact_id[args[0]] = fact.global_id
                temp_blocks.add(args[0])
            elif pred == "holding":
                self.holding_fact_id[args[0]] = fact.global_id
                temp_blocks.add(args[0])
            elif pred == "clear":
                self.clear_fact_id[args[0]] = fact.global_id
                temp_blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_id = fact.global_id
        
        self.blocks = list(temp_blocks)
        
        # 2. Parse goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goal_ontable.add(args[0])
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # 1. Get current physical structure O(N)
        under = {}
        over = {}
        for (top, bottom), f_id in self.on_fact_id.items():
            if (state >> f_id) & 1:
                under[top] = bottom
                over[bottom] = top
        
        on_table = set()
        for b, f_id in self.ontable_fact_id.items():
            if (state >> f_id) & 1:
                on_table.add(b)
        
        holding_block = None
        for b, f_id in self.holding_fact_id.items():
            if (state >> f_id) & 1:
                holding_block = b
                break
        
        # 2. Calculate heights (blocks above) O(N)
        heights = {b: 0 for b in self.blocks}
        for b in self.blocks:
            if b not in over: # b is clear
                curr = b
                h_above = 0
                while curr in under:
                    curr = under[curr]
                    h_above += 1
                    heights[curr] = h_above

        # 3. Misplaced blocks (Term 2) O(N)
        stable = set()
        # Blocks are stable if they are on table as required, or on stable blocks as required
        # Start with base blocks
        for b in on_table:
            # A block on table is stable if it's supposed to be there OR not restricted by goal
            if b not in self.goal_on:
                stable.add(b)
        
        # Propagate stability upwards
        changed = True
        while changed:
            changed = False
            for bottom in list(stable):
                top = over.get(bottom)
                if top and top not in stable:
                    # top is stable if its required 'under' is 'bottom' OR not in goal
                    goal_under = self.goal_on.get(top)
                    if goal_under == bottom or (top not in self.goal_on and top not in self.goal_ontable):
                        stable.add(top)
                        changed = True
        
        misplaced_count = len(self.blocks) - len(stable)
        if holding_block: misplaced_count -= 0 # holding is already "moved"

        # 4. Task-based Cost (Term 1) O(Tasks)
        sum_task_costs = 0
        for task in task_network:
            if not hasattr(task, '_cached_parsed'):
                parts = task.name.split()
                task._cached_parsed = (parts[0], parts[1:])
            
            t_name, t_args = task._cached_parsed
            
            if t_name == "do_put_on":
                x, y = t_args[0], t_args[1]
                is_on = False
                fid = self.on_fact_id.get((x, y))
                if fid is not None and (state >> fid) & 1:
                    is_on = True
                
                if not is_on:
                    # Formula: clear_cost(x) + clear_cost(y) + 5
                    # (2*Hx) + (2*Hy) + 5
                    cost = 2 * heights[x] + 2 * heights[y] + 5
                    if holding_block == x: cost -= 1
                    sum_task_costs += cost
                elif x not in stable: # Goal-stability check
                    sum_task_costs += 5
            
            elif t_name == "do_clear":
                sum_task_costs += 2 * heights[t_args[0]]
            
            elif t_name == "do_on_table":
                x = t_args[0]
                if x not in on_table:
                    sum_task_costs += 2 * heights[x] + 2
                    
            elif t_name == "do_move":
                sum_task_costs += 2
            elif t_name in ["pick-up", "put-down", "stack", "unstack"]:
                sum_task_costs += 1
        
        # 5. Hand penalty
        hand_penalty = 1 if holding_block else 0
        
        return max(sum_task_costs, 2 * misplaced_count) + hand_penalty

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
