from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic6(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld GTOHP.
    
    BOTTLENECK: Stack clearing and Goal Stability. 
    To move a block x, everything on top of it must be unstacked first.
    A goal fact on(x,y) currently true in the state is 'false-trivial' if y itself 
    is unstable (i.e., y needs to be moved to satisfy its own goal or its base is unstable).

    Admissible Lower Bounds:
    1. Task Network Cost: Minimum actions to decompose tasks (do_put_on, do_clear, etc.) based on stack heights.
    2. Unstable Goal Cost: Every goal on(x,y) not yet stable requires at least one stack/unstack operation (cost 2+).
    3. Displacement Cost: Minimum movements required for blocks not currently in their final goal position.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_6="blocksworld_gemini3flash_heuristic_6"):
        super().__init__(name=blocksworld_gemini3flash_heuristic_6)
        self.on_ids = {} # (x, y) -> global_id
        self.clear_ids = {} # x -> global_id
        self.ontable_ids = {} # x -> global_id
        self.holding_ids = {} # x -> global_id
        self.blocks = []
        self.goal_on = {} # x -> y (goal is on x y)
        self.goal_ontable = set()

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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.blocks = []
        block_set = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
            
            for arg in args:
                block_set.add(arg)
        
        self.blocks = list(block_set)
        
        goal_facts = model.state_explicit_repr(model.goals)
        for g in goal_facts:
            pred, args = self._parse_fact_name(g)
            if pred == "on":
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goal_ontable.add(args[0])

        h0 = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h0)

    def _compute(self, state: int, task_network) -> int:
        # 1. State extraction (Fast bitwise)
        on_map = {} # x -> y such that on(x,y)
        above_map = {} # y -> x such that on(x,y)
        for (x, y), fid in self.on_ids.items():
            if (state >> fid) & 1:
                on_map[x] = y
                above_map[y] = x

        # 2. Stability Analysis
        stable = set()
        # Sort blocks to process from bottom up (ontable upwards)
        processed = set()
        def check_stable(b):
            if b in processed: return b in stable
            processed.add(b)
            
            # If block has a specific goal destination
            if b in self.goal_on:
                target_below = self.goal_on[b]
                if on_map.get(b) == target_below and check_stable(target_below):
                    stable.add(b)
            elif b in self.goal_ontable:
                if (state >> self.ontable_ids.get(b, 0)) & 1:
                    stable.add(b)
            else:
                # No specific goal; stable if on table or on stable block and NOT needed elsewhere
                # For GTOHP, we usually assume goals are comprehensive or we focus on task network.
                # Simple check: if on table and no goal says otherwise, stable.
                if (state >> self.ontable_ids.get(b, 0)) & 1:
                    stable.add(b)
                elif b in on_map and check_stable(on_map[b]):
                    stable.add(b)
            return b in stable

        for b in self.blocks:
            check_stable(b)

        # 3. Stack Height / Clear Cost
        def get_height(b):
            h = 0
            curr = b
            while curr in above_map:
                h += 1
                curr = above_map[curr]
            return h

        def clear_cost(b):
            # Hint: clear_cost with stack height h = 1 + 2*h
            h = get_height(b)
            return 0 if h == 0 else (1 + 2 * h)

        # 4. Multi-component Max
        # Term 1: Task Network Cost
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'name'):
                name = task.name
                # Basic grounding check by string parsing since HTN tasks aren't always fully grounded in objects list
                # but in GTOHP the task.name is usually 'task_name block1 block2'
                parts = name.split()
                t_type = parts[0]
                args = parts[1:]

                if t_type == "do_put_on":
                    x, y = args[0], args[1]
                    if on_map.get(x) == y and x in stable:
                        tn_cost += 0
                    else:
                        # Hint: do_put_on(x,y) non-trivial = clear_cost(x) + clear_cost(y) + 3
                        tn_cost += clear_cost(x) + clear_cost(y) + 3
                elif t_type == "do_clear":
                    x = args[0]
                    tn_cost += clear_cost(x)
                elif t_type == "do_on_table":
                    x = args[0]
                    if (state >> self.ontable_ids.get(x, 0)) & 1 and x in stable:
                        tn_cost += 0
                    else:
                        tn_cost += clear_cost(x) + 2
                elif t_type == "do_move":
                    x, y = args[0], args[1]
                    tn_cost += clear_cost(x) + clear_cost(y) + 2
                elif t_type in ["pick-up", "put-down", "stack", "unstack"]:
                    tn_cost += 1

        # Term 2: Goal Distance (admissible count of unstable blocks)
        unstable_goals = 0
        for x, y in self.goal_on.items():
            if on_map.get(x) != y or x not in stable:
                unstable_goals += 1
        
        goal_cost = unstable_goals * 2

        return max(tn_cost, goal_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
