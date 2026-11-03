from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic11(Heuristic):
    """
    Domain-dependent heuristic for Blocksworld_GTOHP.

    BOTTLENECK: Stack clearing and unstacking. To move a block or place something on it, 
    every block currently above it in the stack must be removed first.

    ADMISSIBILITY SKETCH:
    1. h_tasks: Sum of minimum required primitive actions to complete each task in the current task network.
       - do_put_on(x, y) requires clearing x (2*h_x actions), clearing y (2*h_y actions), and moving x to y (2 actions).
       - do_on_table(x) requires clearing x and putting it down (2 actions).
       - This is a lower bound because these tasks are serial and mandatory.
    2. h_goals: Displacement cost. Every block not 'well-placed' (either not in its goal position or 
       on a stack that must be moved) must be moved at least once (2 primitive actions: unstack/pick-up + stack/put-down).
    The max(h_tasks, h_goals) is a safe, admissible lower bound on the remaining cost.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_11="blocksworld_gemini3flash_heuristic_11"):
        super().__init__(blocksworld_gemini3flash_heuristic_11)
        self.on_ids = {}
        self.ontable_ids = {}
        self.clear_ids = {}
        self.holding_ids = {}
        self.handempty_id = -1
        self.all_blocks = []
        self.goal_on = {} # x -> y
        self.goal_ontable = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates and blocks
        block_names = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                block_names.add(args[0])
                block_names.add(args[1])
            elif pred == "ontable":
                self.ontable_ids[args[0]] = f.global_id
                block_names.add(args[0])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "holding":
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
        
        self.all_blocks = list(block_names)

        # 2. Parse Goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
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
        # 1. State Extraction (O(N))
        current_on = {} # x -> y or 'table' or 'hand'
        hand_block = None
        is_clear = {b: False for b in self.all_blocks}

        for b in self.all_blocks:
            if (state >> self.holding_ids[b]) & 1:
                current_on[b] = 'hand'
                hand_block = b
            elif (state >> self.ontable_ids.get(b, 0)) & 1:
                current_on[b] = 'table'
            else:
                found = False
                for other in self.all_blocks:
                    if b == other: continue
                    if (state >> self.on_ids.get((b, other), 0)) & 1:
                        current_on[b] = other
                        found = True
                        break
                if not found: current_on[b] = 'table' # Default fallback
            
            if (state >> self.clear_ids.get(b, 0)) & 1:
                is_clear[b] = True

        # 2. Build Stack structure (O(N))
        on_me = {b: None for b in self.all_blocks}
        on_me['table'] = []
        for b, y in current_on.items():
            if y != 'hand' and y != 'table':
                on_me[y] = b
            elif y == 'table':
                on_me['table'].append(b)

        # 3. Well-placed blocks (O(N) BFS from table)
        well_placed = set()
        queue = on_me['table']
        while queue:
            x = queue.pop(0)
            y = current_on[x]
            # x is on y, and y is either table or well_placed
            base_stable = (y == 'table' or y in well_placed)
            
            satisfies_goal = False
            if x in self.goal_on:
                if self.goal_on[x] == y: satisfies_goal = True
            elif x in self.goal_ontable:
                if y == 'table': satisfies_goal = True
            else:
                satisfies_goal = True # No goal means it can be anywhere stable
            
            if base_stable and satisfies_goal:
                well_placed.add(x)
                next_block = on_me.get(x)
                if next_block:
                    queue.append(next_block)

        # 4. Height Above / Clear Cost (O(N))
        # height_above[x] = total cost to remove all blocks on top of x
        height_above = {b: 0 for b in self.all_blocks}
        for b in self.all_blocks:
            if is_clear[b]:
                curr = b
                h = 0
                while curr != 'table' and curr != 'hand':
                    height_above[curr] = h
                    h += 2
                    curr_below = current_on[curr]
                    if curr_below == 'table' or curr_below == 'hand': break
                    curr = curr_below

        # 5. Term 1: Task Network Cost (O(TN))
        h_tasks = 0
        for task in task_network:
            task_type, args = self._parse_task_name(task.name)
            if task_type == "do_put_on" and len(args) == 2:
                x, y = args[0], args[1]
                if x in well_placed and current_on[x] == y: continue
                cost = 2 + height_above[x] + height_above[y]
                if hand_block == x: cost -= 1
                h_tasks += cost
            elif task_type == "do_on_table" and len(args) >= 1:
                x = args[0]
                if x in well_placed and current_on[x] == 'table': continue
                cost = 2 + height_above[x]
                if hand_block == x: cost -= 1
                h_tasks += cost
            elif task_type == "do_clear" and len(args) == 1:
                h_tasks += height_above[args[0]]
            elif task_type == "do_move" and len(args) == 2:
                x, y = args[0], args[1]
                cost = 2 + height_above[x] + height_above[y]
                if hand_block == x: cost -= 1
                h_tasks += cost

        # 6. Term 2: Displacement cost (O(N))
        # Admissible: 2 actions for every block that MUST move.
        h_goals = sum(2 for b in self.all_blocks if b not in well_placed)
        
        return max(h_tasks, h_goals)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []

    def _parse_task_name(self, name: str):
        if "[" in name:
            return self._parse_fact_name(name)
        parts = name.split()
        return (parts[0], parts[1:]) if parts else ("", [])
