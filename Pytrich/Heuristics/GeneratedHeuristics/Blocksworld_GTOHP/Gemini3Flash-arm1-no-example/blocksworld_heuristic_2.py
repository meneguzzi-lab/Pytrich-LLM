from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic2(Heuristic):
    """
    Heuristic for the Blocksworld_GTOHP domain.

    1. Bottleneck Analysis:
       The primary bottleneck is block accessibility. Actions like moving a block or stacking 
       it require both the source and target to be clear. To clear a block buried under N 
       others, at least 2*N actions are required (unstack and put-down for each).

    2. Admissible Lower Bounds:
       - h_tn: Estimated minimum cost of the task network. For each do_put_on(x, y), we 
         require moving x to y (2 actions) plus the cost of clearing x and y.
       - h_goal: Goals on(x, y) not currently represented in the task network and not 
         satisfied in the state require at least 2 actions.
       - hand_penalty: If a block is currently held or the hand is not empty while a 
         pick-up is needed, it adds at least 1 action (put-down).
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_2="BlocksworldGTOHPGemini3FlashHeuristic2"):
        super().__init__(blocksworld_gemini3flash_heuristic_2)
        self.fid_to_on = {}
        self.fid_to_ontable = {}
        self.fid_to_holding = {}
        self.handempty_id = None
        self.goal_on = []

    def initialize(self, model: Model, initial_node: HTNNode):
        # Preprocess fact IDs for fast bitwise lookups in __call__
        for fid, fact in model.facts.items():
            prefix, args = self._parse_item_name(fact.name)
            if prefix == "on":
                self.fid_to_on[fid] = (args[0], args[1])
            elif prefix == "ontable":
                self.fid_to_ontable[fid] = args[0]
            elif prefix == "holding":
                self.fid_to_holding[fid] = args[0]
            elif prefix == "handempty":
                self.handempty_id = fid

        # Pre-parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            prefix, args = self._parse_item_name(g_name)
            if prefix == "on":
                self.goal_on.append((args[0], args[1]))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state: int, task_network: list) -> int:
        # 1. Parse current state bitwise (efficient O(N) extraction)
        current_on = {}
        is_ontable = set()
        is_holding = None
        
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            bit_pos = lsb.bit_length() - 1
            if bit_pos in self.fid_to_on:
                x, y = self.fid_to_on[bit_pos]
                current_on[x] = y
            elif bit_pos in self.fid_to_ontable:
                is_ontable.add(self.fid_to_ontable[bit_pos])
            elif bit_pos in self.fid_to_holding:
                is_holding = self.fid_to_holding[bit_pos]
            temp_state ^= lsb
            
        # 2. Map blocks to what is directly on top of them
        under = {y: x for x, y in current_on.items()}
            
        memo_clear = {}
        def get_clear_cost(block):
            if block in memo_clear: return memo_clear[block]
            c, curr = 0, under.get(block)
            while curr:
                c += 2
                curr = under.get(curr)
            memo_clear[block] = c
            return c

        # 3. Sum estimated costs of tasks in the network
        h_tn = 0
        tasks_in_tn = set()
        for t in task_network:
            prefix, args = self._parse_item_name(t.name)
            if prefix == "do_put_on":
                x, y = args[0], args[1]
                tasks_in_tn.add((x, y))
                if current_on.get(x) != y:
                    h_tn += 2 + get_clear_cost(x) + get_clear_cost(y)
            elif prefix == "do_clear":
                h_tn += get_clear_cost(args[0])
            elif prefix == "do_move":
                h_tn += 2 + get_clear_cost(args[0]) + get_clear_cost(args[1])
            elif prefix == "do_on_table":
                if args[0] not in is_ontable:
                    h_tn += 2 + get_clear_cost(args[0])
            elif prefix in ["pick-up", "unstack", "stack", "put-down"]:
                h_tn += 1
            elif prefix == "nop":
                pass
            else:
                h_tn += 1
                
        # 4. Sum costs for goal predicates not covered by task network
        h_goal = 0
        for x, y in self.goal_on:
            if (x, y) not in tasks_in_tn and current_on.get(x) != y:
                h_goal += 2
                
        # 5. Hand state logic
        if is_holding:
            h_tn += 1
        elif self.handempty_id is not None and not (state >> self.handempty_id) & 1:
            h_tn += 1

        return h_tn + h_goal

    def _parse_item_name(self, name: str):
        start = name.find("[")
        if start == -1:
            parts = name.split()
            if not parts: return name, []
            prefix = parts[0]
            if prefix.startswith("+") or prefix.startswith("-"): prefix = prefix[1:]
            return prefix, parts[1:]
        prefix = name[:start]
        if prefix.startswith("+") or prefix.startswith("-"): prefix = prefix[1:]
        args = name[start+1:-1].split(",") if "," in name[start+1:-1] else [name[start+1:-1]]
        return prefix, [a for a in args if a]
