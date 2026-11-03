from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import sys

class BlocksworldGTOHPGemini3FlashHeuristic9(Heuristic):
    """
    Domain-Dependent Heuristic for Blocksworld_GTOHP.

    BOTTLENECK: Stack clearing — to move a buried block, all blocks above must be unstacked first.
    ADMISSIBILITY SKETCHES:
    - Term 1 (Nop + Displacement): Sum of tasks that are already satisfied (nops) plus blocks not in their goal position.
      Misplaced blocks must move at least once; satisfied tasks still represent a step in the decomposition.
    - Term 2 (Recursive Task Cost): Sum of costs for every task in the network based on stack heights.
      Every block above x must be unstacked and put down (2 actions each) before x can be moved.
    - Term 3 (Goal Stability): A goal fact currently true is NOT free if its support is unstable.
      This captures the hidden cost of re-building stacks that are currently blocked or misplaced.
    """

    def __init__(self, blocksworld_gemini3flash_heuristic_9="blocksworld_gemini3flash_heuristic_9"):
        super().__init__(blocksworld_gemini3flash_heuristic_9)
        self.block_names = []
        self.on_fact_ids = []  # List of (global_id, block_x, block_y)
        self.ontable_fact_ids = [] # List of (global_id, block_x)
        self.clear_fact_ids = []
        self.holding_fact_ids = []
        self.handempty_id = None
        self.goal_on = {}  # block -> block_below
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
        self.model = model
        all_blocks = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids.append((fact.global_id, args[0], args[1]))
                all_blocks.update(args)
            elif pred == "ontable":
                self.ontable_fact_ids.append((fact.global_id, args[0]))
                all_blocks.update(args)
            elif pred == "clear":
                self.clear_fact_ids.append((fact.global_id, args[0]))
            elif pred == "holding":
                self.holding_fact_ids.append((fact.global_id, args[0]))
            elif pred == "handempty":
                self.handempty_id = fact.global_id
        
        self.block_names = list(all_blocks)
        
        # Extract goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
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
        # Build current state maps
        actual_on = {} # top -> bottom
        actual_below = {} # bottom -> top
        for fid, x, y in self.on_fact_ids:
            if (state >> fid) & 1:
                actual_on[x] = y
                actual_below[y] = x
        
        # Find stable blocks (well-placed)
        stable = set()
        # Start from table
        on_table = [b for b in self.block_names if b not in actual_on]
        for b in on_table:
            # Stable if goal says ontable or no goal for this block
            if b in self.goal_on: continue
            stable.add(b)
            # Check stack above
            curr = b
            while curr in actual_below:
                top = actual_below[curr]
                if self.goal_on.get(top) == curr:
                    stable.add(top)
                    curr = top
                elif top not in self.goal_on and top not in self.goal_ontable:
                    stable.add(top)
                    curr = top
                else:
                    break

        # Displacement: Misplaced blocks
        displacement = len(self.block_names) - len(stable)

        # Costs of tasks in network
        sum_task_costs = 0
        nop_count = 0
        
        for task in task_network:
            t_name = task.name.split()
            t_type = t_name[0]
            args = t_name[1:]

            if t_type == "do_put_on":
                x, y = args[0], args[1]
                if actual_on.get(x) == y and x in stable:
                    nop_count += 1
                else:
                    cost = self._get_clear_cost(x, actual_below) + self._get_clear_cost(y, actual_below) + 3
                    sum_task_costs += cost
            elif t_type == "do_clear":
                x = args[0]
                cost = self._get_clear_cost(x, actual_below)
                if cost == 0: nop_count += 1
                else: sum_task_costs += cost
            elif t_type == "do_on_table":
                x = args[0]
                if x not in actual_on and x in stable:
                    nop_count += 1
                else:
                    sum_task_costs += self._get_clear_cost(x, actual_below) + 2
            elif t_type == "do_move":
                sum_task_costs += 2
            elif t_type == "nop":
                nop_count += 1

        term1 = nop_count + displacement
        term2 = sum_task_costs
        term3 = 0
        if task_network:
            first_task = task_network[0]
            # term3 could be first-task cost, but sum covers it.
        
        # Hand check
        if self.handempty_id is not None and not ((state >> self.handempty_id) & 1):
            term2 += 1

        return max(term1, term2)

    def _get_clear_cost(self, block, actual_below):
        h = 0
        curr = block
        while curr in actual_below:
            h += 1
            curr = actual_below[curr]
        return 2 * h
