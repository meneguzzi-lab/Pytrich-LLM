import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic13(Heuristic):
    """
    Domain-Dependent Heuristic for the Lamps (Grid Flip) domain.

    1. Bottleneck Analysis:
       The serial constraint in this domain is the number of 'flip' operations required to 
       reconfigure the lamps into the goal status. Every lamp that currently has a status 
       different from its target status MUST be flipped at least once. Since 'flip' is an 
       action with cost 1, the number of mismatches provides a base admissible lower bound.

    2. Admissible Lower Bounds:
       - Term 1 (Hamming Distance): Count of lamps (x, y) where the current status does not match 
          the goal status. Each mismatch requires at least one 'flip' action (cost 1).
       - Term 2 (Task Network Cost): Minimum cost to complete the tasks currently in the network. 
         Each primitive action (including 'flip' and '__method_precondition_*') has cost 1. 
         Each 'turn' abstract task will decompose into at least one 'flip'.

    The final heuristic returns max(Hamming Distance, Task Network Cost), which is admissible 
    as both terms are independent lower bounds on the remaining work.
    """

    def __init__(self, lamps_gemini3flash_heuristic_13=None, name="LampsGemini3FlashHeuristic13"):
        super().__init__(name=name)
        self.on_fact_ids = {}  # (x, y, status) -> global_id
        self.goal_map = {}     # (x, y) -> goal_status_string

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once at the start of search. Precomputes goal map and fact lookups.
        """
        self.model = model
        
        # 1. Map (x, y, status) to fact global IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 3:
                # args: [x, y, status]
                self.on_fact_ids[(args[0], args[1], args[2])] = fact.global_id

        # 2. Extract goal lamp statuses from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    self.goal_map[(x, y)] = s

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _parse_fact_name(self, name: str):
        """
        Parses grounded fact name like +on[l0,l0,Y] into (predicate, args).
        """
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

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        """
        Calculates the max of Hamming distance and task network minimum cost.
        """
        # Term 1: Hamming Distance (Mismatches)
        # For every goal 'on[x, y, s]', check if it's currently true.
        mismatches = 0
        for (x, y), target_status in self.goal_map.items():
            fact_id = self.on_fact_ids.get((x, y, target_status))
            if fact_id is not None:
                # If the goal fact is NOT true, the lamp must be flipped at least once
                is_true = (state_bitwise >> fact_id) & 1
                if not is_true:
                    mismatches += 1

        # Term 2: Task Network Cost
        # Primitive actions cost 1. Turn tasks will eventually cost at least 1 (the initial flip).
        tn_cost = 0
        for task in task_network:
            if self.model.iop_init <= task.global_id <= self.model.iop_end:
                # It's an operator (flip, __method_precondition_*, or no-op)
                # Pytrich models usually have operator costs of 1.
                # We'll use 1 explicitly as per domain definition.
                tn_cost += 1
            else:
                # It's an abstract task. 'turn' always produces a 'flip'.
                # 'play' and 'propagate' might end without cost.
                if "turn" in task.name.lower():
                    tn_cost += 1

        # Use max for admissibility
        h = float(max(mismatches, tn_cost))
        
        # Tiny tie-breaker for network length to encourage clearing subtasks
        h += 0.0001 * len(task_network)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Heuristic evaluation called for every node expansion.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic13(goals={len(self.goal_map)})"
