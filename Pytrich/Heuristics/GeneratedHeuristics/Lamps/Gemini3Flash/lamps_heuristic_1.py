"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
Every lamp that is not in its goal status must be flipped. The only primitive action that changes status
is 'flip'. Thus, the minimum number of actions required is at least the number of lamp status mismatches (Hamming distance).

Admissibility Sketches:
1. Lamp Mismatches (Term 1): Each 'flip' action changes exactly one lamp's status and costs 1. Therefore,
   the number of coordinates (x, y) where the current status != goal status is an admissible lower bound.
2. Task Network Cost (Term 2): Each task in the network must be decomposed into primitive actions.
   A 'turn' task always decomposes into one 'flip' and four 'propagate' calls. Each 'propagate' call
   selects a method, and every method in this domain has a precondition which Pytrich compiles into 
    a cost-1 primitive action. Thus, 'turn' has a min-cost of 1+4=5, and 'propagate' has a min-cost of 1.
3. Combination: max(Term 1, Term 2) is admissible because both terms are independent lower bounds.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask
import time

class LampsGemini3FlashHeuristic1(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_1=None, name="LampsGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.goal_map = {}  # (x, y) -> status
        self.on_facts = {}  # (x, y, status) -> global_id
        self.coords = []    # list of (x, y) tuples
        self.min_abstract_cost = {
            'turn': 5,             # flip (1) + 4 * propagate checks (4)
            'propagate_up': 1,    # method precondition check
            'propagate_down': 1, 
            'propagate_left': 1,
            'propagate_right': 1,
            'play': 0             # can terminate via end_game
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map goal status for each coordinate
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_names:
                pred, args = self._parse_fact_name(name)
                if pred == "on" and len(args) == 3:
                    # on(x, y, status)
                    self.goal_map[(args[0], args[1])] = args[2]

        # 2. Map all grounded 'on' facts to their global IDs
        temp_coords = set()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts[(x, y, s)] = fact.global_id
                temp_coords.add((x, y))
        
        self.coords = list(temp_coords)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 1: Hamming distance (mismatches)
        mismatches = 0
        for (x, y) in self.coords:
            goal_s = self.goal_map.get((x, y))
            if goal_s:
                # Check if current status matches goal status
                fact_id = self.on_facts.get((x, y, goal_s))
                if fact_id is not None:
                    is_at_goal = (state >> fact_id) & 1
                    if not is_at_goal:
                        mismatches += 1

        # Term 2: Task network minimum cost
        tn_cost = 0
        for task in task_network:
            if isinstance(task, Operator):
                tn_cost += task.cost
            elif isinstance(task, AbstractTask):
                # Extract base task name
                base_name = task.name.split('[')[0] if '[' in task.name else task.name
                tn_cost += self.min_abstract_cost.get(base_name, 0)

        return float(max(mismatches, tn_cost))

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
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts:
                return parts[0], parts[1:]
        return None, []"
