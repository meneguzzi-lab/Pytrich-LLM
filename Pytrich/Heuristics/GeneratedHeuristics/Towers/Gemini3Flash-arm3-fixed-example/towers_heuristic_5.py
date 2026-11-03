import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Bottleneck Analysis:
The bottleneck in the Towers domain is the recursive dependency of moving rings: 
to move a larger ring, all smaller rings above it must be moved to an auxiliary tower. 
Since only the top ring can be moved at a time (towerTop predicate), this serializes all work.

Admissible Lower Bounds:
1. Misplaced Rings Bound: Each ring not currently on its goal support (or on a support that itself needs to move) 
   requires at least one 'move' action. Since every 'move' has cost 1, the count of misplaced rings is an 
   admissible lower bound.
2. Task Network Bound: Every 'move_abstract' task in the network will eventually result in at least one 'move' action. 
   Every 'rotateTower' or 'exchange' task implies at least one 'move_abstract'. Summing minimum costs 
   per task type provides a lower bound on future actions.
"""

class TowersGemini3FlashHeuristic5(Heuristic):
    def __init__(self, name="towers_gemini3flash_heuristic_5"):
        super().__init__(name=name)
        self.on_facts = {}  # (ring, object) -> global_id
        self.goal_on_facts = {}  # ring -> target_object
        self.ring_list = []
        self.tower_list = []
        self.is_initialized = False

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
        
        # Goal facts are constant in this domain (goal_on is often in init or goal)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        
        # Parse all facts for efficient bitwise lookup
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_facts[(args[0], args[1])] = fact.global_id
                if args[0] not in self.ring_list: self.ring_list.append(args[0])
            elif pred == "goal_on":
                # Some domains put goals in goal_on predicate in initial state
                self.goal_on_facts[args[0]] = args[1]
            elif pred == "smallerThan":
                if args[0] not in self.ring_list: self.ring_list.append(args[0])
        
        # If goal_on wasn't in predicates, parse from model.goals
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on_facts[args[0]] = args[1]

        self.is_initialized = True
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_misplaced_count(self, state_bitwise):
        """
        A ring is misplaced if:
        1. It is not on its goal support.
        2. Its current support is itself misplaced (only if support is a ring).
        """
        misplaced = set()
        # Sort rings roughly by size (largest first) if possible to propagate misplaced status
        # In Hanoi, larger rings are supports for smaller rings.
        # We iterate and check stability.
        
        # Simple admissible version: Ring is misplaced if not on goal support
        count = 0
        for ring, target in self.goal_on_facts.items():
            fact_id = self.on_facts.get((ring, target))
            if fact_id is None or not ((state_bitwise >> fact_id) & 1):
                count += 1
        return count

    def _get_task_bound(self, task_network):
        """
        Estimates minimum remaining moves based on tasks in the network.
        """
        bound = 0
        for task in task_network:
            name = task.name.lower()
            if "move_abstract" in name or "move" == name:
                bound += 1
            elif "exchange" in name:
                # Exchange LR/RL involves at least one move_abstract and one rotateTower
                bound += 1
            elif "rotatetower" in name:
                # Rotate involves move_abstract and exchange
                bound += 1
            elif "shifttower" in name or "selectdirection" in name:
                # High level tasks representing the whole tower move
                # We return a minimal 1 to keep it admissible but guided
                bound += 1
        return bound

    def _compute(self, state, task_network):
        if not self.is_initialized:
            return 0
        
        # Term 1: State-based misplaced rings (Admissible)
        t1 = self._get_misplaced_count(state)
        
        # Term 2: Task-based progress (Admissible lower bound on work in network)
        t2 = self._get_task_bound(task_network)
        
        # Use max to maintain admissibility
        return max(t1, t2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
