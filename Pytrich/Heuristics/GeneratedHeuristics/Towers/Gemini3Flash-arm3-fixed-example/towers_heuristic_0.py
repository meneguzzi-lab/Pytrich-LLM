from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

# Bottleneck Analysis:
# The primary bottleneck in the Towers (Tower of Hanoi) domain is the serial constraint of the rings.
# A ring 'r' cannot be moved unless it is the top of its current stack (towerTop) and the destination 
# object is also a top (towerTop) and satisfies the smallerThan[r, dest] constraint. This serializes 
# work such that the largest misplaced rings must be addressed by moving all smaller rings above 
# them to a buffer tower first.

# Admissible Lower Bounds:
# 1. Misplaced Ring Bound: A ring is misplaced if its current position does not match its goal position,
#    or if the object it is resting on is itself misplaced. Each misplaced ring requires at least 
#    one 'move' action. Admissibility: Moving a ring once is the absolute minimum to reach its goal state.
# 2. Task Network Bound: Every 'move_abstract' task in the task network corresponds to exactly one 
#    physical 'move' action. Other compound tasks like 'rotateTower' represent decompositions that 
#    must eventually produce 'move' actions. Counting these provides a lower bound on remaining effort.

class TowersGemini3FlashHeuristic0(Heuristic):
    def __init__(self, name="towers_gemini3flash_heuristic_0"):
        super().__init__(name=name)
        self.on_ids = {} # (ring, obj) -> global_id
        self.ring_to_goal_obj = {}
        self.ordered_rings_largest_first = []
        self.all_rings = set()
        self.move_abstract_ids = set()
        self.rotate_tower_ids = set()
        self.exchange_ids = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse all facts for mapping
        ring_smaller_counts = {}
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "smallerThan":
                r = args[0]
                # Count how many objects this ring is smaller than to determine size
                ring_smaller_counts[r] = ring_smaller_counts.get(r, 0) + 1
                self.all_rings.add(r)
        
        # 2. Determine ring order (Largest to Smallest)
        # A larger ring is smaller than fewer objects (only towers or larger rings)
        self.ordered_rings_largest_first = sorted(self.all_rings, key=lambda r: ring_smaller_counts.get(r, 0))

        # 3. Precompute goal structure
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.ring_to_goal_obj[args[0]] = args[1]

        # 4. Identify relevant task IDs
        for task in model.abstract_tasks:
            name_lower = task.name.lower()
            if "move_abstract" in name_lower:
                self.move_abstract_ids.add(task.global_id)
            elif "rotate" in name_lower:
                self.rotate_tower_ids.add(task.global_id)
            elif "exchange" in name_lower:
                self.exchange_ids.add(task.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _compute(self, state, task_network):
        # Term 1: Misplaced Ring Count (Admissible Hanoi bound)
        h_goal = 0
        is_misplaced = {}
        for r in self.ordered_rings_largest_first:
            target = self.ring_to_goal_obj.get(r)
            if not target:
                is_misplaced[r] = False
                continue
            
            on_goal_fact_id = self.on_ids.get((r, target))
            
            # Misplaced if not on target OR the object below it is misplaced
            if on_goal_fact_id is None or not ((state >> on_goal_fact_id) & 1) or \
               (target in is_misplaced and is_misplaced[target]):
                h_goal += 1
                is_misplaced[r] = True
            else:
                is_misplaced[r] = False

        # Term 2: Task Network Complexity (Admissible lookahead)
        h_tn = 0
        for task in task_network:
            tid = task.global_id
            if tid in self.move_abstract_ids:
                h_tn += 1
            elif tid in self.rotate_tower_ids:
                h_tn += 1
            elif tid in self.exchange_ids:
                h_tn += 1
            # shiftTower and selectDirection are treated as 0-cost structural tasks

        return max(h_goal, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
