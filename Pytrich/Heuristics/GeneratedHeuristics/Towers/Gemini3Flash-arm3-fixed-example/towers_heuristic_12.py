import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The bottleneck in the Towers (Tower of Hanoi) domain is the recursive ordering constraint.
# Moving a larger ring requires all smaller rings currently above it to be moved to an auxiliary tower.
# The heuristic must prioritize clearing the path for the largest misplaced ring to reach its goal destination.

# Admissible Lower Bounds:
# 1. Misplaced Rings (H1): A ring is misplaced if it is not on its goal object, or if its goal object
#    is another ring that is itself misplaced. Each misplaced ring must be moved at least once. 
#    Summing these is admissible as each move action only shifts one ring.
# 2. Task Network Pressure (H2): Each 'move' or 'move_abstract' task in the current task network 
#    represents at least one unavoidable action in the resulting plan.

class TowersGemini3FlashHeuristic12(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_12=None, name="towers_gemini3flash_heuristic_12"):
        super().__init__(name=name)
        self.on_facts = {} # (ring, object) -> global_id
        self.ring_goal_on = {} # ring -> goal_object
        self.rings = set()
        self.sorted_rings = []
        self.move_task_ids = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse facts and identify rings
        smaller_than_pairs = []
        all_objs = set()
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_facts[(args[0], args[1])] = fact.global_id
                all_objs.add(args[0])
                all_objs.add(args[1])
            elif pred == "smallerThan":
                smaller_than_pairs.append((args[0], args[1]))
            elif pred == "towerTop":
                all_objs.add(args[0])
                all_objs.add(args[1])

        # 2. Identify Rings (anything that is smallerThan something is a ring)
        self.rings = set(r for r, _ in smaller_than_pairs)
        
        # 3. Sort Rings by size (Largest to Smallest)
        # Count how many objects each ring is smaller than.
        # r1 is smaller than all others (high count), r20 is smaller than none (low count).
        smaller_counts = {r: 0 for r in self.rings}
        for r_sub, r_obj in smaller_than_pairs:
            if r_sub in self.rings:
                smaller_counts[r_sub] += 1
        
        # Sort: largest first (count 0) to smallest (high count)
        self.sorted_rings = sorted(list(self.rings), key=lambda r: smaller_counts[r])

        # 4. Extract Goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.ring_goal_on[args[0]] = args[1]

        # 5. Identify move-related tasks
        for i, task in enumerate(model.abstract_tasks):
            if "move" in task.name.lower():
                self.move_task_ids.add(task.global_id)
        for i, op in enumerate(model.operators):
            if "move" in op.name.lower():
                self.move_task_ids.add(op.global_id)

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

    def _compute(self, state_bitwise, task_network):
        # Term 1: Misplaced Rings (Admissible Lower Bound)
        # A ring is misplaced if its 'on' property is wrong OR if the object it is on is misplaced.
        # We iterate from Largest to Smallest to propagate misplaced status upwards.
        is_r_misplaced = {}
        misplaced_count = 0
        
        for r in self.sorted_rings:
            target = self.ring_goal_on.get(r)
            if not target:
                is_r_misplaced[r] = False
                continue
            
            # Check if r is currently on its goal target
            on_goal_id = self.on_facts.get((r, target))
            is_on_target = (state_bitwise >> on_goal_id) & 1 if on_goal_id is not None else 0
            
            if not is_on_target:
                is_r_misplaced[r] = True
            else:
                # On correct target, but is that target (if it's a ring) misplaced?
                if target in is_r_misplaced and is_r_misplaced[target]:
                    is_r_misplaced[r] = True
                else:
                    is_r_misplaced[r] = False
            
            if is_r_misplaced[r]:
                misplaced_count += 1

        # Term 2: Task Network count (Tie-breaker/Guidance)
        # Count move-related tasks in the network
        move_tasks_in_network = 0
        for task in task_network:
            if task.global_id in self.move_task_ids:
                move_tasks_in_network += 1

        # Combine terms: Primary cost (misplaced rings) + tie-breaking guidance
        return misplaced_count + (0.01 * move_tasks_in_network) + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value) if h_value > 1000000 else h_value
