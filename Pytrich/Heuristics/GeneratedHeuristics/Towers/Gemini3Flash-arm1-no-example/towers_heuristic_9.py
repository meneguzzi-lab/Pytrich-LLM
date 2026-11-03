from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The bottleneck in the Towers domain is the recursive dependency of ring placement. 
# To move a larger ring, all smaller rings must be removed from its source and destination. 
# This creates a serialized bottleneck where the largest misplaced ring dictates the progress.

# Admissibility Sketches:
# 1. h_state (Misplaced/Unstable Rings): 
#    If a ring is not at its goal position, or if it sits on a ring that must eventually move, 
#    then that ring must be moved at least once. Counting these 'unstable' rings provides a 
#    strict lower bound on the remaining move actions because each move only affects one ring.
# 2. h_tasks (Planned Work): 
#    The number of 'move_abstract' or 'move' actions already in the task network represents 
#    mandatory future costs. Each such task corresponds to at least one action.
# Final h = max(h_state, h_tasks) + tie_breakers.

class TowersGemini3FlashHeuristic9(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_9="towers_gemini3flash_heuristic_9"):
        super().__init__(towers_gemini3flash_heuristic_9)
        self.on_ids = {} # (ring, obj) -> global_id
        self.goal_map = {} # ring -> target_obj
        self.sorted_rings = [] # Rings sorted from largest to smallest
        self.smaller_than_map = {} # ring -> set of smaller rings

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
        # 1. Map predicates
        all_rings = set()
        smaller_facts = []
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == "smallerThan":
                smaller_facts.append(args)
                # Only track objects that are rings in smallerThan(r, something)
                # HDDL types define types, but parsing args is sufficient here
            
        # 2. Parse goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_map[args[0]] = args[1]
                all_rings.add(args[0])

        # 3. Determine ring sizes and sort them
        # Size of a ring = number of rings smaller than it
        ring_to_smaller = {r: set() for r in all_rings}
        for r_small, r_large in smaller_facts:
            if r_small in all_rings and r_large in all_rings:
                ring_to_smaller[r_large].add(r_small)
        
        # Transitive closure for smallerThan to ensure correct 'unstable' propagation
        changed = True
        while changed:
            changed = False
            for r in all_rings:
                curr_size = len(ring_to_smaller[r])
                for child in list(ring_to_smaller[r]):
                    ring_to_smaller[r].update(ring_to_smaller[child])
                if len(ring_to_smaller[r]) > curr_size:
                    changed = True
        
        self.smaller_than_map = ring_to_smaller
        self.sorted_rings = sorted(list(all_rings), key=lambda r: len(ring_to_smaller[r]), reverse=True)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # 1. State term: Count unstable rings
        unstable = set()
        stable_rings = set()
        
        # Process largest to smallest
        for r in self.sorted_rings:
            target = self.goal_map.get(r)
            if target is None:
                continue
            
            # Check if r is on target
            fact_id = self.on_ids.get((r, target))
            is_on_target = (state >> fact_id) & 1 if fact_id is not None else False
            
            # A ring is stable only if it is on target AND its target is stable
            # (Towers are always stable; rings are stable if they are in stable_rings)
            target_is_stable = (target not in self.sorted_rings) or (target in stable_rings)
            
            if is_on_target and target_is_stable:
                stable_rings.add(r)
            else:
                unstable.add(r)
                # If a ring is unstable, all smaller rings must move to clear it
                unstable.update(self.smaller_than_map[r])

        h_state = len(unstable)

        # 2. Task term: Count unavoidable moves in the network
        h_tasks = 0
        for task in task_network:
            # If it's an operator, check if it's 'move'
            if task.global_id <= self.model.iop_end and task.global_id >= self.model.iop_init:
                if "move" in task.name: h_tasks += 1
            else:
                # Abstract tasks
                tname = task.name.lower()
                if "move_abstract" in tname:
                    h_tasks += 1
                elif "shifttower" in tname or "rotatetower" in tname:
                    h_tasks += 1

        # Tie-breaker for task network depth
        tie_breaker = 0.001 * len(task_network)

        return max(h_state, h_tasks) + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
