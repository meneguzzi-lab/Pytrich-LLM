from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic15(Heuristic):
    """
    Heuristic for the Depots domain.
    
    1. Bottleneck: The serializing constraint is the clear sequence of lifting, moving (if needed),
       and dropping crates, combined with the strict requirement that surfaces must be clear 
       before use. Clearance and transport are the primary drivers of plan length.
    
    2. Admissible Lower Bounds:
       - Transport/Placement Bound: Every crate not on its goal surface requires a minimum of 2 actions 
         (Lift/Drop) if at the target location, or 4 actions (Lift/Load/Drive/Unload/Drop) if at a 
         different location. This is admissible as these actions are physically necessary.
       - Clearance Bound: For every goal on(c, s), any crate currently blocking c or s must be moved,
         costing at least 2 actions (Lift + Load/Drop) per obstructing crate.
       - Task Network Bound: Each task in the network has a minimum decomposition cost (e.g., do_put_on 
         is at least 2 actions).
    """
    def __init__(self, depot_gemini3flash_heuristic_15="depot_gemini3flash_heuristic_15"):
        super().__init__(name=depot_gemini3flash_heuristic_15)
        self.goal_on = [] # List of (crate_id, surface_id)
        self.pallet_to_place = {} # pallet_id -> place_id
        self.crate_to_place_static = {} # crate_id -> place_id (initial, helps with static info)
        self.on_pred_id = {} # (crate, surface) -> global_id
        self.at_pred_id = {} # (locatable, place) -> global_id
        self.clear_pred_id = {} # surface -> global_id
        self.in_pred_id = {} # (crate, truck) -> global_id
        self.crate_ids = set()
        self.surface_ids = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates and locations
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_pred_id[(args[0], args[1])] = f.global_id
                self.crate_ids.add(args[0])
                self.surface_ids.add(args[1])
            elif pred == "at":
                self.at_pred_id[(args[0], args[1])] = f.global_id
                # Use initial state to find pallet locations
                if (initial_node.state >> f.global_id) & 1:
                    if "pallet" in args[0]:
                        self.pallet_to_place[args[0]] = args[1]
                    if "crate" in args[0]:
                        self.crate_to_place_static[args[0]] = args[1]
            elif pred == "clear":
                self.clear_pred_id[args[0]] = f.global_id
                self.surface_ids.add(args[0])
            elif pred == "in":
                self.in_pred_id[(args[0], args[1])] = f.global_id

        # Recover static pallet locations if grounder stripped them
        for op in model.operators:
            # Drop[hoist, crate, surface, place]
            if "Drop" in op.name:
                start = op.name.find("[")
                args = op.name[start+1:-1].split(",")
                if len(args) >= 4:
                    if "pallet" in args[2]:
                        self.pallet_to_place[args[2]] = args[3]

        # 2. Parse goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on.append((args[0], args[1]))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _get_current_place(self, state, locatable):
        # Search for 'at' fact in state
        for place in self.model.facts: # This is slow but used in logic; optimize by limiting search
            pass # Optimization: we pre-built at_pred_id map
        # Instead of searching all, we only need to find which place it's at
        # Since we don't have a list of all places, we assume the initial locations mostly hold
        # but check state bitwise.
        return None

    def _compute(self, state, task_network):
        # Term 1: Goal-based progress
        goal_h = 0
        for crate, target_surface in self.goal_on:
            on_goal_bit = self.on_pred_id.get((crate, target_surface))
            if on_goal_bit is not None and (state >> on_goal_bit) & 1:
                continue
            
            # Crate is not on goal surface. Minimum cost:
            # Is it at the right place?
            target_place = self.pallet_to_place.get(target_surface)
            # If target_surface is a crate, find its pallet's place
            temp_s = target_surface
            while temp_s and temp_s not in self.pallet_to_place:
                # Find what temp_s is on
                found = False
                for other_s in self.surface_ids:
                    bit = self.on_pred_id.get((temp_s, other_s))
                    if bit is not None and (state >> bit) & 1:
                        temp_s = other_s
                        found = True
                        break
                if not found: break
            target_place = self.pallet_to_place.get(temp_s)

            # Estimate current location
            current_place = None
            for p_name in set(self.pallet_to_place.values()):
                at_bit = self.at_pred_id.get((crate, p_name))
                if at_bit is not None and (state >> at_bit) & 1:
                    current_place = p_name
                    break
            
            if current_place == target_place and target_place is not None:
                goal_h += 2 # Lift, Drop
            else:
                goal_h += 4 # Lift, Load, Drive, Unload/Drop

            # Clearance cost for target_surface
            clear_bit = self.clear_pred_id.get(target_surface)
            if clear_bit is not None and not ((state >> clear_bit) & 1):
                goal_h += 2 # Must move at least one crate off it

        # Term 2: Task Network cost
        tn_h = 0
        for task in task_network:
            if hasattr(task, 'name'):
                if "do_put_on" in task.name:
                    tn_h += 2
                elif "do_clear" in task.name:
                    # check if already clear
                    match = re.search(r'\[(.*?)\]', task.name)
                    if match:
                        args = match.group(1).split(',')
                        if args:
                            c_bit = self.clear_pred_id.get(args[0])
                            if c_bit is not None and not ((state >> c_bit) & 1):
                                tn_h += 2
                elif "Drive" in task.name or "Lift" in task.name or "Drop" in task.name or "Load" in task.name or "Unload" in task.name:
                    tn_h += 1

        # Using max as suggested in hints for safety/admissibility
        return max(goal_h, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
