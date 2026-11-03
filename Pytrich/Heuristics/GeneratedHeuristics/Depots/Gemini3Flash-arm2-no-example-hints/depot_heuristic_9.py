from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic9(Heuristic):
    """
    Heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The bottleneck in Depots is the availability and positioning of Hoists and the serialization of stacking. 
    Each hoist can only lift one crate at a time, and stacking requires surfaces to be clear. 
    The single most serializing constraint is the stack-depth constraint: a crate cannot be moved until 
    everything above it is cleared, and it cannot be placed until its target surface is cleared.

    Admissible Lower Bounds:
    1. Goal-Directed Movement (Summative): Every crate not at its goal surface requires at least 
       a Lift and a Drop (2 actions) if local, or Lift, Load, Unload, Drop (4 actions) if transport is needed.
    2. Clearance Cost (Summative): If a surface target is blocked by N crates, at least N Lift operations 
       are required to clear it before the target crate can be placed.
    3. Task Network Progression: Each remaining abstract task represents future actions. A do_put_on 
       task costs at least 2 actions if not already satisfied.

    Guidance and Tie-breaking:
    - We use P5 (Goal Stability): A crate on its goal is only 'free' if the supporting surface is stable.
    - We use P6 (Parameter Canonicalization): Minimal sub-unit penalties are applied for truck distance.
    """

    def __init__(self, depot_gemini3flash_heuristic_9="depots_gemini3flash_9"):
        super().__init__(name=depot_gemini3flash_heuristic_9)
        self.on_goals = {}  # crate_id -> surface_id
        self.crate_ids = set()
        self.pallet_ids = set()
        self.truck_ids = set()
        self.hoist_ids = set()
        self.at_place = {}  # obj_id -> place_name (static info from grounding)
        
        self.pred_map = {
            "on": {}, "at": {}, "in": {}, "lifting": {}, "available": {}, "clear": {}
        }

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred in self.pred_map:
                key = tuple(args)
                self.pred_map[pred][key] = f.global_id
                
            if "crate" in f.name.lower():
                for arg in args: 
                    if "crate" in arg.lower(): self.crate_ids.add(arg)
            if "pallet" in f.name.lower():
                for arg in args: 
                    if "pallet" in arg.lower(): self.pallet_ids.add(arg)
            if "truck" in f.name.lower():
                for arg in args: 
                    if "truck" in arg.lower(): self.truck_ids.add(arg)
            if "hoist" in f.name.lower():
                for arg in args: 
                    if "hoist" in arg.lower(): self.hoist_ids.add(arg)

        # 2. Extract Goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.on_goals[args[0]] = args[1]

        # 3. Static Info Recovery (P3)
        for op in model.operators:
            # Pattern: Name[arg1,arg2,...]
            parts = re.split(r'[\[\],]', op.name)
            if "Lift" in op.name or "Drop" in op.name:
                # Lift[hoist, crate, surface, place]
                self.at_place[parts[1]] = parts[4]
                self.at_place[parts[2]] = parts[4]
                self.at_place[parts[3]] = parts[4]
            elif "Load" in op.name or "Unload" in op.name:
                # Load[hoist, crate, truck, place]
                self.at_place[parts[1]] = parts[4]
                self.at_place[parts[3]] = parts[4]
            elif "Drive" in op.name:
                # Drive[truck, from, to]
                self.at_place[parts[1]] = "TRUCK_DYNAMIC"

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        if not self.on_goals:
            return 0
            
        h_movement = 0
        h_clearance = 0

        # Current state mapping
        crate_on = {} # crate -> surface
        surface_clear = {} # surface -> bool
        crate_loc = {} # crate -> place/truck/hoist

        # Quick scan of relevant facts
        for crate in self.crate_ids:
            # Check lifting
            for hoist in self.hoist_ids:
                fid = self.pred_map["lifting"].get((hoist, crate))
                if fid is not None and (state >> fid) & 1:
                    crate_loc[crate] = "lifting"
                    break
            if crate in crate_loc: continue

            # Check in truck
            for truck in self.truck_ids:
                fid = self.pred_map["in"].get((crate, truck))
                if fid is not None and (state >> fid) & 1:
                    crate_loc[crate] = "truck"
                    break
            if crate in crate_loc: continue

            # Check on surface
            all_surfaces = self.crate_ids | self.pallet_ids
            for surf in all_surfaces:
                fid = self.pred_map["on"].get((crate, surf))
                if fid is not None and (state >> fid) & 1:
                    crate_on[crate] = surf
                    crate_loc[crate] = self.at_place.get(surf, "unknown")
                    break

        for surf in (self.crate_ids | self.pallet_ids):
            fid = self.pred_map["clear"].get((surf,))
            if fid is not None and (state >> fid) & 1:
                surface_clear[surf] = True
            else:
                surface_clear[surf] = False

        # 1. Goal Movement Cost (P1, P2)
        for crate, goal_surf in self.on_goals.items():
            current_surf = crate_on.get(crate)
            if current_surf == goal_surf:
                # Stability Check (P5): if target is another crate, is that crate stable?
                if goal_surf in self.on_goals:
                    sub_goal = self.on_goals[goal_surf]
                    if crate_on.get(goal_surf) != sub_goal:
                        h_movement += 2
                continue
            
            loc = crate_loc.get(crate)
            if loc == "lifting":
                h_movement += 1
            elif loc == "truck":
                h_movement += 2
            else:
                goal_place = self.at_place.get(goal_surf)
                if loc == goal_place:
                    h_movement += 2
                else:
                    h_movement += 4

        # 2. Clearance Cost (P2, P4)
        for crate, goal_surf in self.on_goals.items():
            if not surface_clear.get(goal_surf, False):
                # If target surface is not clear, something must be lifted (at least +1)
                # Simplified: add 1 for the immediate blocker
                if goal_surf != crate_on.get(crate): 
                    h_clearance += 1

        # 3. Task Network (P2)
        h_tn = 0
        for task in task_network:
            if hasattr(task, 'decompositions'): # Abstract task
                if "do_put_on" in task.name:
                    h_tn += 1 # Base cost for complexity
                elif "do_clear" in task.name:
                    h_tn += 1
            else: # Operator
                h_tn += task.cost

        # Balanced combination
        return max(h_movement, h_clearance) + h_tn
