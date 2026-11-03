"""
Domain-dependent heuristic for the Depots domain.

1. Bottleneck Analysis:
The primary bottleneck in the Depots domain is the stacking constraint on crates and the logistical requirement of using trucks to move crates between locations. Crates must be moved from their initial stack to their goal stack, often requiring clearance of multiple obstacles and transport through a limited set of trucks and hoists.

2. Admissible Lower Bounds:
- Term 1: Crate Displacement Cost. Any crate not on its goal surface must be lifted and dropped, requiring at least 2 actions. This is a conservative lower bound.
- Term 2: Logistic Transportation Cost. If a crate is at place A and its goal surface is at place B (where A != B), the crate must be loaded into a truck and unloaded from a truck at the destination, adding at least 2 actions to its plan.
- Term 3: Structural Clearance Cost. If a crate 'C' needs to be moved, or a surface 'S' needs to be clear to receive a crate, any crate 'C_above' currently on top of 'C' or 'S' must be moved first, adding at least 2 actions (Lift/Drop or Lift/Load) per blocking crate.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic3(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_3=None, name="DepotsGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.goal_on = {}           # crate_name -> target_surface_name
        self.surface_places = {}     # surface_name -> place_name (pallets are static)
        self.goal_place = {}         # crate_name -> target_place_name
        
        self.at_ids = {}             # (locatable, place) -> fact_id
        self.on_ids = {}             # (crate, surface) -> fact_id
        self.clear_ids = {}          # surface -> fact_id
        self.all_crates = []
        self.all_places = []
        self.all_surfaces = []

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # 1. Map all facts for fast lookup
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                self.at_ids[(args[0], args[1])] = fact.global_id
                if args[1] not in self.all_places: self.all_places.append(args[1])
            elif pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
                if args[0] not in self.all_crates: self.all_crates.append(args[0])
                if args[1] not in self.all_surfaces: self.all_surfaces.append(args[1])
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id

        # 2. Identify static locations of pallets and goal surfaces
        # Note: In Depots, pallets are stationary at places.
        initial_state = initial_node.state
        for (loc, p), fid in self.at_ids.items():
            if (initial_state >> fid) & 1:
                self.surface_places[loc] = p

        # 3. Parse Goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                crate, target_s = args[0], args[1]
                self.goal_on[crate] = target_s
                
                # Trace target stack to find target place
                curr = target_s
                visited = set()
                while curr not in self.surface_places and curr in self.goal_on and curr not in visited:
                    visited.add(curr)
                    curr = self.goal_on[curr]
                
                if curr in self.surface_places:
                    self.goal_place[crate] = self.surface_places[curr]

        self.preprocessing_time = time.time() - start_time
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
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            return parts[0], parts[1:]
        return None, []

    def _compute(self, state: int, task_network) -> int:
        h = 0
        # Admissible Term 1 & 2: Goal Displacement and Logistics
        for crate, target_s in self.goal_on.items():
            # Check if current goal 'on' is satisfied
            goal_fid = self.on_ids.get((crate, target_s))
            if goal_fid is not None and not ((state >> goal_fid) & 1):
                h += 2 # Min cost: Lift + Drop
                
                # Logistic check: Current location vs Goal location
                target_p = self.goal_place.get(crate)
                current_p = None
                for p in self.all_places:
                    at_fid = self.at_ids.get((crate, p))
                    if at_fid is not None and (state >> at_fid) & 1:
                        current_p = p
                        break
                
                if current_p and target_p and current_p != target_p:
                    h += 2 # Extra cost: Load + Unload (Drive is 1, but 2 is safe)

        # Admissible Term 3: Clearance (Simple check for immediate blockers)
        # If a crate must move, and it's not clear, add 2 for the blocker.
        for crate in self.goal_on:
            goal_fid = self.on_ids.get((crate, self.goal_on[crate]))
            if goal_fid is not None and not ((state >> goal_fid) & 1):
                # Must move crate. Check if clear.
                clear_fid = self.clear_ids.get(crate)
                if clear_fid is not None and not ((state >> clear_fid) & 1):
                    h += 2 # At least one Lift needed for a crate above it
                
                # Check if target surface is clear
                target_s = self.goal_on[crate]
                target_clear_fid = self.clear_ids.get(target_s)
                if target_clear_fid is not None and not ((state >> target_clear_fid) & 1):
                    h += 2 # Something is blocking the spot where we want to put it

        # Tie-breaker: penalized deeper task networks to guide towards primitives
        h += len(task_network) * 0.1

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
