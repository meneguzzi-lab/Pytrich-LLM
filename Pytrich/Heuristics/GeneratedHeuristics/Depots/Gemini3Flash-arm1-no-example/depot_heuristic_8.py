from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The primary bottleneck in the Depots domain is the coordination of hoists and trucks to move crates across locations 
# while managing stack constraints (clear surfaces). Transporting crates between places is significantly more 
# expensive than moving them within the same place.
#
# Admissibility Sketches:
# 1. Goal satisfaction (h_on): Each unfulfilled 'on(crate, surface)' goal requires at least one 'Drop' action (cost 1).
# 2. Logistics cost (h_loc): If a crate is at a different place than its target surface, it must undergo a 
#    Lift-Load-Drive-Unload-Drop sequence (minimum cost 5 actions). If it is at the same place but not on its goal 
#    surface, it requires at least Lift-Drop (cost 2). 
# 3. Task network size: Each non-trivial task in the network must be decomposed into at least one action (cost 1).

class DepotsGemini3FlashHeuristic8(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_8="DepotsGemini3FlashHeuristic8"):
        super().__init__(depot_gemini3flash_heuristic_8)
        self.at_map = {}  # (loc_str, place_str) -> global_id
        self.on_map = {}  # (crate_str, surf_str) -> global_id
        self.goal_info = []  # List of (crate_str, surface_str, target_place_str)

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
        # Map facts for fast bitwise access
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_map[(args[0], args[1])] = f.global_id
            elif pred == "on":
                self.on_map[(args[0], args[1])] = f.global_id

        # Identify initial locations of all locatables (to find pallets and places)
        init_at = {}
        init_facts = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "at":
                init_at[args[0]] = args[1]

        # Parse goals and identify target places for crates
        goal_on = {}
        goal_facts = model.state_explicit_repr(model.goals)
        for g_fact in goal_facts:
            pred, args = self._parse_fact_name(g_fact)
            if pred == "on":
                goal_on[args[0]] = args[1]

        # Determine the target place for every crate in a goal stack
        target_place = {}
        changed = True
        while changed:
            changed = False
            for crate, surface in goal_on.items():
                if crate in target_place: continue
                if surface in init_at and surface not in goal_on:
                    # Surface is a fixed base (pallet) or not moving
                    target_place[crate] = init_at[surface]
                    changed = True
                elif surface in target_place:
                    # Surface is a crate whose target location is known
                    target_place[crate] = target_place[surface]
                    changed = True

        # Build final goal info list
        for crate, surface in goal_on.items():
            tp = target_place.get(crate)
            if not tp and surface in init_at: 
                tp = init_at[surface]
            if tp:
                self.goal_info.append((crate, surface, tp))

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        depot_gemini3flash_heuristic_8_val = self._compute(node.state, node.task_network)
        super().update_info(depot_gemini3flash_heuristic_8_val)
        return depot_gemini3flash_heuristic_8_val

    def _compute(self, state: int, task_network: list) -> int:
        # Start with task network length as a baseline estimate
        h = len(task_network)

        # Check each goal crate against its target surface and target place
        for crate, surface, t_place in self.goal_info:
            # Goal: (on crate surface)
            on_id = self.on_map.get((crate, surface))
            if on_id is not None:
                # If not currently on goal surface
                if not ((state >> on_id) & 1):
                    h += 1  # Unavoidable Drop action required
                    
                    # Check if crate is at the correct place
                    at_id = self.at_map.get((crate, t_place))
                    if at_id is not None:
                        if not ((state >> at_id) & 1):
                            # Needs Lift + Load + Unload (Drive and Drop are handled/implied)
                            h += 2
                        else:
                            # At correct place, just needs Lift (Drop already counted)
                            h += 1
        return h
