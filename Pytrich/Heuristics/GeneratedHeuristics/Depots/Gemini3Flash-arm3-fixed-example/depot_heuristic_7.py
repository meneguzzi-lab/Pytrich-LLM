"""
Domain-dependent heuristic for the Depots HDDL domain.

Bottleneck Identification:
The primary bottleneck in the Depots domain is the serial nature of hoist and surface availability. 
A hoist can only lift one crate at a time, and a surface (pallet or crate) must be clear to receive another crate.
Moving crates between locations also requires synchronized truck usage.

Admissible Lower Bounds:
1. h_displacement: Each crate not at its goal surface requires a minimum number of actions.
   - Same location: Lift + Drop = 2 actions.
   - Different location: Lift + Load + Unload + Drop = 4 actions.
   - Being lifted or in truck: 1 or 2 actions remaining.
   This is admissible because these actions are unique to each crate's movement.

2. h_travel: If one or more crates must move from location A to B, at least one 'Drive' action is required.
   Admissible lower bound: Number of unique source locations that contain at least one crate destined for a different location.

3. h_blocking: If a crate is currently at its goal position but sits beneath a crate that needs to move,
   it must eventually be clear. However, more importantly, if a goal surface (like a pallet) is covered 
   by a crate that doesn't belong there, that crate must be lifted (at least 2 actions).

Guidance:
Tie-breaking is added using the task network length to prefer states closer to task decomposition completion.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic7(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_7=None, name="DepotsGemini3FlashHeuristic7"):
        super().__init__(name=name)
        self.goal_crate_to_surface = {}
        self.goal_surface_to_place = {}
        self.pallet_to_place = {}
        
        self.on_facts = []      # List of (fact_id, crate, surface)
        self.at_facts = []      # List of (fact_id, locatable, place)
        self.in_facts = []      # List of (fact_id, crate, truck)
        self.lifting_facts = [] # List of (fact_id, hoist, crate)
        self.clear_facts = []   # List of (fact_id, surface)
        
        self.all_crates = set()
        self.all_surfaces = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse all facts into categorized lists for fast bitwise checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                self.on_facts.append((fact.global_id, args[0], args[1]))
                self.all_crates.add(args[0])
                self.all_surfaces.add(args[0])
                self.all_surfaces.add(args[1])
            elif pred == "at":
                self.at_facts.append((fact.global_id, args[0], args[1]))
                if "pallet" in args[0]:
                    self.pallet_to_place[args[0]] = args[1]
                    self.all_surfaces.add(args[0])
            elif pred == "in":
                self.in_facts.append((fact.global_id, args[0], args[1]))
            elif pred == "lifting":
                self.lifting_facts.append((fact.global_id, args[0], args[1]))
            elif pred == "clear":
                self.clear_facts.append((fact.global_id, args[0]))

        # 2. Extract Goals
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_crate_to_surface[args[0]] = args[1]

        # 3. Precompute goal locations for all crates
        # A crate's goal place is the place of its goal pallet (base of the stack)
        self.goal_surface_to_place = self.pallet_to_place.copy()
        changed = True
        while changed:
            changed = False
            for crate, g_surf in self.goal_crate_to_surface.items():
                if crate not in self.goal_surface_to_place:
                    if g_surf in self.goal_surface_to_place:
                        self.goal_surface_to_place[crate] = self.goal_surface_to_place[g_surf]
                        changed = True

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def _compute(self, state, task_network):
        # 1. Build current state maps
        curr_on = {}     # crate -> surface
        curr_at = {}     # locatable -> place
        curr_in = {}     # crate -> truck
        curr_lift = {}   # crate -> hoist
        
        for fid, c, s in self.on_facts:
            if (state >> fid) & 1: curr_on[c] = s
        for fid, l, p in self.at_facts:
            if (state >> fid) & 1: curr_at[l] = p
        for fid, c, t in self.in_facts:
            if (state >> fid) & 1: curr_in[c] = t
        for fid, h, c in self.lifting_facts:
            if (state >> fid) & 1: curr_lift[c] = h

        h = 0
        source_places = set()

        # 2. Crate Displacement Cost (Admissible)
        for crate, g_surf in self.goal_crate_to_surface.items():
            c_on = curr_on.get(crate)
            if c_on == g_surf: 
                continue

            g_place = self.goal_surface_to_place.get(g_surf)
            
            if c_on: # Crate is on some surface
                c_place = curr_at.get(crate)
                if c_place == g_place:
                    h += 2 # Lift + Drop
                else:
                    h += 4 # Lift + Load + Unload + Drop
                    if c_place: source_places.add(c_place)
            elif crate in curr_in:
                # In truck - needs unload and drop
                h += 2 # Unload + Drop
                # Travel for the truck is handled by shared travel costs
            elif crate in curr_lift:
                # Being lifted
                h += 1 # Just Drop

        # 3. Truck Travel Cost (Admissible)
        h += len(source_places)

        # 4. Blocking Cost (Admissible)
        # If a crate is NOT in its goal position, it must be lifted (already counted).
        # If a crate is IN its goal position but something is on top of it that doesn't belong,
        # the thing on top must be lifted.
        goal_surfaces = set(self.goal_crate_to_surface.values())
        top_of = {s: c for c, s in curr_on.items()}
        
        for surf in goal_surfaces:
            if surf in top_of:
                c_above = top_of[surf]
                # If the crate on top is NOT the one that's supposed to be there
                if self.goal_crate_to_surface.get(c_above) != surf:
                    # This c_above must be moved. If it was already misplaced,
                    # its Lift action was counted in displacement. 
                    # If it WAS correctly placed but is blocking another goal, 
                    # we add 2 for the extra Lift/Drop required.
                    if c_above in self.goal_crate_to_surface and curr_on.get(c_above) == self.goal_crate_to_surface[c_above]:
                        h += 2

        # 5. HTN Guidance
        h += 0.001 * len(task_network)

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
