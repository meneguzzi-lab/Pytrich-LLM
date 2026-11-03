"""
Domain-dependent heuristic for the Depots domain.

Bottleneck Analysis:
The primary bottleneck in Depots is the serial interaction between the Blocksworld-style stacking constraints 
(on, clear) and the Logistics-style transportation constraints (in, at, truck-travel). 
A crate can only be moved if it is clear, and its final placement is only 'settled' if it is on the correct 
surface and that surface itself is already settled.

Admissibility Sketches:
1. Well-Placed Cost: A crate C is 'well-placed' if it's on its goal surface and that surface is either a pallet 
   or a settled crate. Any unsettled crate requires at least a Lift and a Drop (2 actions). 
   If the crate is at the wrong location, it also requires at least a Load and an Unload (2 additional actions).
2. Chain Dependency: To settle a stack of N crates, they must be processed from bottom to top. 
   This heuristic calculates the 'well-placed' status by tracing the goal-chains starting from pallets.
3. Task Network Guidance: Each non-trivial task in the network represents a lower bound on work 
   remaining to reach the goal state from the current decomposition level.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic19(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_19=None, name="DepotsGemini3FlashHeuristic19"):
        super().__init__(name=name)
        self.goal_on = {}  # crate -> target surface
        self.home_place = {}  # surface -> place where it belongs/is fixed
        self.pallets = set()
        self.crates = set()
        self.trucks = set()
        self.hoists = set()
        
        # Fact Map: fid -> metadata
        self.on_fact_map = {} # fid -> (crate, surface)
        self.at_fact_map = {} # fid -> (locatable, place)
        self.in_fact_map = {} # fid -> (crate, truck)
        self.lifting_fact_map = {} # fid -> (hoist, crate)
        
        # Static object locations
        self.pallet_at = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse goals
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on':
                self.goal_on[args[0]] = args[1]

        # 2. Categorize objects and map facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                self.on_fact_map[f.global_id] = (args[0], args[1])
                self.crates.add(args[0])
            elif pred == 'at':
                self.at_fact_map[f.global_id] = (args[0], args[1])
                # Distinguish locatables
                if 'pallet' in args[0]: 
                    self.pallets.add(args[0])
                    self.pallet_at[args[0]] = args[1]
                elif 'truck' in args[0]: self.trucks.add(args[0])
                elif 'hoist' in args[0]: self.hoists.add(args[0])
                elif 'crate' in args[0]: self.crates.add(args[0])
            elif pred == 'in':
                self.in_fact_map[f.global_id] = (args[0], args[1])
                self.crates.add(args[0])
                self.trucks.add(args[1])
            elif pred == 'lifting':
                self.lifting_fact_map[f.global_id] = (args[0], args[1])
                self.hoists.add(args[0])
                self.crates.add(args[1])

        # 3. Precompute home places (where crates must end up)
        for p, place in self.pallet_at.items():
            self.home_place[p] = place
        
        changed = True
        while changed:
            changed = False
            for c in self.crates:
                if c not in self.home_place:
                    gs = self.goal_on.get(c)
                    if gs in self.home_place:
                        self.home_place[c] = self.home_place[gs]
                        changed = True

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return None, []

    def _compute(self, state, task_network):
        curr_on = {} # surface -> crate
        curr_at = {} # locatable -> place
        curr_in = {} # crate -> truck
        curr_lifting = {} # crate -> hoist
        
        # Bitwise iteration over set facts is much faster for large states
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            fid = lsb.bit_length() - 1
            temp_state ^= lsb
            
            if fid in self.on_fact_map:
                c, s = self.on_fact_map[fid]
                curr_on[s] = c
            elif fid in self.at_fact_map:
                l, p = self.at_fact_map[fid]
                curr_at[l] = p
            elif fid in self.in_fact_map:
                c, t = self.in_fact_map[fid]
                curr_in[c] = t
            elif fid in self.lifting_fact_map:
                h, c = self.lifting_fact_map[fid]
                curr_lifting[c] = h

        # Determine which crates are well-placed
        well_placed = set(self.pallets)
        for p in self.pallets:
            curr = p
            while curr in curr_on:
                c = curr_on[curr]
                if self.goal_on.get(c) == curr:
                    well_placed.add(c)
                    curr = c
                else:
                    break

        h = 0
        for c in self.crates:
            if c not in well_placed:
                # Base move cost
                cost = 2
                
                # Find current place
                cp = None
                if c in curr_in:
                    cp = curr_at.get(curr_in[c])
                    cost -= 1 # Partially processed (loaded)
                elif c in curr_lifting:
                    cp = curr_at.get(curr_lifting[c])
                    cost -= 1 # Partially processed (lifting)
                else:
                    # Find location of surface it is on
                    # Need to trace down to a pallet/locatable with an 'at' fact
                    scan = c
                    visited = {scan}
                    while scan in curr_on:
                        scan = curr_on[scan]
                        if scan in visited: break
                        visited.add(scan)
                    cp = curr_at.get(scan)
                
                # Transport penalty
                gp = self.home_place.get(c)
                if gp and cp and cp != gp:
                    cost += 2
                
                h += cost
        
        # HTN guidance: add task network length
        h += len(task_network)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
