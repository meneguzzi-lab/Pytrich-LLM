"""
Domain-dependent heuristic for the Depots domain.

1. Bottleneck Analysis:
The primary bottleneck in the Depots domain is the serial manipulation of crates using a limited number of hoists 
and the multi-stage transport process (Lift-Load-Drive-Unload-Drop). A crate cannot be moved until all crates 
above it are cleared, and it cannot be placed until the target surface is cleared.

2. Admissible Lower Bounds:
- Term 1 (Crate Movement Cost): Each crate not in its goal-stable position requires at least a Lift and a Drop (2 actions).
  If the crate must move between different places, it further requires a Load and an Unload (2 additional actions).
- Term 2 (Clearance Cost): For each crate that needs to be moved or used as a target surface, every crate currently 
  above it must be lifted and placed elsewhere, costing at least 2 actions (Lift and Drop) per obstructing crate.
- Term 3 (Task Cost): Each high-level task in the task network has a minimum number of decomposition steps 
  remaining based on the current state (e.g., a do_put_on task for a crate at a different location).

Admissibility Sketch:
Each of these terms counts unique, unavoidable actions. Since one hoist can only Lift/Drop one crate at a time 
and each crate requires its own set of manipulation actions, summing costs for distinct crates remains admissible.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic17(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_17=None, name="DepotsGemini3FlashHeuristic17"):
        super().__init__(name=name)
        self.on_facts = {}       # crate -> [(fact_id, surface)]
        self.at_facts = {}       # locatable -> [(fact_id, place)]
        self.in_facts = {}       # crate -> [(fact_id, truck)]
        self.clear_facts = {}    # surface -> fact_id
        self.goal_on = {}        # crate -> surface
        self.pallets = set()
        self.crates = set()
        self.trucks = set()
        self.places = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse Goal State
        if hasattr(model, 'goals') and model.goals:
            goal_names = model.state_explicit_repr(model.goals)
            for gname in goal_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) == 2:
                    self.goal_on[args[0]] = args[1]

        # Parse Facts for Mapping
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                crate, surface = args[0], args[1]
                if crate not in self.on_facts: self.on_facts[crate] = []
                self.on_facts[crate].append((fact.global_id, surface))
                self.crates.add(crate)
            elif pred == "at":
                loc, place = args[0], args[1]
                if loc not in self.at_facts: self.at_facts[loc] = []
                self.at_facts[loc].append((fact.global_id, place))
                self.places.add(place)
            elif pred == "in":
                crate, truck = args[0], args[1]
                if crate not in self.in_facts: self.in_facts[crate] = []
                self.in_facts[crate].append((fact.global_id, truck))
                self.trucks.add(truck)
            elif pred == "clear":
                self.clear_facts[args[0]] = fact.global_id
            elif pred == "lifting":
                self.crates.add(args[1])

        # Infer Pallets
        all_surfaces = set(self.clear_facts.keys())
        self.pallets = all_surfaces - self.crates

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

    def _get_crate_info(self, state, crate):
        # Returns (surface, place, in_truck)
        surface, place, truck = None, None, None
        
        # Check if on a surface
        if crate in self.on_facts:
            for fid, s in self.on_facts[crate]:
                if (state >> fid) & 1:
                    surface = s
                    break
        
        # Check if in a truck
        if not surface and crate in self.in_facts:
            for fid, t in self.in_facts[crate]:
                if (state >> fid) & 1:
                    truck = t
                    break
        
        # Determine Place
        if surface:
            for fid, p in self.at_facts.get(surface, []):
                if (state >> fid) & 1:
                    place = p
                    break
            # If surface is a crate, recurse to find place (rare but possible)
            if not place and surface in self.crates:
                _, place, _ = self._get_crate_info(state, surface)
        elif truck:
            for fid, p in self.at_facts.get(truck, []):
                if (state >> fid) & 1:
                    place = p
                    break
        else:
            # Check direct 'at' fact (lifting scenario)
            for fid, p in self.at_facts.get(crate, []):
                if (state >> fid) & 1:
                    place = p
                    break
                    
        return surface, place, truck

    def _get_stack_height(self, state, surface, on_map):
        count = 0
        curr = surface
        while curr in on_map:
            count += 1
            curr = on_map[curr]
        return count

    def _compute(self, state, task_network):
        # 1. Build current state mapping for fast stack calculation
        on_map = {} # surface -> crate
        for crate, facts in self.on_facts.items():
            for fid, surface in facts:
                if (state >> fid) & 1:
                    on_map[surface] = crate
                    break

        # 2. Estimate Goal-Driven Cost
        h_state = 0
        goal_crates = set(self.goal_on.keys())
        
        # A crate is stable if it's on its goal surface and everything below it is stable
        stable_crates = set()
        check_list = list(self.pallets)
        while check_list:
            s = check_list.pop(0)
            c = on_map.get(s)
            if c and c in goal_crates and self.goal_on[c] == s:
                stable_crates.add(c)
                check_list.append(c)

        for crate in goal_crates:
            if crate in stable_crates: continue
            
            # Crate must move
            h_state += 2 # Lift + Drop
            curr_s, curr_p, curr_t = self._get_crate_info(state, crate)
            goal_s = self.goal_on[crate]
            _, goal_p, _ = self._get_crate_info(state, goal_s)
            
            if curr_p != goal_p:
                h_state += 2 # Load + Unload
            
            # Obstructions
            h_state += 2 * self._get_stack_height(state, crate, on_map)
            h_state += 2 * self._get_stack_height(state, goal_s, on_map)

        # 3. Estimate Task-Network Cost
        h_tn = 0
        for task in task_network:
            name = task.name.lower()
            if "do_put_on" in name:
                # Args usually (crate, surface)
                parts = task.name.replace("(", "").replace(")", "").split()
                if len(parts) >= 3:
                    c, s = parts[1], parts[2]
                    if c in stable_crates and self.goal_on.get(c) == s:
                        continue
                    h_tn += 2
                    _, cp, _ = self._get_crate_info(state, c)
                    _, sp, _ = self._get_crate_info(state, s)
                    if cp != sp: h_tn += 2
            elif "do_clear" in name or "do_lift" in name:
                h_tn += 1
            elif "do_load" in name or "do_unload" in name:
                h_tn += 2

        return max(h_state, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
