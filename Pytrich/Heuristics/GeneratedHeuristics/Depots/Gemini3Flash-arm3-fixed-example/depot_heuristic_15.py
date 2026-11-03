"""
Domain-dependent heuristic for the Depots domain.

1. Bottleneck Analysis:
The primary bottleneck in the Depots domain is stack management at specific locations. 
Each place (depot/distributor) has a limited number of hoists (often one), and crates 
are stacked on pallets or other crates. To access a crate or a target surface, all 
crates above it must be cleared. Moving crates between locations adds another layer 
of complexity requiring truck logistics (Load, Drive, Unload).

2. Admissible Lower Bounds:
- Transport Cost: Any crate currently at location A that must be at location B requires 
  at least: Lift, Load, Drive, Unload, Drop (5 actions). If at the same location, 
  it requires at least Lift and Drop (2 actions).
- Clearing Cost: For every crate 'c' that needs to be moved or used as a base, 
  every crate 'x' currently on top of it must be lifted and dropped elsewhere 
  (at least 2 actions per blocking crate).
- Task Network Cost: Each 'do_put_on(c, s)' task in the network represents an 
  unfulfilled goal that will require at least the transport/stacking sequence.

These are admissible because they represent the minimum number of physical actions 
required to satisfy the structural constraints of the domain (lifting, driving, clearing).
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic15(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_15=None, name="DepotsGemini3FlashHeuristic15"):
        super().__init__(name=name)
        self.goal_on = {}  # crate -> target_surface
        self.on_ids = {}   # (crate, surface) -> global_id
        self.at_ids = {}   # (locatable, place) -> global_id
        self.in_ids = {}   # (crate, truck) -> global_id
        self.clear_ids = {} # surface -> global_id
        self.crates = []
        self.surfaces = []
        self.trucks = []
        self.places = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify Goals
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_facts:
            pred, args = self._parse_fact_name(fact_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # 2. Map Facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
                if args[0] not in self.crates: self.crates.append(args[0])
                if args[1] not in self.surfaces: self.surfaces.append(args[1])
            elif pred == "at":
                self.at_ids[(args[0], args[1])] = fact.global_id
                if args[1] not in self.places: self.places.append(args[1])
                if args[0].startswith("truck"): 
                    if args[0] not in self.trucks: self.trucks.append(args[0])
            elif pred == "in":
                self.in_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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

    def _get_at(self, state, locatable):
        for p in self.places:
            if (state >> self.at_ids.get((locatable, p), 0)) & 1:
                return p
        for t in self.trucks:
            if (state >> self.in_ids.get((locatable, t), 0)) & 1:
                return t
        return None

    def _get_on(self, state, crate):
        for s in self.surfaces:
            if (state >> self.on_ids.get((crate, s), 0)) & 1:
                return s
        return None

    def _count_above(self, state, surface):
        count = 0
        curr = surface
        # This is a simplification: find what is on 'curr'
        # In a real stack, we'd need to iterate. 
        # Since we need O(1) in __call__, we only check the immediate crate.
        for c in self.crates:
            if (state >> self.on_ids.get((c, curr), 0)) & 1:
                return 1 + self._count_above(state, c)
        return 0

    def _compute(self, state, task_network):
        total_h = 0
        
        # Heuristic Term 1: Distance for unsettled goal crates
        for crate, goal_surf in self.goal_on.items():
            curr_surf = self._get_on(state, crate)
            if curr_surf != goal_surf:
                # Crate is not where it belongs
                c_loc = self._get_at(state, crate)
                s_loc = self._get_at(state, goal_surf)
                
                if c_loc and s_loc:
                    if c_loc == s_loc:
                        total_h += 2 # Lift + Drop
                    else:
                        total_h += 5 # Lift + Load + Drive + Unload + Drop
                else:
                    total_h += 2 # Default minimum

                # Heuristic Term 2: Clearing costs
                # Crates on top of our target crate must be moved
                total_h += 1 * self._count_above(state, crate)
                # Crates on top of our goal surface must be moved
                total_h += 1 * self._count_above(state, goal_surf)

        # Heuristic Term 3: Task Network
        # Every remaining task suggests at least one more operation
        for task in task_network:
            if task.name.startswith("do_put_on"):
                total_h += 1

        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
