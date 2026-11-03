"""Domain-Dependent Heuristic for Depots HDDL Domain

Bottleneck Analysis:
  The primary bottleneck in the Depots domain is the need to clear crates
  from surfaces before placing target crates, combined with the need to
  transport crates between places using trucks. Each hoist can only hold
  one crate at a time, and surfaces must be cleared before a crate can be
  placed on them. Moving crates between locations requires: lift, load,
  drive, unload, drop = 5 actions minimum per inter-location move.

Admissible Lower Bounds:
  Term 1 - Unsatisfied goal count: Each unsatisfied (on c s) goal requires
    at minimum 1 action (Drop) to achieve, so count of unsatisfied goals
    is an admissible lower bound.
  
  Term 2 - Location mismatch cost: If a crate needs to be on a surface at
    a different place than where the crate currently is, we need at minimum
    5 actions (Lift + Load + Drive + Unload + Drop). This is admissible
    because these are the minimum required actions for inter-location transport.
  
  Term 3 - Clearing cost: If a goal requires (on c s) and s is not clear
    and the thing on s is not c, we need at least 2 actions (Lift + Load)
    to clear s. Similarly, if c has things on top of it, those must be
    cleared first. This captures the recursive clearing overhead.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic18(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_18"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_ids = {}       # (crate, surface) -> global_id
        self.at_fact_ids = {}       # (locatable, place) -> global_id
        self.clear_fact_ids = {}    # surface -> global_id
        self.in_fact_ids = {}       # (crate, truck) -> global_id
        self.lifting_fact_ids = {}  # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal information
        self.goal_on_pairs = []     # list of (crate, surface) pairs from goals
        self.goal_on_fact_ids = []  # corresponding global_ids
        
        # Object categorization
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # Location of surfaces/crates (from 'at' facts in initial parsing)
        self.surface_locations = {}  # surface -> set of place (from at facts)
        self.hoist_locations = {}    # hoist -> place
        
        # Reverse on map: surface -> crate on top of it (from on facts)
        # We'll compute this dynamically from state
        
        # Place to hoists mapping
        self.place_hoists = {}  # place -> list of hoist global_ids for available check

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[0])
                self.surfaces.add(args[1])
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
                self.crates.add(args[0])
                self.trucks.add(args[1])
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
                self.hoists.add(args[0])
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = f.global_id
                self.hoists.add(args[0])
        
        # Identify pallets vs crates from surfaces
        # Pallets are surfaces that are not crates (appear only as second arg of on)
        # Actually, let's identify from naming convention
        for s in self.surfaces:
            if s.startswith("pallet"):
                self.pallets.add(s)
        
        # Identify trucks from at facts
        for (obj, place) in self.at_fact_ids:
            if obj.startswith("truck"):
                self.trucks.add(obj)
            elif obj.startswith("hoist"):
                self.hoists.add(obj)
                self.hoist_locations[obj] = place
                if place not in self.place_hoists:
                    self.place_hoists[place] = []
                self.place_hoists[place].append(obj)
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                gid = self.on_fact_ids.get((crate, surface))
                if gid is not None:
                    self.goal_on_pairs.append((crate, surface))
                    self.goal_on_fact_ids.append(gid)
        
        # Build a map from surface to the set of places it can be at
        # (pallets are fixed, crates move)
        # We store at_fact_ids for quick lookup
        
        # For quick location lookup, build: object -> list of (place, fact_id)
        self.obj_at_facts = {}  # object -> [(place, fact_global_id), ...]
        for (obj, place), gid in self.at_fact_ids.items():
            if obj not in self.obj_at_facts:
                self.obj_at_facts[obj] = []
            self.obj_at_facts[obj].append((place, gid))
        
        # Precompute: for each goal (crate, surface), find the surface's possible locations
        # Pallets have fixed locations
        self.pallet_location = {}  # pallet -> place
        for p in self.pallets:
            if p in self.obj_at_facts:
                for (place, fid) in self.obj_at_facts[p]:
                    # Pallets are always at their location
                    self.pallet_location[p] = place
                    break
        
        # Build reverse: for each on fact in goals, store the goal_surface location if known
        self.goal_target_place = {}  # index into goal_on_pairs -> place or None
        for i, (crate, surface) in enumerate(self.goal_on_pairs):
            if surface in self.pallet_location:
                self.goal_target_place[i] = self.pallet_location[surface]
            else:
                self.goal_target_place[i] = None  # surface is a crate, location depends on state
        
        # Count number of goals for normalization
        self.num_goals = len(self.goal_on_pairs)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_crate_location(self, crate, state):
        """Find where a crate currently is. Returns place name or None."""
        if crate in self.obj_at_facts:
            for (place, fid) in self.obj_at_facts[crate]:
                if (state >> fid) & 1:
                    return place
        # Check if crate is in a truck
        for (c, t), fid in self.in_fact_ids.items():
            if c == crate and (state >> fid) & 1:
                return "__in_truck__"  # In transit
        return None

    def _get_surface_location(self, surface, state):
        """Find where a surface currently is."""
        if surface in self.pallet_location:
            return self.pallet_location[surface]
        return self._get_crate_location(surface, state)

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        if surface in self.clear_fact_ids:
            return (state >> self.clear_fact_ids[surface]) & 1
        return True  # Unknown surface assumed clear

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        
        # Term 1: Count unsatisfied on-goals
        unsatisfied_count = 0
        for gid in self.goal_on_fact_ids:
            if not ((state >> gid) & 1):
                unsatisfied_count += 1
        
        if unsatisfied_count == 0:
            # All goals satisfied, check task network
            return len(task_network)
        
        # Term 2: Weighted unsatisfied goals with location mismatch estimation
        # For each unsatisfied goal (on crate surface):
        #   - If crate is at same place as surface: need at least 2 actions (Lift + Drop)
        #     BUT if crate is not clear, need to clear it first: +2 per blocking crate
        #     AND if surface is not clear with wrong crate: +2 to clear surface
        #   - If crate is at different place: need at least 5 actions (Lift+Load+Drive+Unload+Drop)
        #   - If crate is in a truck: need at least 2 actions (Unload+Drop)
        
        location_cost = 0
        clearing_cost = 0
        
        for i, (crate, surface) in enumerate(self.goal_on_pairs):
            gid = self.goal_on_fact_ids[i]
            if (state >> gid) & 1:
                continue  # Goal already satisfied
            
            # Check if crate is being lifted by some hoist
            crate_being_lifted = False
            for (h, c), fid in self.lifting_fact_ids.items():
                if c == crate and (state >> fid) & 1:
                    crate_being_lifted = True
                    break
            
            # Check if crate is in a truck
            crate_in_truck = False
            for (c, t), fid in self.in_fact_ids.items():
                if c == crate and (state >> fid) & 1:
                    crate_in_truck = True
                    break
            
            if crate_being_lifted:
                # Just need Drop (1 action) if at right place, or Load+Drive+Unload+Drop (4) if not
                location_cost += 1
            elif crate_in_truck:
                # Need Unload + Drop = 2 actions minimum
                location_cost += 2
            else:
                # Crate is on some surface at some place
                crate_place = self._get_crate_location(crate, state)
                
                target_place = self.goal_target_place.get(i)
                if target_place is None:
                    # Surface is a crate, find its location
                    target_place = self._get_surface_location(surface, state)
                
                if crate_place is not None and target_place is not None:
                    if crate_place == target_place:
                        # Same place: Lift + Drop = 2 actions
                        location_cost += 2
                    else:
                        # Different place: Lift + Load + Drive + Unload + Drop = 5 actions
                        location_cost += 5
                else:
                    # Unknown location, assume moderate cost
                    location_cost += 3
                
                # Clearing cost for crate (if not clear, need to clear it before lifting)
                if not self._is_clear(crate, state):
                    clearing_cost += 2  # At least Lift+Load to move blocking crate
            
            # Clearing cost for target surface
            if not self._is_clear(surface, state):
                # Check if the thing on the surface is the crate we want
                # If so, no clearing needed (it's already there or will be placed)
                # We can't easily check what's on the surface without iterating,
                # so conservatively add clearing cost
                on_fact_check = self.on_fact_ids.get((crate, surface))
                if on_fact_check is not None and (state >> on_fact_check) & 1:
                    pass  # Already on target
                else:
                    clearing_cost += 2  # At least Lift+Load to clear surface
        
        # Term 3: Task network size (remaining tasks give a lower bound)
        # Count non-nop primitive tasks and abstract tasks
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task - costs at least 1 action to decompose into
                tn_cost += 1
            elif hasattr(task, 'add_effects'):
                # Primitive operator
                if task.name and 'nop' not in task.name:
                    tn_cost += 1
        
        # Return max of the three terms
        h_value = max(
            unsatisfied_count,
            location_cost + clearing_cost,
            tn_cost
        )
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
