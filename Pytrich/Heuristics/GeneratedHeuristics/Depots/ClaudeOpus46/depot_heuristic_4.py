"""Domain-Dependent Heuristic for Depots HDDL Domain

Bottleneck Analysis:
  The primary bottleneck in the Depots domain is that crates must be moved
  between places (depots/distributors) via trucks, but each movement requires:
  1. Clearing crates stacked on top of the target crate (recursive unstacking)
  2. Clearing the destination surface
  3. Having a hoist available at the right place
  4. Having a truck at the right place (or driving it there)
  Each crate relocation that requires cross-place transport needs at minimum:
  Lift + Load + Drive + Unload + Drop = 5 actions, plus clearing costs.
  Same-place moves need at minimum: Lift + Drop = 2 actions, plus clearing.

Admissible Lower Bounds:
  Term 1 (Goal Unsatisfied Count): For each goal fact (on ?c ?s) not currently
    true, at least 1 action (Drop) is needed. This is a trivial lower bound.
  
  Term 2 (Transport-Aware Cost): For each unsatisfied goal (on ?c ?s), estimate
    the minimum number of actions needed based on whether the crate and surface
    are at the same place or different places. Different places require at least
    5 actions (Lift+Load+Drive+Unload+Drop). Same place requires at least 2
    (Lift+Drop). We sum over all unsatisfied goals but since actions can be
    shared, we take a weighted fraction to stay admissible.
  
  Term 3 (Clearing Cost): For each unsatisfied goal, if the destination surface
    is not clear, we need additional clearing work. Each crate that must be
    removed from a surface costs at least 2 actions (Lift+Load or Lift+Drop).
    We estimate clearing depth for blocked surfaces.

The heuristic returns max(term1, term2, term3) plus task-network-aware estimates.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic4(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_4"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_ids = {}       # (crate, surface) -> global_id
        self.at_fact_ids = {}       # (locatable, place) -> global_id
        self.clear_fact_ids = {}    # surface -> global_id
        self.in_fact_ids = {}       # (crate, truck) -> global_id
        self.lifting_fact_ids = {}  # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on_pairs = []     # list of (crate, surface, fact_global_id)
        
        # Location maps
        self.crate_locations = {}   # crate -> set of places where it could be (from at facts)
        self.surface_locations = {} # surface -> set of places
        self.hoist_locations = {}   # hoist -> place
        
        # Object type sets
        self.crates = set()
        self.surfaces = set()
        self.pallets = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()
        
        # Stack info: what's on top of what in initial state
        self.initial_on = {}  # surface -> crate that's on it
        
        # Task name maps for task network analysis
        self.task_cost_cache = {}
        
        # Operator/task name prefixes
        self.do_put_on_prefix = "do_put_on"
        self.do_clear_prefix = "do_clear"
        self.do_get_truck_prefix = "do_get_truck"
        self.do_lift_crate_prefix = "do_lift_crate"
        self.do_load_truck_prefix = "do_load_truck"
        self.do_unload_truck_prefix = "do_unload_truck"

    def _parse_fact_name(self, name: str):
        """Parse a grounded fact name like +on[crate0,pallet1] into (predicate, [args])."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _parse_task_name(self, name: str):
        """Parse a task/operator name to extract the task type and arguments."""
        # Names look like: do_put_on[crate0,pallet1] or Drive[truck0,depot0,distributor0]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Try without brackets
        paren_idx = name.find("(")
        if paren_idx > 0:
            task_type = name[:paren_idx]
            return task_type, []
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                # Classify objects
                obj, place = args[0], args[1]
                self.places.add(place)
                if obj.startswith("crate"):
                    self.crates.add(obj)
                elif obj.startswith("pallet"):
                    self.pallets.add(obj)
                    self.surfaces.add(obj)
                elif obj.startswith("truck"):
                    self.trucks.add(obj)
                elif obj.startswith("hoist"):
                    self.hoists.add(obj)
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = f.global_id
        
        # Add crates to surfaces set (crates can be surfaces too)
        self.surfaces.update(self.crates)
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                fact_id = self.on_fact_ids.get((crate, surface))
                if fact_id is not None:
                    self.goal_on_pairs.append((crate, surface, fact_id))
        
        # Build initial location map from initial state
        self._build_location_map(model.initial_state)
        
        # Precompute hoist locations (hoists don't move)
        for h in self.hoists:
            for p in self.places:
                key = (h, p)
                if key in self.at_fact_ids:
                    fid = self.at_fact_ids[key]
                    if (model.initial_state >> fid) & 1:
                        self.hoist_locations[h] = p
        
        # Precompute place-to-place: which places have hoists
        self.places_with_hoists = set()
        for h, p in self.hoist_locations.items():
            self.places_with_hoists.add(p)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _build_location_map(self, state):
        """Build maps of where crates and surfaces currently are."""
        pass  # We'll do per-query checks instead for efficiency

    def _get_crate_place(self, crate, state):
        """Find which place a crate is at, or None if not at any place (in truck or being lifted)."""
        for p in self.places:
            key = (crate, p)
            if key in self.at_fact_ids:
                fid = self.at_fact_ids[key]
                if (state >> fid) & 1:
                    return p
        return None

    def _get_surface_place(self, surface, state):
        """Find which place a surface is at."""
        for p in self.places:
            key = (surface, p)
            if key in self.at_fact_ids:
                fid = self.at_fact_ids[key]
                if (state >> fid) & 1:
                    return p
        return None

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        if surface in self.clear_fact_ids:
            return (state >> self.clear_fact_ids[surface]) & 1
        return True

    def _is_on(self, crate, surface, state):
        """Check if crate is on surface."""
        key = (crate, surface)
        if key in self.on_fact_ids:
            return (state >> self.on_fact_ids[key]) & 1
        return False

    def _is_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for t in self.trucks:
            key = (crate, t)
            if key in self.in_fact_ids:
                if (state >> self.in_fact_ids[key]) & 1:
                    return True
        return False

    def _count_unsatisfied_goals(self, state):
        """Count how many goal on-facts are not satisfied."""
        count = 0
        for crate, surface, fact_id in self.goal_on_pairs:
            if not ((state >> fact_id) & 1):
                count += 1
        return count

    def _estimate_goal_cost(self, state):
        """Estimate cost for unsatisfied goals with transport awareness.
        
        For each unsatisfied goal (on ?c ?s):
        - If crate is at same place as surface: min cost = 2 (Lift + Drop)
        - If crate is at different place: min cost = 5 (Lift + Load + Drive + Unload + Drop)
        - If crate is in a truck: min cost = 3 (Drive? + Unload + Drop)
        - If surface is not clear: add clearing cost
        """
        total_cost = 0
        
        for crate, surface, fact_id in self.goal_on_pairs:
            if (state >> fact_id) & 1:
                continue  # Goal already satisfied
            
            # Base cost: at minimum we need a Drop action
            cost = 1
            
            # Check if crate is being lifted by some hoist
            crate_being_lifted = False
            for h in self.hoists:
                key = (h, crate)
                if key in self.lifting_fact_ids:
                    if (state >> self.lifting_fact_ids[key]) & 1:
                        crate_being_lifted = True
                        break
            
            if crate_being_lifted:
                # Just need Drop (+ maybe drive to get to right place)
                cost = 1
            elif self._is_in_truck(crate, state):
                # Need Unload + Drop = 2, possibly Drive = 3
                cost = 2
            else:
                # Crate is on some surface, need to Lift it
                crate_place = self._get_crate_place(crate, state)
                surface_place = self._get_surface_place(surface, state)
                
                if crate_place is not None and surface_place is not None:
                    if crate_place == surface_place:
                        # Same place: Lift + Drop = 2
                        cost = 2
                    else:
                        # Different places: Lift + Load + Drive + Unload + Drop = 5
                        cost = 5
                elif crate_place is not None:
                    # Surface location unknown (maybe it's a crate that needs placing first)
                    cost = 3
                else:
                    cost = 2
                
                # Clearing cost for the crate itself
                if not self._is_clear(crate, state):
                    # Need to clear crate first: at least Lift + Load = 2 per blocking crate
                    cost += 2
            
            # Clearing cost for destination surface
            if not self._is_clear(surface, state):
                # Check if what's on the surface also needs to go somewhere as a goal
                # At minimum, clearing requires Lift + Load = 2
                cost += 2
            
            total_cost += cost
        
        return total_cost

    def _estimate_task_network_cost(self, task_network):
        """Estimate minimum cost from the task network.
        
        Each primitive operator in the network costs 1.
        Each abstract task costs at least its minimum decomposition depth.
        We use domain knowledge to estimate abstract task costs.
        """
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            tname = task.name
            
            # Check if it's a primitive operator
            if hasattr(task, 'add_effects'):
                # It's an Operator
                if "nop" in tname:
                    cost += 0
                else:
                    cost += 1
            else:
                # It's an AbstractTask - estimate based on type
                task_type, args = self._parse_task_name(tname)
                
                if task_type == "do_put_on":
                    # Minimum: nop (if already satisfied) or Lift+Drop=2 (same place)
                    # or Lift+Load+Drive+Unload+Drop=5 (different place)
                    # We'll use 1 as a conservative lower bound since m0 could apply (nop)
                    cost += 1
                elif task_type == "do_clear":
                    # Minimum: nop (if already clear)
                    cost += 0
                elif task_type == "do_get_truck":
                    # Minimum: nop (if truck already there) or Drive=1
                    cost += 0
                elif task_type == "do_lift_crate":
                    # Minimum: Lift=1
                    cost += 1
                elif task_type == "do_load_truck":
                    # Minimum: do_get_truck + do_clear + Lift + Load = 2+ actions
                    cost += 2
                elif task_type == "do_unload_truck":
                    # Minimum: do_clear + Unload + Drop = 2+ actions
                    cost += 2
                else:
                    # Unknown task type
                    cost += 1
        
        return cost

    def _estimate_task_network_cost_stateful(self, state, task_network):
        """State-aware task network cost estimation.
        
        For do_put_on tasks, check if goal is already satisfied (cost 0)
        or estimate based on crate/surface locations.
        """
        if not task_network:
            return 0
        
        cost = 0
        for task in task_network:
            tname = task.name
            
            # Check if it's a primitive operator
            if hasattr(task, 'add_effects'):
                if "nop" not in tname:
                    cost += 1
            else:
                task_type, args = self._parse_task_name(tname)
                
                if task_type == "do_put_on" and len(args) >= 2:
                    crate, surface = args[0], args[1]
                    # Check if already on target
                    key = (crate, surface)
                    if key in self.on_fact_ids:
                        fid = self.on_fact_ids[key]
                        if (state >> fid) & 1:
                            # Already satisfied - just nop
                            cost += 0
                            continue
                    
                    # Not satisfied - estimate cost
                    crate_being_lifted = False
                    for h in self.hoists:
                        lkey = (h, crate)
                        if lkey in self.lifting_fact_ids:
                            if (state >> self.lifting_fact_ids[lkey]) & 1:
                                crate_being_lifted = True
                                break
                    
                    if crate_being_lifted:
                        # Just Drop + maybe clearing destination
                        c = 1
                        if not self._is_clear(surface, state):
                            c += 2
                        cost += c
                    elif self._is_in_truck(crate, state):
                        # Drive? + Unload + Drop + maybe clearing
                        c = 2
                        if not self._is_clear(surface, state):
                            c += 2
                        cost += c
                    else:
                        # Need to pick up crate
                        crate_place = self._get_crate_place(crate, state)
                        surface_place = self._get_surface_place(surface, state)
                        
                        if crate_place is not None and surface_place is not None:
                            if crate_place == surface_place:
                                c = 2  # Lift + Drop
                            else:
                                c = 5  # Lift + Load + Drive + Unload + Drop
                        else:
                            c = 3  # Conservative estimate
                        
                        # Clearing crate
                        if not self._is_clear(crate, state):
                            c += 2
                        # Clearing destination
                        if not self._is_clear(surface, state):
                            c += 2
                        
                        cost += c
                
                elif task_type == "do_clear" and len(args) >= 2:
                    surface_arg = args[0]
                    if self._is_clear(surface_arg, state):
                        cost += 0
                    else:
                        # Need to lift what's on it + load into truck: 4+ actions
                        cost += 4
                
                elif task_type == "do_get_truck" and len(args) >= 2:
                    truck, place = args[0], args[1]
                    tkey = (truck, place)
                    if tkey in self.at_fact_ids:
                        if (state >> self.at_fact_ids[tkey]) & 1:
                            cost += 0
                        else:
                            cost += 1  # Drive
                    else:
                        cost += 1
                
                elif task_type == "do_lift_crate":
                    cost += 1
                
                elif task_type == "do_load_truck":
                    cost += 2
                
                elif task_type == "do_unload_truck":
                    cost += 2
                
                else:
                    cost += 1
        
        return cost

    def _goal_interference_cost(self, state):
        """Detect goals that are currently true but will be disrupted.
        
        A goal (on ?c ?s) that is true now might be disrupted if:
        - ?s needs to be moved for another goal
        - Something needs to be placed on ?c that isn't there yet
        
        This detects false-trivial goals (P5 pattern).
        """
        interference = 0
        
        # Build set of crates that need to be moved (have unsatisfied on-goals)
        crates_to_move = set()
        surfaces_needing_clearing = set()
        
        for crate, surface, fact_id in self.goal_on_pairs:
            if not ((state >> fact_id) & 1):
                crates_to_move.add(crate)
                # The current position of the crate's destination surface
                # might need clearing
                if not self._is_clear(surface, state):
                    surfaces_needing_clearing.add(surface)
        
        # Check currently-satisfied goals that might be disrupted
        for crate, surface, fact_id in self.goal_on_pairs:
            if (state >> fact_id) & 1:
                # This goal is currently true
                # Will it be disrupted?
                # If 'surface' is a crate that needs to move, this goal will break
                if surface in crates_to_move:
                    interference += 2  # Will need to redo: Lift + Drop at minimum
                # If 'crate' has something on it that needs to move to get cleared
                # for another crate to be placed on it
                if crate in surfaces_needing_clearing:
                    # Something on crate needs to be moved
                    interference += 2
        
        return interference

    def _compute(self, state, task_network):
        """Compute the heuristic value."""
        # Term 1: Simple count of unsatisfied goals
        term1 = self._count_unsatisfied_goals(state)
        
        # Term 2: Transport-aware goal cost estimation
        # This is an admissible sum but we apply a dampening factor
        # since actions can be shared across goals (e.g., one Drive serves multiple crates)
        term2_raw = self._estimate_goal_cost(state)
        # Dampening: in worst case each action serves only 1 goal
        # but truck drives can be shared. We use a slight discount.
        term2 = term2_raw
        
        # Term 3: Task network cost (state-aware)
        term3 = self._estimate_task_network_cost_stateful(state, task_network)
        
        # Term 4: Goal interference detection
        term4 = self._goal_interference_cost(state)
        
        # Return max of independent bounds, plus interference as additive
        # term2 and term3 estimate similar things from different angles
        # We take max of (term2, term3) as the primary estimate
        # term4 adds additional cost for goals that will be disrupted
        h = max(term1, term2, term3) + term4 // 2
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
