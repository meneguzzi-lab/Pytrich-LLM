# Depots Domain Heuristic
# Bottleneck: Hoists are serializing resources - one per place, each lift/drop/load/unload
# requires exclusive hoist access. Moving crates between places requires 5+ actions.
# Stack clearing dominates plan length for deeply stacked crates.
#
# Admissible lower bounds:
# T1: Goal displacement - min actions to place each crate on its goal surface
#     (2 if same place, 5 if different place). Admissible: can't do it in fewer.
# T2: Task network depth - each non-nop abstract task needs >= 1 primitive action.
#     Weighted by task type to estimate realistic decomposition cost.
# T3: Unsatisfied goal count as baseline admissible bound.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic1(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_1"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_ids = {}       # (crate, surface) -> global_id
        self.at_fact_ids = {}       # (locatable, place) -> global_id
        self.in_fact_ids = {}       # (crate, truck) -> global_id
        self.clear_fact_ids = {}    # surface -> global_id
        self.lifting_fact_ids = {}  # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on_pairs = []     # list of (crate, surface, fact_global_id)
        
        # Location maps
        self.crate_initial_place = {}  # crate -> place (from initial state)
        self.surface_place = {}        # surface -> place
        self.hoist_place = {}          # hoist -> place
        
        # All objects by type
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # For stack analysis
        self.initial_on = {}  # crate -> surface it's on initially
        self.initial_stack_above = {}  # surface -> list of crates above it (bottom to top)
        
        # Task type info
        self.task_name_prefix_cost = {}
        
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

    def _parse_task_name(self, name: str):
        """Extract the task type prefix from a grounded task/operator name."""
        # Names look like: do_put_on[crate1,pallet2] or Drive[truck0,depot0,depot1]
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            return name[:bracket_idx]
        return name

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = f.global_id
        
        # Identify object types from fact arguments
        for (obj, place) in self.at_fact_ids:
            if obj.startswith("crate"):
                self.crates.add(obj)
                self.places.add(place)
            elif obj.startswith("truck"):
                self.trucks.add(obj)
                self.places.add(place)
            elif obj.startswith("hoist"):
                self.hoists.add(obj)
                self.places.add(place)
                self.hoist_place[obj] = place  # hoists don't move
            elif obj.startswith("pallet"):
                self.pallets.add(obj)
                self.places.add(place)
                self.surface_place[obj] = place  # pallets don't move
        
        self.surfaces = self.crates | self.pallets
        
        # Parse goal state to find goal on-relations
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gf_name in goal_fact_names:
            pred, args = self._parse_fact_name(gf_name)
            if pred == "on" and len(args) == 2:
                fact_id = self.on_fact_ids.get((args[0], args[1]))
                if fact_id is not None:
                    self.goal_on_pairs.append((args[0], args[1], fact_id))
        
        # Build initial state info for location tracking
        initial_state = model.initial_state
        for (obj, place), fid in self.at_fact_ids.items():
            if (initial_state >> fid) & 1:
                if obj.startswith("crate"):
                    self.crate_initial_place[obj] = place
                    self.surface_place[obj] = place  # initial location for crates as surfaces
        
        # Build initial on-relations for stack analysis
        for (crate, surface), fid in self.on_fact_ids.items():
            if (initial_state >> fid) & 1:
                if crate.startswith("crate"):
                    self.initial_on[crate] = surface
        
        # Precompute goal surface locations (where the surface needs to be)
        # For pallets, location is fixed. For crates used as surfaces in goals,
        # we need to figure out where they'll end up - this is complex, so we
        # use initial location as approximation
        self.goal_surface_location = {}
        for (crate, surface, fid) in self.goal_on_pairs:
            if surface in self.surface_place:
                self.goal_surface_location[surface] = self.surface_place[surface]
        
        # Precompute: for each place, which hoists are there
        self.place_hoists = {}
        for h, p in self.hoist_place.items():
            if p not in self.place_hoists:
                self.place_hoists[p] = []
            self.place_hoists[p].append(h)
        
        # Precompute number of goals
        self.num_goals = len(self.goal_on_pairs)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible lower bounds."""
        t1 = self._goal_displacement_cost(state)
        t2 = self._task_network_cost(state, task_network)
        t3 = self._unsatisfied_goals_count(state)
        
        return max(t1, t2, t3)
    
    def _unsatisfied_goals_count(self, state):
        """Simple admissible bound: count unsatisfied goal on-relations.
        Each unsatisfied goal needs at least 2 actions (lift + drop)."""
        count = 0
        for (crate, surface, fid) in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                count += 1
        return count * 2
    
    def _goal_displacement_cost(self, state):
        """For each unsatisfied goal on(crate, surface), estimate minimum actions needed.
        
        If crate is at the same place as surface: min 2 actions (lift + drop)
        But we also need to clear crate and clear surface first.
        If crate is at different place: min 5 actions (lift + load + drive + unload + drop)
        If crate is in a truck: min 2 actions (unload + drop)
        If crate is being lifted: min 1 action (drop)
        
        We also add clearing costs.
        """
        total_cost = 0
        
        for (crate, surface, fid) in self.goal_on_pairs:
            if (state >> fid) & 1:
                # Goal already satisfied
                continue
            
            cost = 0
            
            # Determine crate's current status
            crate_in_truck = False
            crate_being_lifted = False
            crate_place = None
            
            # Check if crate is being lifted by any hoist
            for hoist in self.hoists:
                lift_key = (hoist, crate)
                if lift_key in self.lifting_fact_ids:
                    lfid = self.lifting_fact_ids[lift_key]
                    if (state >> lfid) & 1:
                        crate_being_lifted = True
                        # Hoist location is the crate's effective location
                        crate_place = self.hoist_place.get(hoist)
                        break
            
            if not crate_being_lifted:
                # Check if crate is in any truck
                for truck in self.trucks:
                    in_key = (crate, truck)
                    if in_key in self.in_fact_ids:
                        ifid = self.in_fact_ids[in_key]
                        if (state >> ifid) & 1:
                            crate_in_truck = True
                            break
                
                if not crate_in_truck:
                    # Crate is at some place - find which
                    for place in self.places:
                        at_key = (crate, place)
                        if at_key in self.at_fact_ids:
                            afid = self.at_fact_ids[at_key]
                            if (state >> afid) & 1:
                                crate_place = place
                                break
            
            # Determine surface's location
            surface_place = None
            if surface.startswith("pallet"):
                surface_place = self.surface_place.get(surface)
            else:
                # Surface is a crate - find its current location
                for place in self.places:
                    at_key = (surface, place)
                    if at_key in self.at_fact_ids:
                        afid = self.at_fact_ids[at_key]
                        if (state >> afid) & 1:
                            surface_place = place
                            break
            
            if crate_being_lifted:
                # Just need to drop (possibly after driving if different place)
                if crate_place == surface_place and surface_place is not None:
                    cost = 1  # just drop
                else:
                    cost = 1  # drop (conservatively; might need transport)
            elif crate_in_truck:
                # Need unload + drop = 2 minimum
                cost = 2
            else:
                # Crate is on some surface at some place
                if crate_place is not None and surface_place is not None:
                    if crate_place == surface_place:
                        # Same place: lift + drop = 2
                        cost = 2
                    else:
                        # Different place: lift + load + drive + unload + drop = 5
                        cost = 5
                else:
                    # Unknown location, conservative estimate
                    cost = 2
            
            # Add clearing cost for crate (need it to be clear to lift it)
            if not crate_being_lifted and not crate_in_truck:
                clear_crate_id = self.clear_fact_ids.get(crate)
                if clear_crate_id is not None:
                    if not ((state >> clear_crate_id) & 1):
                        # Crate is not clear - something is on top
                        # Need at least 4 actions to clear one crate off (lift+load)
                        # But we'll be conservative with 2 (lift + put somewhere)
                        cost += 2
            
            # Add clearing cost for surface (need it to be clear to drop onto)
            if not crate_being_lifted or crate_place != surface_place:
                clear_surface_id = self.clear_fact_ids.get(surface)
                if clear_surface_id is not None:
                    if not ((state >> clear_surface_id) & 1):
                        # Check if crate is already on surface (which means it's satisfied)
                        on_key = (crate, surface)
                        on_fid = self.on_fact_ids.get(on_key)
                        # Surface not clear and goal not satisfied
                        # Need to clear surface first: at least 2 actions per blocking crate
                        cost += 2
            
            total_cost += cost
        
        return total_cost
    
    def _task_network_cost(self, state, task_network):
        """Estimate minimum primitive actions from remaining task network.
        
        For each task, estimate its minimum decomposition cost based on
        current state."""
        if not task_network:
            return 0
        
        total = 0
        for task_or_op in task_network:
            name = task_or_op.name
            
            # Check if it's an operator (primitive action)
            if hasattr(task_or_op, 'add_effects'):
                # It's a primitive operator
                task_prefix = self._parse_task_name(name)
                if task_prefix == "nop":
                    total += 0
                else:
                    total += 1
            else:
                # It's an abstract task - estimate decomposition cost
                task_prefix = self._parse_task_name(name)
                cost = self._estimate_abstract_task_cost(task_prefix, name, state)
                total += cost
        
        return total
    
    def _estimate_abstract_task_cost(self, task_prefix, full_name, state):
        """Estimate minimum number of primitive actions for an abstract task.
        
        Uses domain knowledge about task decomposition structure."""
        
        if task_prefix == "do_put_on":
            # Parse arguments: do_put_on[crate, surface]
            bracket_idx = full_name.find("[")
            if bracket_idx > 0 and full_name.endswith("]"):
                args_str = full_name[bracket_idx+1:-1]
                args = args_str.split(",")
                if len(args) == 2:
                    crate, surface = args[0].strip(), args[1].strip()
                    return self._estimate_put_on_cost(crate, surface, state)
            return 2  # minimum: lift + drop
        
        elif task_prefix == "do_clear":
            # Parse arguments: do_clear[surface, place]
            bracket_idx = full_name.find("[")
            if bracket_idx > 0 and full_name.endswith("]"):
                args_str = full_name[bracket_idx+1:-1]
                args = args_str.split(",")
                if len(args) == 2:
                    surface = args[0].strip()
                    clear_id = self.clear_fact_ids.get(surface)
                    if clear_id is not None and (state >> clear_id) & 1:
                        return 0  # already clear
                    return 4  # lift + load (at minimum to clear one crate off)
            return 0
        
        elif task_prefix == "do_get_truck":
            # Parse: do_get_truck[truck, place]
            bracket_idx = full_name.find("[")
            if bracket_idx > 0 and full_name.endswith("]"):
                args_str = full_name[bracket_idx+1:-1]
                args = args_str.split(",")
                if len(args) == 2:
                    truck, place = args[0].strip(), args[1].strip()
                    at_key = (truck, place)
                    if at_key in self.at_fact_ids:
                        afid = self.at_fact_ids[at_key]
                        if (state >> afid) & 1:
                            return 0  # truck already there
                    return 1  # one Drive action
            return 0
        
        elif task_prefix == "do_lift_crate":
            # Minimum: 1 action (Lift) or 2 (Drive truck + Unload)
            return 1
        
        elif task_prefix == "do_load_truck":
            # do_get_truck + do_clear + Lift + Load = at least 2 actions
            return 2
        
        elif task_prefix == "do_unload_truck":
            # do_clear + Unload + Drop = at least 2 actions
            return 2
        
        return 1  # default conservative estimate
    
    def _estimate_put_on_cost(self, crate, surface, state):
        """Estimate minimum actions for do_put_on(crate, surface)."""
        
        # Check if already satisfied (m0)
        on_key = (crate, surface)
        on_fid = self.on_fact_ids.get(on_key)
        if on_fid is not None and (state >> on_fid) & 1:
            return 0  # already on target
        
        cost = 0
        
        # Check crate status
        crate_in_truck = False
        crate_at_place = None
        crate_being_lifted = False
        
        for hoist in self.hoists:
            lift_key = (hoist, crate)
            if lift_key in self.lifting_fact_ids:
                lfid = self.lifting_fact_ids[lift_key]
                if (state >> lfid) & 1:
                    crate_being_lifted = True
                    crate_at_place = self.hoist_place.get(hoist)
                    break
        
        if not crate_being_lifted:
            for truck in self.trucks:
                in_key = (crate, truck)
                if in_key in self.in_fact_ids:
                    ifid = self.in_fact_ids[in_key]
                    if (state >> ifid) & 1:
                        crate_in_truck = True
                        break
            
            if not crate_in_truck:
                for place in self.places:
                    at_key = (crate, place)
                    if at_key in self.at_fact_ids:
                        afid = self.at_fact_ids[at_key]
                        if (state >> afid) & 1:
                            crate_at_place = place
                            break
        
        # Find surface location
        surface_place = None
        if surface.startswith("pallet"):
            surface_place = self.surface_place.get(surface)
        else:
            for place in self.places:
                at_key = (surface, place)
                if at_key in self.at_fact_ids:
                    afid = self.at_fact_ids[at_key]
                    if (state >> afid) & 1:
                        surface_place = place
                        break
        
        if crate_being_lifted:
            # Already lifted, just need drop (+ maybe drive)
            if crate_at_place == surface_place and surface_place is not None:
                cost = 1  # drop
            else:
                # Need to load into truck, drive, unload, drop
                cost = 4
        elif crate_in_truck:
            # m2: do_get_truck + do_clear(s2) + Unload + Drop
            # minimum 2 (unload + drop), possibly more if truck not at right place
            cost = 2
        elif crate_at_place is not None:
            if crate_at_place == surface_place:
                # m1: do_clear(c) + do_clear(s2) + do_lift_crate + Drop
                # minimum 2 (lift + drop)
                cost = 2
                
                # Add cost for clearing crate
                clear_crate_id = self.clear_fact_ids.get(crate)
                if clear_crate_id is not None and not ((state >> clear_crate_id) & 1):
                    cost += 4  # at least lift+load to clear the thing on top
                
                # Add cost for clearing surface
                clear_surface_id = self.clear_fact_ids.get(surface)
                if clear_surface_id is not None and not ((state >> clear_surface_id) & 1):
                    cost += 4  # at least lift+load to clear the thing on top
            else:
                # m3: do_load_truck + Drive + do_unload_truck
                # minimum: (lift + load) + drive + (unload + drop) = 5
                cost = 5
                
                # Add clearing costs
                clear_crate_id = self.clear_fact_ids.get(crate)
                if clear_crate_id is not None and not ((state >> clear_crate_id) & 1):
                    cost += 4
                
                clear_surface_id = self.clear_fact_ids.get(surface)
                if clear_surface_id is not None and not ((state >> clear_surface_id) & 1):
                    cost += 4
        else:
            # Crate location unknown - conservative
            cost = 2
        
        return cost
