"""Domain-dependent heuristic for the Depots HDDL domain.

Bottleneck: The serializing constraint is surface clearing - to place a crate on
a surface, that surface must be clear, and clearing requires moving blocking crates
(each costing at least 2 actions). Combined with transport between places (drive +
load/unload), this creates cascading dependencies.

Admissible lower bounds:
1. Displacement cost: Each unsatisfied on(c,s) goal needs >= 2 actions (lift+drop)
   if c is at the right place, >= 4 actions (lift+load+drive+unload+drop - 1 shared)
   if c needs transport. Admissible: these are physically unavoidable steps.
2. Blocking cost: Each crate sitting on a goal-target surface that isn't the goal
   crate must be removed (>= 2 actions). Admissible: physically must be moved.
3. Task network estimate: For each remaining abstract task, estimate minimum
   decomposition cost using current state. Admissible: uses minimum possible costs.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic2(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_2"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_id = {}       # (crate, surface) -> global_id
        self.at_fact_id = {}       # (locatable, place) -> global_id
        self.in_fact_id = {}       # (crate, truck) -> global_id
        self.clear_fact_id = {}    # surface -> global_id
        self.lifting_fact_id = {}  # (hoist, crate) -> global_id
        self.available_fact_id = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on = {}          # crate -> target_surface for goal on(c,s)
        self.goal_on_facts = {}    # (crate, surface) -> fact_global_id
        
        # Place info
        self.crate_objects = set()
        self.surface_objects = set()
        self.place_objects = set()
        self.truck_objects = set()
        self.hoist_objects = set()
        self.hoist_at_place = {}   # place -> list of hoists
        
        # For stack analysis
        self.on_pairs_by_surface = {}  # surface -> (crate, fact_id) - what's on this surface
        
        # Task name patterns for cost estimation
        self.task_name_cache = {}
        
        # Abstract task type identifiers
        self.do_put_on_ids = set()
        self.do_clear_ids = set()
        self.do_get_truck_ids = set()
        self.do_lift_crate_ids = set()
        self.do_load_truck_ids = set()
        self.do_unload_truck_ids = set()
        
        # Operator type identifiers
        self.drive_op_ids = set()
        self.lift_op_ids = set()
        self.drop_op_ids = set()
        self.load_op_ids = set()
        self.unload_op_ids = set()
        self.nop_op_ids = set()

    def _parse_fact_name(self, name: str):
        """Parse a grounded fact name like '+on[crate0,pallet1]' into (predicate, [args])."""
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

    def _parse_component_name(self, name: str):
        """Parse a component name like 'drive[truck0,depot0,distributor1]'."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # No brackets
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Preprocess all facts and build lookup structures."""
        self.model = model
        
        # Build fact lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
        
        # Extract object types from fact patterns
        for (obj, place) in self.at_fact_id:
            if obj.startswith("crate"):
                self.crate_objects.add(obj)
                self.place_objects.add(place)
            elif obj.startswith("truck"):
                self.truck_objects.add(obj)
                self.place_objects.add(place)
            elif obj.startswith("hoist"):
                self.hoist_objects.add(obj)
                self.place_objects.add(place)
            elif obj.startswith("pallet"):
                self.surface_objects.add(obj)
                self.place_objects.add(place)
        
        self.surface_objects.update(self.crate_objects)
        
        # Build on_pairs_by_surface: for each surface, which (crate, fact_id) can be on it
        for (crate, surface), fid in self.on_fact_id.items():
            if surface not in self.on_pairs_by_surface:
                self.on_pairs_by_surface[surface] = []
            self.on_pairs_by_surface[surface].append((crate, fid))
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                self.goal_on[crate] = surface
                fid = self.on_fact_id.get((crate, surface))
                if fid is not None:
                    self.goal_on_facts[(crate, surface)] = fid
        
        # Classify abstract tasks
        for at in model.abstract_tasks:
            ttype, args = self._parse_component_name(at.name)
            if ttype == "do_put_on":
                self.do_put_on_ids.add(at.global_id)
            elif ttype == "do_clear":
                self.do_clear_ids.add(at.global_id)
            elif ttype == "do_get_truck":
                self.do_get_truck_ids.add(at.global_id)
            elif ttype == "do_lift_crate":
                self.do_lift_crate_ids.add(at.global_id)
            elif ttype == "do_load_truck":
                self.do_load_truck_ids.add(at.global_id)
            elif ttype == "do_unload_truck":
                self.do_unload_truck_ids.add(at.global_id)
        
        # Classify operators
        for op in model.operators:
            ttype, args = self._parse_component_name(op.name)
            if ttype == "drive":
                self.drive_op_ids.add(op.global_id)
            elif ttype == "lift":
                self.lift_op_ids.add(op.global_id)
            elif ttype == "drop":
                self.drop_op_ids.add(op.global_id)
            elif ttype == "load":
                self.load_op_ids.add(op.global_id)
            elif ttype == "unload":
                self.unload_op_ids.add(op.global_id)
            elif ttype == "nop":
                self.nop_op_ids.add(op.global_id)
        
        # Precompute static hoist locations (hoists don't move in depots)
        # Check initial state for hoist locations
        initial_state = initial_node.state
        for hoist in self.hoist_objects:
            for place in self.place_objects:
                fid = self.at_fact_id.get((hoist, place))
                if fid is not None and (initial_state >> fid) & 1:
                    if place not in self.hoist_at_place:
                        self.hoist_at_place[place] = []
                    self.hoist_at_place[place].append(hoist)
                    break
        
        # Precompute: for each place, which pallets are there (static)
        self.pallets_at_place = {}
        for (obj, place), fid in self.at_fact_id.items():
            if obj.startswith("pallet"):
                if (initial_state >> fid) & 1:
                    if place not in self.pallets_at_place:
                        self.pallets_at_place[place] = []
                    self.pallets_at_place[place].append(obj)
        
        # Build task_name_cache for abstract tasks and operators
        for at in model.abstract_tasks:
            self.task_name_cache[at.global_id] = self._parse_component_name(at.name)
        for op in model.operators:
            self.task_name_cache[op.global_id] = self._parse_component_name(op.name)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        """Compute max of multiple admissible lower bounds."""
        h1 = self._goal_displacement_cost(state)
        h2 = self._task_network_cost(state, task_network)
        h3 = self._blocking_cost(state)
        return max(h1, h2, h3)

    def _get_crate_location(self, state, crate):
        """Find which place a crate is at, or None if not at any place (in truck or being lifted)."""
        for place in self.place_objects:
            fid = self.at_fact_id.get((crate, place))
            if fid is not None and (state >> fid) & 1:
                return place
        return None

    def _get_surface_location(self, state, surface):
        """Find which place a surface is at. For pallets this is static."""
        for place in self.place_objects:
            fid = self.at_fact_id.get((surface, place))
            if fid is not None and (state >> fid) & 1:
                return place
        return None

    def _is_crate_in_truck(self, state, crate):
        """Check if crate is in any truck."""
        for truck in self.truck_objects:
            fid = self.in_fact_id.get((crate, truck))
            if fid is not None and (state >> fid) & 1:
                return True
        return False

    def _is_crate_being_lifted(self, state, crate):
        """Check if crate is being lifted by any hoist."""
        for hoist in self.hoist_objects:
            fid = self.lifting_fact_id.get((hoist, crate))
            if fid is not None and (state >> fid) & 1:
                return True
        return False

    def _goal_displacement_cost(self, state):
        """Lower bound based on unsatisfied goal on(c,s) predicates.
        
        For each unsatisfied goal on(c,s):
        - If crate is being lifted at the target place: 1 action (drop)
        - If crate is at target place on wrong surface: 2 actions (lift + drop)
        - If crate is in a truck: 3 actions (unload + ... but minimum is complex)
        - If crate is at different place: 4 actions (lift + load + unload + drop) + 1 drive
        - Plus clearing cost if target surface is not clear
        """
        total_cost = 0
        
        for (crate, target_surface), fid in self.goal_on_facts.items():
            # Check if goal already satisfied
            if (state >> fid) & 1:
                continue
            
            # Goal not satisfied - compute minimum cost
            # Check if crate is being lifted
            is_lifted = False
            lifted_at_place = None
            for hoist in self.hoist_objects:
                lfid = self.lifting_fact_id.get((hoist, crate))
                if lfid is not None and (state >> lfid) & 1:
                    is_lifted = True
                    # Find where this hoist is
                    for place in self.place_objects:
                        hfid = self.at_fact_id.get((hoist, place))
                        if hfid is not None and (state >> hfid) & 1:
                            lifted_at_place = place
                            break
                    break
            
            # Find target surface location
            target_place = self._get_surface_location(state, target_surface)
            
            if is_lifted:
                if lifted_at_place == target_place:
                    # Just need to drop (if surface is clear) - 1 action
                    cost = 1
                else:
                    # Need load + drive + unload + drop - 4 actions
                    cost = 4
            elif self._is_crate_in_truck(state, crate):
                # Need: drive (maybe) + unload + drop = 2-3 actions minimum
                cost = 2  # unload + drop at minimum
            else:
                # Crate is at some place on some surface
                crate_place = self._get_crate_location(state, crate)
                if crate_place is not None:
                    if crate_place == target_place:
                        # Same place: lift + drop = 2
                        cost = 2
                    else:
                        # Different place: lift + load + drive + unload + drop = 5
                        cost = 5
                else:
                    # Shouldn't happen but fallback
                    cost = 2
            
            # Add clearing cost for crate itself (if not clear, need to clear what's on top)
            if not is_lifted and not self._is_crate_in_truck(state, crate):
                clear_fid = self.clear_fact_id.get(crate)
                if clear_fid is not None and not ((state >> clear_fid) & 1):
                    # Crate is not clear - at least 2 more actions to clear it (lift + load the thing on top)
                    cost += 2
            
            # Check if target surface is clear
            clear_target_fid = self.clear_fact_id.get(target_surface)
            if clear_target_fid is not None and not ((state >> clear_target_fid) & 1):
                # Need to check if what's on target is the goal crate itself
                # If not, add clearing cost
                is_self_on_target = False
                on_fid = self.on_fact_id.get((crate, target_surface))
                if on_fid is not None and (state >> on_fid) & 1:
                    is_self_on_target = True
                
                if not is_self_on_target:
                    # Something else is on the target - need at least 2 actions to clear
                    cost += 2
            
            total_cost += cost
        
        return total_cost

    def _blocking_cost(self, state):
        """Lower bound based on crates blocking goal target surfaces.
        
        For each goal on(c, s), if s is not clear and the thing on s is not c,
        that blocker must be moved. If the blocker itself has a goal elsewhere,
        we count its minimum move cost.
        """
        blocking_cost = 0
        
        for (crate, target_surface), fid in self.goal_on_facts.items():
            # Check if goal already satisfied
            if (state >> fid) & 1:
                continue
            
            # Check if target surface is clear
            clear_fid = self.clear_fact_id.get(target_surface)
            if clear_fid is None:
                continue
            if (state >> clear_fid) & 1:
                continue  # Already clear
            
            # Find what's on the target surface
            if target_surface in self.on_pairs_by_surface:
                for (blocker, on_fid) in self.on_pairs_by_surface[target_surface]:
                    if (state >> on_fid) & 1:
                        # This blocker is on the target surface
                        if blocker != crate:
                            # Must move this blocker: at least lift + load = 2
                            blocking_cost += 2
                            # If blocker also not clear, need to clear it first
                            blocker_clear_fid = self.clear_fact_id.get(blocker)
                            if blocker_clear_fid is not None and not ((state >> blocker_clear_fid) & 1):
                                blocking_cost += 2  # Clear the blocker's top
                        break
        
        return blocking_cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network.
        
        For each task in the network, estimate minimum actions needed.
        """
        if not task_network:
            return 0
        
        total = 0
        # Limit iteration for very large networks to avoid timeout
        max_tasks = min(len(task_network), 500)
        
        for i in range(max_tasks):
            task = task_network[i]
            gid = task.global_id
            
            # Check if it's an operator (primitive)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                if gid in self.nop_op_ids:
                    total += 0
                else:
                    total += 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task - estimate minimum decomposition cost
                total += self._estimate_abstract_task_cost(state, gid)
        
        return total

    def _estimate_abstract_task_cost(self, state, task_gid):
        """Estimate minimum cost to decompose an abstract task."""
        cached = self.task_name_cache.get(task_gid)
        if cached is None:
            return 1
        
        ttype, args = cached
        
        if ttype == "do_put_on":
            return self._estimate_do_put_on(state, args)
        elif ttype == "do_clear":
            return self._estimate_do_clear(state, args)
        elif ttype == "do_get_truck":
            return self._estimate_do_get_truck(state, args)
        elif ttype == "do_lift_crate":
            return self._estimate_do_lift_crate(state, args)
        elif ttype == "do_load_truck":
            return self._estimate_do_load_truck(state, args)
        elif ttype == "do_unload_truck":
            return self._estimate_do_unload_truck(state, args)
        
        return 1

    def _estimate_do_put_on(self, state, args):
        """Estimate cost of do_put_on(crate, surface).
        
        Methods:
        m0: already on target -> 0 (nop)
        m1: crate at some place -> clear_crate + clear_surface + lift + drop >= 2
        m2: crate in truck -> get_truck + clear_surface + unload + drop >= 2  
        m3: general -> load_truck + drive + unload_truck >= 5
        """
        if len(args) < 2:
            return 1
        crate, surface = args[0], args[1]
        
        # Check m0: already on target
        on_fid = self.on_fact_id.get((crate, surface))
        if on_fid is not None and (state >> on_fid) & 1:
            return 0  # nop
        
        # Check if crate is being lifted
        for hoist in self.hoist_objects:
            lfid = self.lifting_fact_id.get((hoist, crate))
            if lfid is not None and (state >> lfid) & 1:
                # Crate is being lifted - just need drop (if at right place)
                # But do_put_on doesn't have a method for this directly
                # It would go through m1 path minimum
                return 1
        
        # Check if crate is in a truck (m2 path)
        if self._is_crate_in_truck(state, crate):
            # m2: get_truck + clear_surface + unload + drop
            # Minimum: 0 (truck already there) + 0 (surface clear) + 1 + 1 = 2
            cost = 2  # unload + drop minimum
            # Check if surface is clear
            clear_fid = self.clear_fact_id.get(surface)
            if clear_fid is not None and not ((state >> clear_fid) & 1):
                cost += 2  # Need to clear surface (at minimum lift + load = 2)
            return cost
        
        # Check if crate is at the same place as surface (m1 path potential)
        crate_place = self._get_crate_location(state, crate)
        surface_place = self._get_surface_location(state, surface)
        
        if crate_place is not None and crate_place == surface_place:
            # m1: clear_crate + clear_surface + lift + drop
            cost = 2  # lift + drop minimum
            
            # Check if crate is clear
            clear_crate_fid = self.clear_fact_id.get(crate)
            if clear_crate_fid is not None and not ((state >> clear_crate_fid) & 1):
                cost += 4  # Clear crate (recursive: clear + lift + get_truck + load = 4)
            
            # Check if surface is clear
            clear_surface_fid = self.clear_fact_id.get(surface)
            if clear_surface_fid is not None and not ((state >> clear_surface_fid) & 1):
                # Check if what's on surface is the crate itself
                is_self = False
                self_on_fid = self.on_fact_id.get((crate, surface))
                if self_on_fid is not None and (state >> self_on_fid) & 1:
                    is_self = True
                if not is_self:
                    cost += 4  # Clear surface
            
            return cost
        
        # m3 path: load_truck + drive + unload_truck
        # load_truck = get_truck + clear_crate + lift + load = min 2
        # drive = 1
        # unload_truck = clear_surface + unload + drop = min 2
        cost = 5  # minimum for transport path
        
        # Add clearing costs
        if crate_place is not None:
            clear_crate_fid = self.clear_fact_id.get(crate)
            if clear_crate_fid is not None and not ((state >> clear_crate_fid) & 1):
                cost += 4
        
        if surface_place is not None:
            clear_surface_fid = self.clear_fact_id.get(surface)
            if clear_surface_fid is not None and not ((state >> clear_surface_fid) & 1):
                cost += 4
        
        return cost

    def _estimate_do_clear(self, state, args):
        """Estimate cost of do_clear(surface, place).
        
        m4: surface is clear and at place -> 0 (nop)
        m5: not clear -> clear_top + lift + get_truck + load >= 4
        """
        if len(args) < 2:
            return 0
        surface, place = args[0], args[1]
        
        clear_fid = self.clear_fact_id.get(surface)
        if clear_fid is not None and (state >> clear_fid) & 1:
            return 0  # Already clear
        
        # Need to clear: at minimum lift + load = 2, plus get_truck might need drive = 1
        # Plus recursive clear of top crate
        cost = 2  # lift + load minimum
        
        # Find what's on this surface to check if it's clear
        if surface in self.on_pairs_by_surface:
            for (top_crate, on_fid) in self.on_pairs_by_surface[surface]:
                if (state >> on_fid) & 1:
                    top_clear_fid = self.clear_fact_id.get(top_crate)
                    if top_clear_fid is not None and not ((state >> top_clear_fid) & 1):
                        cost += 4  # Need to clear the top crate too (recursive)
                    break
        
        return cost

    def _estimate_do_get_truck(self, state, args):
        """Estimate cost of do_get_truck(truck, place).
        
        m6: truck at place -> 0 (nop)
        m7: truck not at place -> 1 (drive)
        """
        if len(args) < 2:
            return 0
        truck, place = args[0], args[1]
        
        at_fid = self.at_fact_id.get((truck, place))
        if at_fid is not None and (state >> at_fid) & 1:
            return 0
        return 1

    def _estimate_do_lift_crate(self, state, args):
        """Estimate cost of do_lift_crate(crate, place, hoist).
        
        m8: crate in truck -> get_truck + unload >= 1
        m9: crate on surface at place -> lift = 1
        """
        if len(args) < 3:
            return 1
        crate, place, hoist = args[0], args[1], args[2]
        
        # Check if crate is at the place
        at_fid = self.at_fact_id.get((crate, place))
        if at_fid is not None and (state >> at_fid) & 1:
            return 1  # Just lift
        
        # Crate in truck: get_truck + unload
        return 2

    def _estimate_do_load_truck(self, state, args):
        """Estimate cost of do_load_truck(crate, surface, place, truck).
        
        m10: get_truck + clear_crate + lift + load >= 2
        """
        if len(args) < 4:
            return 2
        crate = args[0]
        
        cost = 2  # lift + load minimum
        
        # Check if truck is at the place
        truck, place = args[3], args[2]
        truck_at_fid = self.at_fact_id.get((truck, place))
        if truck_at_fid is not None and not ((state >> truck_at_fid) & 1):
            cost += 1  # drive
        
        # Check if crate is clear
        clear_fid = self.clear_fact_id.get(crate)
        if clear_fid is not None and not ((state >> clear_fid) & 1):
            cost += 4  # clear crate
        
        return cost

    def _estimate_do_unload_truck(self, state, args):
        """Estimate cost of do_unload_truck(crate, surface, place, truck).
        
        m11: clear_surface + unload + drop >= 2
        """
        if len(args) < 4:
            return 2
        surface = args[1]
        
        cost = 2  # unload + drop minimum
        
        # Check if surface is clear
        clear_fid = self.clear_fact_id.get(surface)
        if clear_fid is not None and not ((state >> clear_fid) & 1):
            cost += 4  # clear surface
        
        return cost
