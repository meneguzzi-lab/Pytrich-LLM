# Depots Domain Heuristic
# Bottleneck: Crates must be moved between places via trucks (lift->load->drive->unload->drop = 5 actions min),
# and hoists serialize operations at each place. Stack clearing adds recursive cost.
#
# Admissible lower bounds:
# T1: Per unsatisfied goal on(c,s): min 2 actions if same place, min 5 if different place
# T2: Task network lower bound: each abstract task contributes state-aware minimum cost
# T3: Crates needing transport: each cross-place move needs at least 5 actions
#
# max(T1, T2, T3) is admissible since each term individually underestimates.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic15(Heuristic):
    """Domain-dependent heuristic for the Depots domain.
    
    Uses multi-component max of:
    1. State-aware unsatisfied goal cost
    2. Task network decomposition lower bound
    3. Cross-place transport lower bound
    """

    def __init__(self, name="depot_anthropicopus46_heuristic_15"):
        super().__init__(name)
        # Fact ID maps: predicate -> (args_tuple) -> global_id
        self.on_fact_id = {}      # (crate, surface) -> gid
        self.at_fact_id = {}      # (locatable, place) -> gid
        self.in_fact_id = {}      # (crate, truck) -> gid
        self.clear_fact_id = {}   # (surface,) -> gid
        self.lifting_fact_id = {} # (hoist, crate) -> gid
        self.available_fact_id = {} # (hoist,) -> gid
        
        # Goal information
        self.goal_on_pairs = []   # list of (crate, surface, fact_gid)
        
        # Object location tracking
        self.crate_initial_places = {}  # crate -> set of places where at(crate, place) facts exist
        self.surface_places = {}        # surface -> set of places where at(surface, place) facts exist
        self.hoist_places = {}          # hoist -> place
        self.truck_at_facts = {}        # truck -> list of (place, gid)
        
        # All crates, surfaces, places, trucks, hoists
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        
        # For task network analysis
        self.task_name_to_type = {}  # abstract task name prefix -> type string
        
        # Precomputed: for each on(c,s) goal, the places where s is located (static)
        self.goal_target_places = {}  # (crate, surface) -> set of places
        
        # Crate -> all on facts involving this crate
        self.crate_on_facts = {}  # crate -> list of (surface, gid)
        
        # Surface -> on facts where something is on this surface
        self.surface_on_top = {}  # surface -> list of (crate, gid)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like +on[crate0,pallet1] into (predicate, [args])."""
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

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts once, build lookup structures for fast heuristic computation."""
        self.model = model
        
        # Parse all facts and build ID maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                # Track crate's on relationships
                if args[0] not in self.crate_on_facts:
                    self.crate_on_facts[args[0]] = []
                self.crate_on_facts[args[0]].append((args[1], f.global_id))
                # Track what's on top of each surface
                if args[1] not in self.surface_on_top:
                    self.surface_on_top[args[1]] = []
                self.surface_on_top[args[1]].append((args[0], f.global_id))
                self.crates.add(args[0])
                self.surfaces.add(args[0])  # crates are also surfaces
                self.surfaces.add(args[1])
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
                # Determine object type from name
                obj = args[0]
                place = args[1]
                if obj.startswith("crate"):
                    if obj not in self.crate_initial_places:
                        self.crate_initial_places[obj] = set()
                    self.crate_initial_places[obj].add(place)
                    self.crates.add(obj)
                elif obj.startswith("pallet"):
                    if obj not in self.surface_places:
                        self.surface_places[obj] = set()
                    self.surface_places[obj].add(place)
                    self.surfaces.add(obj)
                elif obj.startswith("hoist"):
                    self.hoist_places[obj] = place
                    self.hoists.add(obj)
                elif obj.startswith("truck"):
                    if obj not in self.truck_at_facts:
                        self.truck_at_facts[obj] = []
                    self.truck_at_facts[obj].append((place, f.global_id))
                    self.trucks.add(obj)
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
                self.hoists.add(args[0])

        # Also record surface (pallet) locations - pallets have static at facts
        for (obj, place), gid in self.at_fact_id.items():
            if obj.startswith("pallet"):
                if obj not in self.surface_places:
                    self.surface_places[obj] = set()
                self.surface_places[obj].add(place)
            elif obj.startswith("crate"):
                if obj not in self.surface_places:
                    self.surface_places[obj] = set()
                self.surface_places[obj].add(place)

        # Parse goal on-facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        self.goal_on_pairs = []
        self.goal_on_set = set()  # (crate, surface) pairs that are goals
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                gid = self.on_fact_id.get((crate, surface))
                if gid is not None:
                    self.goal_on_pairs.append((crate, surface, gid))
                    self.goal_on_set.add((crate, surface))

        # Precompute: for each goal on(c,s), the static places of surface s
        # For pallets, place is static. For crates used as surfaces, place can change.
        for crate, surface, gid in self.goal_on_pairs:
            places = set()
            if surface.startswith("pallet"):
                # Pallets have static locations
                if surface in self.surface_places:
                    places = self.surface_places[surface]
            self.goal_target_places[(crate, surface)] = places

        # Build a set of abstract task name prefixes for task network analysis
        # We'll categorize tasks by their HDDL task type
        for at in model.abstract_tasks:
            name = at.name
            # Abstract task names look like: do_put_on[crate0,pallet1]
            bracket_idx = name.find("[")
            if bracket_idx > 0:
                task_type = name[:bracket_idx]
            else:
                task_type = name
            self.task_name_to_type[at.global_id] = task_type

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for the given node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        """Compute max of multiple admissible lower bounds."""
        t1 = self._unsatisfied_goal_cost(state)
        t2 = self._task_network_cost(state, task_network)
        return max(t1, t2)

    def _get_crate_location(self, state, crate):
        """Determine where a crate currently is: returns (location_type, place_or_truck).
        location_type: 'at' (at a place), 'in' (in a truck), 'lifting' (being lifted), 'unknown'
        Uses fast bitwise checks."""
        # Check at(crate, place) facts
        for (obj, place), gid in self.at_fact_id.items():
            if obj == crate:
                if (state >> gid) & 1:
                    return ('at', place)
        # Check in(crate, truck) facts
        for (obj, truck), gid in self.in_fact_id.items():
            if obj == crate:
                if (state >> gid) & 1:
                    return ('in', truck)
        # Check lifting(hoist, crate) facts
        for (hoist, obj), gid in self.lifting_fact_id.items():
            if obj == crate:
                if (state >> gid) & 1:
                    return ('lifting', hoist)
        return ('unknown', None)

    def _get_surface_place(self, surface):
        """Get the static place of a surface (works for pallets). Returns None if unknown."""
        if surface in self.surface_places:
            places = self.surface_places[surface]
            if places:
                return next(iter(places))  # pallets have exactly one place
        return None

    def _unsatisfied_goal_cost(self, state):
        """Compute lower bound based on unsatisfied on-goals.
        
        For each unsatisfied on(c,s) goal:
        - If c is already at the same place as s: min 2 actions (lift + drop)
        - If c is at a different place: min 5 actions (lift + load + drive + unload + drop)
        - If c is in a truck: min 2 actions (unload + drop) 
        - If c is being lifted at same place: min 1 action (drop)
        - If the target surface s is not clear, we need additional clearing actions
        
        This is admissible because each goal independently needs at least this many actions.
        We take the sum, but discount by considering that some actions might be shared.
        Actually, for admissibility with sum, we need to be careful. Let's use a
        conservative per-goal estimate and sum a fraction.
        """
        total_cost = 0
        
        for crate, surface, gid in self.goal_on_pairs:
            # Check if goal already satisfied
            if (state >> gid) & 1:
                # Goal is currently true - but might be undone
                # For now, count as 0 (optimistic)
                continue
            
            # Goal not satisfied - estimate minimum cost
            min_cost = 2  # At minimum: lift + drop (or unload + drop)
            
            # Check if crate is in a truck
            crate_in_truck = False
            for (obj, truck), in_gid in self.in_fact_id.items():
                if obj == crate and (state >> in_gid) & 1:
                    crate_in_truck = True
                    # Crate is in truck - need unload + drop = 2 minimum
                    # But truck might need to drive to target place
                    if surface.startswith("pallet"):
                        target_place = self._get_surface_place(surface)
                        if target_place is not None:
                            # Check if truck is at target place
                            truck_at_target = False
                            for (t, p), at_gid in self.at_fact_id.items():
                                if t == truck and p == target_place:
                                    if (state >> at_gid) & 1:
                                        truck_at_target = True
                                    break
                            if not truck_at_target:
                                min_cost = 3  # drive + unload + drop
                    break
            
            if not crate_in_truck:
                # Check if crate is being lifted
                crate_being_lifted = False
                for (hoist, obj), lift_gid in self.lifting_fact_id.items():
                    if obj == crate and (state >> lift_gid) & 1:
                        crate_being_lifted = True
                        # Need at least drop (1), or load+drive+unload+drop (4) if different place
                        min_cost = 1
                        if surface.startswith("pallet"):
                            target_place = self._get_surface_place(surface)
                            hoist_place = self.hoist_places.get(hoist)
                            if target_place and hoist_place and target_place != hoist_place:
                                min_cost = 4  # load + drive + unload + drop
                        break
                
                if not crate_being_lifted:
                    # Crate is at some place - check if same place as target surface
                    crate_place = None
                    for (obj, place), at_gid in self.at_fact_id.items():
                        if obj == crate and (state >> at_gid) & 1:
                            crate_place = place
                            break
                    
                    if crate_place is not None and surface.startswith("pallet"):
                        target_place = self._get_surface_place(surface)
                        if target_place and crate_place != target_place:
                            min_cost = 5  # lift + load + drive + unload + drop
                        else:
                            min_cost = 2  # lift + drop
                    elif crate_place is not None and surface.startswith("crate"):
                        # Target surface is a crate - its location may change
                        # Conservative: assume 2 (same place) 
                        min_cost = 2
            
            # Check if target surface is clear
            if surface in self.clear_fact_id:
                clear_gid = self.clear_fact_id[surface]
                if not ((state >> clear_gid) & 1):
                    # Surface is not clear - need at least 2 more actions to clear it
                    min_cost += 2
            
            # Check if crate itself is clear (needed for lifting)
            if crate in self.clear_fact_id:
                clear_gid = self.clear_fact_id[crate]
                if not ((state >> clear_gid) & 1) and not crate_in_truck:
                    # Crate is not clear - need to clear it first (at least 2 actions per crate on top)
                    min_cost += 2
            
            total_cost += min_cost
        
        return total_cost

    def _task_network_cost(self, state, task_network):
        """Compute lower bound from remaining tasks in the task network.
        
        Each task type has a minimum number of primitive actions:
        - do_put_on: 0 if already satisfied (m0_nop), else at least 2 (lift+drop) or 5 (with transport)
        - do_clear: 0 if already clear, else at least 2 (lift+load)
        - do_get_truck: 0 if truck already there, else 1 (drive)
        - do_lift_crate: at least 1 (lift or unload)
        - do_load_truck: at least 2 (lift+load)
        - do_unload_truck: at least 2 (unload+drop)
        - primitive operators: 1
        """
        from Pytrich.model import Operator as OpClass
        from Pytrich.model import AbstractTask as ATClass
        
        total = 0
        
        for task in task_network:
            if isinstance(task, OpClass):
                # Primitive operator - costs 1 action (unless nop)
                if "nop" in task.name:
                    continue
                total += 1
            elif isinstance(task, ATClass):
                # Abstract task - estimate minimum decomposition cost
                cost = self._estimate_abstract_task_cost(state, task)
                total += cost
        
        return total

    def _estimate_abstract_task_cost(self, state, task):
        """Estimate minimum primitive actions needed to accomplish an abstract task."""
        name = task.name
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
        else:
            return 1  # Unknown task type

        if task_type == "do_put_on":
            return self._estimate_do_put_on(state, args)
        elif task_type == "do_clear":
            return self._estimate_do_clear(state, args)
        elif task_type == "do_get_truck":
            return self._estimate_do_get_truck(state, args)
        elif task_type == "do_lift_crate":
            return 1  # At least one action (Lift or Unload)
        elif task_type == "do_load_truck":
            return 2  # At least Lift + Load
        elif task_type == "do_unload_truck":
            return 2  # At least Unload + Drop
        else:
            return 1

    def _estimate_do_put_on(self, state, args):
        """Estimate cost for do_put_on(crate, surface)."""
        if len(args) < 2:
            return 1
        
        crate, surface = args[0], args[1]
        
        # Check if already satisfied (m0: on(c, s2) -> nop)
        on_gid = self.on_fact_id.get((crate, surface))
        if on_gid is not None and (state >> on_gid) & 1:
            return 0
        
        # Check if crate is in a truck (m2: needs get_truck + clear_s2 + unload + drop)
        crate_in_truck = False
        for (obj, truck), in_gid in self.in_fact_id.items():
            if obj == crate and (state >> in_gid) & 1:
                crate_in_truck = True
                break
        
        if crate_in_truck:
            # m2: do_get_truck + do_clear(s2) + Unload + Drop = at least 2
            # Plus potential drive cost and clear cost
            cost = 2  # Unload + Drop minimum
            # Check if surface is clear
            if surface in self.clear_fact_id:
                clear_gid = self.clear_fact_id[surface]
                if not ((state >> clear_gid) & 1):
                    cost += 2  # Need to clear the surface
            return cost
        
        # Check if crate is at some place (m1: clear_c + clear_s2 + lift_crate + drop)
        crate_at_place = None
        for (obj, place), at_gid in self.at_fact_id.items():
            if obj == crate and (state >> at_gid) & 1:
                crate_at_place = place
                break
        
        if crate_at_place is not None:
            # Check if surface is at the same place
            surface_at_same = False
            if surface.startswith("pallet"):
                target_place = self._get_surface_place(surface)
                if target_place == crate_at_place:
                    surface_at_same = True
            else:
                # Surface is a crate - check dynamically
                for (obj, place), at_gid in self.at_fact_id.items():
                    if obj == surface and place == crate_at_place and (state >> at_gid) & 1:
                        surface_at_same = True
                        break
            
            if surface_at_same:
                # m1: do_clear(c) + do_clear(s2) + do_lift_crate(c) + Drop
                cost = 2  # lift + drop minimum
                # Check if crate is clear
                if crate in self.clear_fact_id:
                    clear_gid = self.clear_fact_id[crate]
                    if not ((state >> clear_gid) & 1):
                        cost += 2  # need to clear crate
                # Check if surface is clear
                if surface in self.clear_fact_id:
                    clear_gid = self.clear_fact_id[surface]
                    if not ((state >> clear_gid) & 1):
                        cost += 2  # need to clear surface
                return cost
            else:
                # Need transport: m3: do_load_truck + Drive + do_unload_truck
                # = (Lift + Load) + Drive + (Unload + Drop) = 5 minimum
                cost = 5
                # Check if crate is clear
                if crate in self.clear_fact_id:
                    clear_gid = self.clear_fact_id[crate]
                    if not ((state >> clear_gid) & 1):
                        cost += 2
                # Check if surface is clear
                if surface in self.clear_fact_id:
                    clear_gid = self.clear_fact_id[surface]
                    if not ((state >> clear_gid) & 1):
                        cost += 2
                return cost
        
        # Crate is being lifted
        # Need at least 1 (drop) or 4 (load+drive+unload+drop)
        return 1

    def _estimate_do_clear(self, state, args):
        """Estimate cost for do_clear(surface, place)."""
        if len(args) < 2:
            return 0
        
        surface, place = args[0], args[1]
        
        # Check if surface is at the place and clear
        surface_at_place = False
        at_gid = self.at_fact_id.get((surface, place))
        if at_gid is not None:
            surface_at_place = (state >> at_gid) & 1
        
        if surface in self.clear_fact_id:
            clear_gid = self.clear_fact_id[surface]
            if (state >> clear_gid) & 1 and surface_at_place:
                return 0  # Already clear and at the right place
        
        # Not clear: need to lift something off + load into truck
        # m5: do_clear(c) + Lift + do_get_truck + Load = at least 2 (Lift + Load)
        # Plus recursive clearing
        return 2

    def _estimate_do_get_truck(self, state, args):
        """Estimate cost for do_get_truck(truck, place)."""
        if len(args) < 2:
            return 0
        
        truck, place = args[0], args[1]
        
        # Check if truck already at place
        at_gid = self.at_fact_id.get((truck, place))
        if at_gid is not None and (state >> at_gid) & 1:
            return 0  # Already there
        
        return 1  # Need to drive
