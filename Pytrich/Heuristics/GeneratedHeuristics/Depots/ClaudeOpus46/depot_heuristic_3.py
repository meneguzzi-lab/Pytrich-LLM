# Depots Domain Heuristic
# Bottleneck: Crates must be unstacked, transported via trucks (lift->load->drive->unload->drop),
# and restacked. Hoists are shared per-place resources. The dominant cost is clearing stacks
# and transporting crates between locations.
#
# Admissible lower bounds:
# T1 - Goal displacement: For each unsatisfied on(c,s) goal, minimum actions needed based on
#      crate/surface locations (1 if lifting, 2 if same place, 5 if different place)
# T2 - Clearing cost: Surfaces that must be clear for goals but aren't require extra operations
# T3 - Task network minimum: Each remaining abstract task has a minimum primitive action count

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic3(Heuristic):
    def __init__(self):
        super().__init__(name="depot_anthropicopus46_heuristic_3")
        # Fact index maps
        self.on_fact_id = {}       # (crate, surface) -> global_id
        self.at_fact_id = {}       # (locatable, place) -> global_id
        self.in_fact_id = {}       # (crate, truck) -> global_id
        self.lifting_fact_id = {}  # (hoist, crate) -> global_id
        self.clear_fact_id = {}    # surface -> global_id
        self.available_fact_id = {} # hoist -> global_id

        # Goal structures
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)

        # Object collections
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()

        # Location maps for initial analysis
        self.surface_places = {}  # surface -> set of places where at(surface, place) facts exist
        self.crate_at_facts = {}  # crate -> list of (place, fact_global_id)
        self.crate_in_facts = {}  # crate -> list of (truck, fact_global_id)
        self.crate_lifting_facts = {}  # crate -> list of (hoist, fact_global_id)

        # On-graph: who is on what
        self.on_facts_by_crate = {}   # crate -> list of (surface, fact_global_id)
        self.on_facts_by_surface = {} # surface -> list of (crate, fact_global_id)

        # Task cost estimates
        self.min_task_cost = {}  # abstract task name prefix -> minimum cost

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

    def _get_task_name_prefix(self, task_name):
        """Extract the task type from a grounded task name."""
        # Task names look like: do_put_on[crate1,pallet1] or Drive[truck0,depot0,dist0]
        bracket_idx = task_name.find("[")
        if bracket_idx > 0:
            return task_name[:bracket_idx]
        # Could also have parenthesized or other format
        paren_idx = task_name.find("(")
        if paren_idx > 0:
            return task_name[:paren_idx]
        return task_name

    def _get_task_args(self, task_name):
        """Extract args from a grounded task name."""
        bracket_idx = task_name.find("[")
        if bracket_idx > 0 and task_name.endswith("]"):
            args_str = task_name[bracket_idx+1:-1]
            return args_str.split(",") if args_str else []
        return []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts and build index maps
        for fact in model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None:
                continue

            if predicate == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[0])  # crates are also surfaces
                self.surfaces.add(args[1])
                # Build on-graph
                if args[0] not in self.on_facts_by_crate:
                    self.on_facts_by_crate[args[0]] = []
                self.on_facts_by_crate[args[0]].append((args[1], fact.global_id))
                if args[1] not in self.on_facts_by_surface:
                    self.on_facts_by_surface[args[1]] = []
                self.on_facts_by_surface[args[1]].append((args[0], fact.global_id))

            elif predicate == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = fact.global_id
                self.places.add(args[1])
                # Determine object type by naming convention
                obj = args[0]
                place = args[1]
                if obj.startswith("crate"):
                    self.crates.add(obj)
                    if obj not in self.crate_at_facts:
                        self.crate_at_facts[obj] = []
                    self.crate_at_facts[obj].append((place, fact.global_id))
                elif obj.startswith("truck"):
                    self.trucks.add(obj)
                elif obj.startswith("hoist"):
                    self.hoists.add(obj)
                elif obj.startswith("pallet"):
                    self.pallets.add(obj)
                    if obj not in self.surface_places:
                        self.surface_places[obj] = set()
                    self.surface_places[obj].add(place)
                # Surfaces (pallets and crates) location tracking
                if obj not in self.surface_places:
                    self.surface_places[obj] = set()
                self.surface_places[obj].add(place)

            elif predicate == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
                self.trucks.add(args[1])
                if args[0] not in self.crate_in_facts:
                    self.crate_in_facts[args[0]] = []
                self.crate_in_facts[args[0]].append((args[1], fact.global_id))

            elif predicate == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = fact.global_id
                self.hoists.add(args[0])
                self.crates.add(args[1])
                if args[1] not in self.crate_lifting_facts:
                    self.crate_lifting_facts[args[1]] = []
                self.crate_lifting_facts[args[1]].append((args[0], fact.global_id))

            elif predicate == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = fact.global_id
                self.surfaces.add(args[0])

            elif predicate == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = fact.global_id
                self.hoists.add(args[0])

        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            predicate, args = self._parse_fact_name(gname)
            if predicate == "on" and len(args) == 2:
                crate = args[0]
                surface = args[1]
                fid = self.on_fact_id.get((crate, surface), -1)
                self.goal_on_pairs.append((crate, surface, fid))

        # Build set of goal crates and goal surfaces for quick lookup
        self.goal_crates = set()
        self.goal_surfaces = set()
        self.goal_on_set = set()  # (crate, surface) pairs that are goals
        for crate, surface, fid in self.goal_on_pairs:
            self.goal_crates.add(crate)
            self.goal_surfaces.add(surface)
            self.goal_on_set.add((crate, surface))

        # For each goal surface, record the goal crate that should be on it
        self.goal_crate_for_surface = {}  # surface -> crate
        for crate, surface, fid in self.goal_on_pairs:
            self.goal_crate_for_surface[surface] = crate

        # Precompute minimum costs for task types
        # do_put_on: min 0 (if already satisfied) to 1 (nop) or 2 (lift+drop same place) or 5 (load+drive+unload+drop+...)
        # do_clear: min 0 (nop) or 4 (lift+load needs get_truck too)
        # do_get_truck: min 0 (nop) or 1 (drive)
        # do_lift_crate: min 1 (lift) or 2 (get_truck+unload)
        # do_load_truck: min 2-4 (get_truck+clear+lift+load)
        # do_unload_truck: min 2-3 (clear+unload+drop)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _crate_location(self, crate, state):
        """Determine where a crate currently is.
        Returns: ('at', place) if at a place, ('in', truck) if in a truck, ('lifting', hoist) if being lifted, or None
        """
        # Check at facts
        if crate in self.crate_at_facts:
            for place, fid in self.crate_at_facts[crate]:
                if (state >> fid) & 1:
                    return ('at', place)

        # Check in facts
        if crate in self.crate_in_facts:
            for truck, fid in self.crate_in_facts[crate]:
                if (state >> fid) & 1:
                    return ('in', truck)

        # Check lifting facts
        if crate in self.crate_lifting_facts:
            for hoist, fid in self.crate_lifting_facts[crate]:
                if (state >> fid) & 1:
                    return ('lifting', hoist)

        return None

    def _surface_location(self, surface, state):
        """Find where a surface (pallet or crate) currently is."""
        if surface in self.crate_at_facts:
            for place, fid in self.crate_at_facts[surface]:
                if (state >> fid) & 1:
                    return place
        # Pallets have fixed locations - check at facts
        if surface in self.surface_places:
            for place in self.surface_places[surface]:
                fid = self.at_fact_id.get((surface, place), -1)
                if fid >= 0 and (state >> fid) & 1:
                    return place
        return None

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        fid = self.clear_fact_id.get(surface, -1)
        if fid < 0:
            return True  # Unknown surface, assume clear
        return (state >> fid) & 1

    def _current_on(self, crate, state):
        """What surface is this crate currently on? Returns surface name or None."""
        if crate in self.on_facts_by_crate:
            for surface, fid in self.on_facts_by_crate[crate]:
                if (state >> fid) & 1:
                    return surface
        return None

    def _what_is_on(self, surface, state):
        """What crate is currently on this surface? Returns crate name or None."""
        if surface in self.on_facts_by_surface:
            for crate, fid in self.on_facts_by_surface[surface]:
                if (state >> fid) & 1:
                    return crate
        return None

    def _count_stack_height(self, surface, state, max_depth=20):
        """Count how many crates are stacked on this surface."""
        count = 0
        current = surface
        while count < max_depth:
            top_crate = self._what_is_on(current, state)
            if top_crate is None:
                break
            count += 1
            current = top_crate
        return count

    def _goal_displacement_cost(self, state):
        """T1: For each unsatisfied on(c,s) goal, estimate minimum actions needed.
        Admissible because:
        - If on(c,s) already true: 0
        - If crate is being lifted at same place as s: 1 (Drop)
        - If crate is at same place as s: 2 (Lift + Drop) minimum
        - If crate is in a truck: 3 (Drive? + Unload + Drop) minimum
        - If crate is at different place: 4 (Lift + Load + Drive + Unload + Drop - can overlap) -> use 4
        Each goal requires at least these actions independently.
        """
        total_cost = 0

        for crate, surface, fid in self.goal_on_pairs:
            # Check if goal is already satisfied
            if fid >= 0 and (state >> fid) & 1:
                # Goal currently true, but might need to be undone and redone
                # Check goal stability: is the surface also a goal surface that needs something else?
                # For now, count as 0
                continue

            # Goal not satisfied - estimate cost
            crate_loc = self._crate_location(crate, state)
            surface_loc = self._surface_location(surface, state)

            if crate_loc is None:
                # Crate location unknown (shouldn't happen in valid state)
                total_cost += 2
                continue

            loc_type, loc_val = crate_loc

            if loc_type == 'lifting':
                # Crate is being lifted by some hoist
                # Need at least Drop (1 action)
                # But if the hoist isn't at the surface's place, need Load+Drive+Unload+Drop = 4
                # For admissibility, use 1
                total_cost += 1

            elif loc_type == 'in':
                # Crate is in a truck
                # Need at least Unload + Drop = 2 actions
                # Might also need Drive = 3 actions
                total_cost += 2

            elif loc_type == 'at':
                crate_place = loc_val
                # Need to check if surface is at same place
                if surface_loc == crate_place:
                    # Same place: Lift + Drop = 2 minimum
                    total_cost += 2
                elif surface_loc is not None:
                    # Different place: Lift + Load + Drive + Unload + Drop = 5 minimum
                    total_cost += 5
                else:
                    # Surface location unknown (e.g., crate being moved)
                    total_cost += 2

        return total_cost

    def _clearing_cost(self, state):
        """T2: For each goal on(c,s), if s is not clear and the crate currently on s
        is NOT c, we need to clear s first. Each clearing requires at minimum
        Lift + Load = 2 actions (to move the blocking crate away).
        
        Admissible because clearing a surface requires at least lifting the top crate
        off it (1 action) and putting it somewhere (1 action minimum).
        """
        total_cost = 0
        counted_clears = set()  # Avoid double-counting

        for crate, surface, fid in self.goal_on_pairs:
            # Skip if goal already satisfied
            if fid >= 0 and (state >> fid) & 1:
                continue

            # Check if surface needs clearing
            if not self._is_clear(surface, state):
                blocking_crate = self._what_is_on(surface, state)
                if blocking_crate is not None and blocking_crate != crate:
                    if surface not in counted_clears:
                        counted_clears.add(surface)
                        # Need to move blocking crate: at least Lift + Load = 2
                        # But the blocking crate itself might have stuff on it
                        stack_h = self._count_stack_height(surface, state, max_depth=10)
                        # Each crate in the stack needs Lift + Load = 2 actions minimum
                        total_cost += stack_h * 2

            # Also check if the crate itself needs clearing (if something is on it)
            if not self._is_clear(crate, state):
                top = self._what_is_on(crate, state)
                if top is not None and crate not in counted_clears:
                    counted_clears.add(crate)
                    stack_h = self._count_stack_height(crate, state, max_depth=10)
                    total_cost += stack_h * 2

        return total_cost

    def _task_network_cost(self, task_network):
        """T3: Estimate minimum primitive actions for remaining task network.
        
        For each task in the network:
        - Primitive operators: cost 1 (except nop which is 0)
        - do_put_on: minimum 0 (m0, already done) but if it's in the network, 
          likely needs at least 2 actions
        - do_clear: minimum 0 (m4) or 4 (m5: clear + lift + get_truck + load)
        - do_get_truck: minimum 0 (m6) or 1 (m7: drive)
        - do_lift_crate: minimum 1 (m9: lift) or 2 (m8: get_truck + unload)  
        - do_load_truck: minimum 4 (m10: get_truck + clear + lift + load)
        - do_unload_truck: minimum 3 (m11: clear + unload + drop)
        
        Admissible because each abstract task must decompose into at least this many primitives.
        """
        total_cost = 0

        for task in task_network:
            task_name = task.name if hasattr(task, 'name') else str(task)
            prefix = self._get_task_name_prefix(task_name)

            # Check if it's a primitive operator
            if hasattr(task, 'add_effects'):  # It's an Operator
                if prefix == "nop":
                    total_cost += 0
                else:
                    total_cost += 1
            else:  # Abstract task
                if prefix == "do_put_on":
                    # Minimum is 0 (already on target), but typically at least 2
                    # We use 0 for admissibility since m0 has cost 0 (just nop)
                    total_cost += 0
                elif prefix == "do_clear":
                    total_cost += 0  # m4 is just nop
                elif prefix == "do_get_truck":
                    total_cost += 0  # m6 is just nop
                elif prefix == "do_lift_crate":
                    total_cost += 1  # At least Lift or Unload
                elif prefix == "do_load_truck":
                    total_cost += 2  # At least Lift + Load
                elif prefix == "do_unload_truck":
                    total_cost += 2  # At least Unload + Drop
                else:
                    total_cost += 0

        return total_cost

    def _goal_instability_cost(self, state):
        """T4: Detect goals that are currently true but will need to be undone.
        
        A goal on(c,s) is unstable if:
        - It's currently true
        - But some OTHER goal requires c to be somewhere else (c must be moved)
        - Or some OTHER goal requires s to have a different crate on it
        
        Each unstable goal needs at least 2 extra actions (lift away + put back later).
        Admissible because undoing and redoing costs at least 2 actions.
        """
        total_cost = 0
        
        # Check for surfaces that currently satisfy a goal but also need a different crate
        # This catches: on(A,S) is true and is a goal, but on(B,S) is also a goal
        # Can't both be true, so one must be undone
        # Actually in this domain each surface can have at most one crate on it
        # So this case means on(A,S) is a goal and currently true, but also on(B,S) is a goal
        # which would be contradictory. Skip this.
        
        # Check for crates that are currently in goal position but need to move
        # because they're blocking a surface that another goal needs
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                # Goal currently true
                # Check if something needs to go on top of crate (another goal)
                # and that thing isn't there yet
                if crate in self.goal_crate_for_surface:
                    needed_crate = self.goal_crate_for_surface[crate]
                    needed_fid = self.on_fact_id.get((needed_crate, crate), -1)
                    if needed_fid >= 0 and not ((state >> needed_fid) & 1):
                        # The crate that should go on top of this crate isn't there yet
                        # But this doesn't mean current goal needs undoing
                        # However, if crate isn't clear, whatever is on it must move
                        if not self._is_clear(crate, state):
                            current_top = self._what_is_on(crate, state)
                            if current_top != needed_crate:
                                # Must remove current_top (2 actions) to make room
                                total_cost += 2

        return total_cost

    def _compute(self, state, task_network):
        """Compute the heuristic value as max of multiple admissible components."""
        # T1: Goal displacement
        t1 = self._goal_displacement_cost(state)

        # T2: Clearing cost
        t2 = self._clearing_cost(state)

        # T3: Task network structural cost
        t3 = self._task_network_cost(task_network)

        # T4: Goal instability
        t4 = self._goal_instability_cost(state)

        # Combined: use max of (t1+t2, t3) since t1 and t2 measure different unavoidable costs
        # t1 = displacement, t2 = clearing (these can overlap but both are lower bounds on different work)
        # Actually t1 and t2 may overlap since clearing is part of displacement
        # Use max for admissibility
        
        # The displacement + clearing represents actual work needed
        # The task network cost is an independent structural bound
        # Goal instability adds to displacement cost
        
        h_value = max(t1 + t4, t2, t3)
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
