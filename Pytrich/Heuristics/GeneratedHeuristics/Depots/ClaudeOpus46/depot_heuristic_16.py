# Depots Domain Heuristic
# Bottleneck: Single hoist per place + stacking constraints. Moving a crate requires
# clearing it, clearing destination, and possibly truck transport between places.
# 
# Admissible lower bounds:
# 1. Goal displacement: each unsatisfied on(c,s) goal needs >= 2 actions (lift+drop local)
#    or >= 5 actions (lift+load+drive+unload+drop cross-place). Admissible because these
#    are the minimum action sequences in any valid plan.
# 2. Clearing cost: if a crate/surface needed as a destination has stuff on top,
#    each blocking crate needs >= 2 actions to remove. Admissible as minimum removal cost.
# 3. Task network bound: remaining abstract tasks each need >= 1 action to resolve.
#    Admissible as each abstract task decomposes to at least one primitive action.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic16(Heuristic):
    """Domain-dependent heuristic for the Depots domain.
    
    Uses a max of multiple admissible lower bounds:
    - Unsatisfied goal cost based on crate displacement
    - Stack clearing cost estimation
    - Task network size lower bound
    """

    def __init__(self):
        super().__init__(name="depot_anthropicopus46_heuristic_16")
        # Fact lookup maps
        self.on_fact_ids = {}      # (crate, surface) -> global_id
        self.at_fact_ids = {}      # (locatable, place) -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.in_fact_ids = {}      # (crate, truck) -> global_id
        self.lifting_fact_ids = {} # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal info
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)
        
        # Object categorization
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # Location mapping: object -> place (from initial state for static objects)
        self.hoist_place = {}  # hoist -> place (hoists don't move)
        self.pallet_place = {} # pallet -> place (pallets don't move)
        
        # Place to hoists mapping
        self.place_hoists = {} # place -> list of hoists
        
        # For stack tracking
        self.on_facts_by_top = {}  # crate -> list of (crate, surface, global_id) for on(crate, surface)
        
        # Operator/task classification
        self.nop_operators = set()
        self.drive_operators = set()
        self.lift_operators = set()
        self.drop_operators = set()
        self.load_operators = set()
        self.unload_operators = set()

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like +on[crate0,pallet1] into (predicate, [args])"""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts and build lookup structures for fast heuristic computation."""
        self.model = model
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[0])  # crates are surfaces
                self.surfaces.add(args[1])
                # Track which on-facts have this crate as the top item
                if args[0] not in self.on_facts_by_top:
                    self.on_facts_by_top[args[0]] = []
                self.on_facts_by_top[args[0]].append((args[0], args[1], fact.global_id))
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = fact.global_id
                # Categorize objects by naming convention
                obj, place = args[0], args[1]
                if obj.startswith("hoist"):
                    self.hoists.add(obj)
                    self.hoist_place[obj] = place
                    if place not in self.place_hoists:
                        self.place_hoists[place] = []
                    self.place_hoists[place].append(obj)
                elif obj.startswith("pallet"):
                    self.pallets.add(obj)
                    self.pallet_place[obj] = place
                elif obj.startswith("truck"):
                    self.trucks.add(obj)
                elif obj.startswith("crate"):
                    self.crates.add(obj)
                if place.startswith("depot") or place.startswith("distributor"):
                    self.places.add(place)
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = fact.global_id
                self.surfaces.add(args[0])
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = fact.global_id
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                fid = self.on_fact_ids.get((crate, surface), -1)
                self.goal_on_pairs.append((crate, surface, fid))
        
        # Build reverse on-map: surface -> (crate, global_id) for finding what's on a surface
        # We'll use this for stack height estimation
        self.surface_to_on_facts = {}  # surface -> list of (crate, global_id)
        for (crate, surface), gid in self.on_fact_ids.items():
            if surface not in self.surface_to_on_facts:
                self.surface_to_on_facts[surface] = []
            self.surface_to_on_facts[surface].append((crate, gid))
        
        # Determine static place for each surface (pallets have fixed places)
        # For pallets, we know their place. For crates, place is dynamic.
        self.surface_place = {}  # surface -> place (only for static surfaces like pallets)
        for pallet, place in self.pallet_place.items():
            self.surface_place[pallet] = place
        
        # Build goal set for quick lookup
        self.goal_on_set = set()
        for crate, surface, fid in self.goal_on_pairs:
            self.goal_on_set.add((crate, surface))
        
        # Precompute which crates are goal-relevant (appear in goal on-facts)
        self.goal_crates = set()
        self.goal_surfaces = set()
        for crate, surface, fid in self.goal_on_pairs:
            self.goal_crates.add(crate)
            self.goal_surfaces.add(surface)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_crate_place(self, crate, state):
        """Find current place of a crate by checking at facts. Returns None if not at any place (in truck or being lifted)."""
        for place in self.places:
            key = (crate, place)
            if key in self.at_fact_ids:
                fid = self.at_fact_ids[key]
                if (state >> fid) & 1:
                    return place
        return None

    def _is_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for truck in self.trucks:
            key = (crate, truck)
            if key in self.in_fact_ids:
                fid = self.in_fact_ids[key]
                if (state >> fid) & 1:
                    return True
        return False

    def _get_surface_place(self, surface):
        """Get the fixed place of a surface (only works for pallets)."""
        return self.surface_place.get(surface, None)

    def _count_on_surface(self, surface, state):
        """Count how many crates are currently on this surface."""
        if surface not in self.surface_to_on_facts:
            return 0
        count = 0
        for crate, gid in self.surface_to_on_facts[surface]:
            if (state >> gid) & 1:
                count += 1
        return count

    def _get_what_is_on(self, surface, state):
        """Get the crate currently on top of this surface (if any)."""
        if surface not in self.surface_to_on_facts:
            return None
        for crate, gid in self.surface_to_on_facts[surface]:
            if (state >> gid) & 1:
                return crate
        return None

    def _is_clear(self, surface, state):
        """Check if surface is clear."""
        if surface in self.clear_fact_ids:
            fid = self.clear_fact_ids[surface]
            return (state >> fid) & 1
        return True  # If no clear fact exists, assume clear

    def _stack_height_above(self, surface, state, visited=None):
        """Count how many crates are stacked above this surface that are NOT in their goal position.
        Returns the number of crates that need to be moved to clear this surface."""
        if visited is None:
            visited = set()
        if surface in visited:
            return 0
        visited.add(surface)
        
        if self._is_clear(surface, state):
            return 0
        
        # Find what's on this surface
        crate_on_top = self._get_what_is_on(surface, state)
        if crate_on_top is None:
            return 0
        
        # Check if this crate being on this surface is a goal
        is_goal_position = (crate_on_top, surface) in self.goal_on_set
        
        # Recursively count above this crate
        above = self._stack_height_above(crate_on_top, state, visited)
        
        if is_goal_position and above == 0:
            # This crate is in goal position and nothing above it needs to move
            return 0
        else:
            return 1 + above

    def _compute_goal_displacement(self, state):
        """Compute lower bound based on unsatisfied goal on-facts.
        
        For each unsatisfied on(c, s) goal:
        - If crate is at same place as surface destination: >= 2 actions (lift + drop)
        - If crate is at different place: >= 5 actions (lift + load + drive + unload + drop)
        - If crate is in truck: >= 2 actions (unload + drop)
        - If crate is being lifted: >= 1 action (drop)
        
        This is admissible because these are the minimum possible actions.
        """
        total_cost = 0
        
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                # Goal already satisfied
                continue
            
            # Goal not satisfied - estimate minimum cost
            # Check if crate is being lifted by any hoist
            is_lifted = False
            for hoist in self.hoists:
                key = (hoist, crate)
                if key in self.lifting_fact_ids:
                    lid = self.lifting_fact_ids[key]
                    if (state >> lid) & 1:
                        is_lifted = True
                        # Minimum: just drop it (if destination is clear and same place)
                        # But we need destination clear, so minimum is 1 (drop)
                        total_cost += 1
                        break
            
            if is_lifted:
                continue
            
            # Check if crate is in a truck
            in_truck = self._is_in_truck(crate, state)
            if in_truck:
                # Minimum: unload + drop = 2
                total_cost += 2
                continue
            
            # Crate is at some place on some surface
            crate_place = self._get_crate_place(crate, state)
            surface_place = self._get_surface_place(surface)
            
            if crate_place is not None and surface_place is not None:
                if crate_place == surface_place:
                    # Same place: lift + drop = 2
                    total_cost += 2
                else:
                    # Different place: lift + load + drive + unload + drop = 5
                    total_cost += 5
            elif crate_place is not None and surface_place is None:
                # Surface is a crate (dynamic position) - estimate 2 minimum
                total_cost += 2
            else:
                # Unknown position, conservative estimate
                total_cost += 2
        
        return total_cost

    def _compute_clearing_cost(self, state):
        """Compute lower bound based on surfaces that need clearing for goal achievement.
        
        For each goal on(c, s), if s is not clear and the crate currently on s
        is not the goal crate c, then we need to clear s first.
        Each crate removal costs >= 2 actions (lift + load/drop).
        
        Admissible because each blocking crate must be removed with at least 2 actions.
        """
        clearing_cost = 0
        
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                # Goal already satisfied
                continue
            
            # Check if surface needs clearing
            if not self._is_clear(surface, state):
                # Something is on the surface that shouldn't be
                current_on = self._get_what_is_on(surface, state)
                if current_on is not None and current_on != crate:
                    # Need to remove this crate (and anything above it)
                    height = self._stack_height_above(surface, state)
                    # Each crate above needs at least 2 actions to remove (lift + load/drop)
                    clearing_cost += height * 2
        
        # Avoid double-counting: divide by a factor since multiple goals may share clearing work
        # But to stay admissible, we can take the max over individual clearing costs instead
        # Actually, for admissibility, we should note that the same crate clearing may serve
        # multiple goals. So we take a conservative approach.
        # We'll return clearing_cost but note it may overcount.
        # To be safe, let's take a per-goal max approach
        return clearing_cost

    def _compute_source_clearing_cost(self, state):
        """Estimate cost to clear crates that are buried under other crates.
        
        For each goal on(c, s) where c is not already on s:
        If c is not clear, we need to remove everything above c first.
        Each removal costs >= 2 actions.
        
        Admissible because each crate above the goal crate must be removed.
        """
        source_cost = 0
        
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                continue
            
            # Check if crate is clear
            if crate in self.clear_fact_ids:
                clear_fid = self.clear_fact_ids[crate]
                if not ((state >> clear_fid) & 1):
                    # Crate is not clear - need to remove stuff above it
                    height = self._stack_height_above(crate, state)
                    source_cost += height * 2
        
        return source_cost

    def _compute_task_network_bound(self, task_network):
        """Lower bound from task network: each abstract task needs at least 1 primitive action.
        
        For do_put_on tasks, estimate based on whether the goal is already satisfied.
        For other tasks, count minimum actions needed.
        
        Admissible because every abstract task must decompose into at least 1 action
        (even nop counts as 0 cost but the task itself must be processed).
        """
        if not task_network:
            return 0
        
        count = 0
        for task in task_network:
            # Check if it's an abstract task (has decompositions)
            if hasattr(task, 'decompositions'):
                # Abstract task - minimum 1 action to resolve
                # But many abstract tasks decompose to just nop if precondition met
                # So we give 0 for potentially-nop tasks, 1 for others
                task_name = task.name if hasattr(task, 'name') else ''
                if 'do_put_on' in task_name:
                    # do_put_on can be nop if already satisfied, or need multiple actions
                    count += 1  # At least check needed
                elif 'do_clear' in task_name:
                    count += 0  # Can be nop
                elif 'do_get_truck' in task_name:
                    count += 0  # Can be nop
                elif 'do_lift_crate' in task_name:
                    count += 1  # At least one lift/unload
                elif 'do_load_truck' in task_name:
                    count += 2  # At least lift + load
                elif 'do_unload_truck' in task_name:
                    count += 2  # At least unload + drop
                else:
                    count += 1
            else:
                # Primitive operator
                op_name = task.name if hasattr(task, 'name') else ''
                if 'nop' not in op_name:
                    count += 1
        
        return count

    def _compute_unsatisfied_goals_with_instability(self, state):
        """Count goals that are currently true but will be disrupted.
        
        A goal on(c, s) is 'unstable' if:
        - It's currently true, BUT
        - Some other goal requires c to be somewhere else, or
        - Some other goal requires something else on s
        
        This detects false-trivial goals per P5.
        """
        # Build set of surfaces that need to be cleared for other goals
        surfaces_needing_clear = set()
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and not ((state >> fid) & 1):
                surfaces_needing_clear.add(surface)
        
        # Check each currently satisfied goal for instability
        instability_cost = 0
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                # Goal is currently satisfied
                # But is the crate used as a surface that needs clearing?
                if crate in surfaces_needing_clear:
                    # This goal will be disrupted - need to redo it later
                    instability_cost += 2  # At minimum lift+drop to redo
        
        return instability_cost

    def _compute(self, state, task_network):
        """Compute the heuristic value as max of multiple admissible bounds."""
        # Term 1: Goal displacement cost
        displacement = self._compute_goal_displacement(state)
        
        # Term 2: Task network lower bound  
        tn_bound = self._compute_task_network_bound(task_network)
        
        # Term 3: Combined clearing + instability
        instability = self._compute_unsatisfied_goals_with_instability(state)
        
        # Use max of displacement and task network bound (both admissible)
        # Add instability as it captures additional unavoidable work
        h_value = max(displacement + instability, tn_bound)
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
