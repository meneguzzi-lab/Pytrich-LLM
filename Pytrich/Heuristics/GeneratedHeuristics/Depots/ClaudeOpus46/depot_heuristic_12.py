# Bottleneck: Stacking/unstacking crates with single-carry hoists.
# Each hoist can lift only one crate at a time, and clearing stacks
# to reach buried crates or surfaces dominates plan length.
#
# Independent admissible lower bounds:
# T1 - Unsatisfied goal count: Each unsatisfied on(c,s) goal requires at least
#      2 actions (lift+drop or unload+drop), so count * 2 is a lower bound.
# T2 - Displacement cost: If a crate must move between places, it needs at least
#      lift+load+drive+unload+drop = 5 actions. Crates already at the right place
#      but wrong position need at least lift+drop = 2 actions.
# T3 - Stack clearing cost: For each goal on(c,s), if c is not available (not clear,
#      not lifted, not in truck), we need to unstack crates above it. Similarly for s.
#      Each crate removed from a stack costs at least 2 actions (lift+load).

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic12(Heuristic):
    """Domain-dependent heuristic for the Depots domain.
    
    Uses max of multiple admissible lower bounds:
    1. Goal-based displacement estimation
    2. Task network structural cost
    3. Stack clearing cost estimation
    """

    def __init__(self, name="depot_anthropicopus46_heuristic_12"):
        super().__init__(name)
        # Fact lookups - populated in initialize
        self.on_fact_ids = {}      # (crate, surface) -> global_id
        self.at_fact_ids = {}      # (locatable, place) -> global_id
        self.in_fact_ids = {}      # (crate, truck) -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.lifting_fact_ids = {} # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)
        self.goal_facts_set = set()  # set of global_ids that are goal facts
        
        # Object categorization
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # Location tracking
        self.crate_at_facts = {}   # crate -> list of (place, global_id)
        self.surface_at_facts = {} # surface -> list of (place, global_id)
        
        # Stack structure: on_fact for (crate, surface)
        self.crate_on_facts = {}   # crate -> list of (surface, global_id)
        self.surface_topped_by = {} # surface -> list of (crate, global_id) - what's on top
        
        # Task type identification
        self.do_put_on_task_name = "do_put_on"
        self.do_clear_task_name = "do_clear"
        self.do_get_truck_task_name = "do_get_truck"
        self.do_lift_crate_task_name = "do_lift_crate"
        self.do_load_truck_task_name = "do_load_truck"
        self.do_unload_truck_task_name = "do_unload_truck"
        
        # Operator identification
        self.nop_op_names = set()
        self.drive_op_names = set()
        self.lift_op_names = set()
        self.drop_op_names = set()
        self.load_op_names = set()
        self.unload_op_names = set()

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
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Parse all facts, build lookup structures, identify goals."""
        self.model = model
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                self.on_fact_ids[(crate, surface)] = fact.global_id
                self.crates.add(crate)
                self.surfaces.add(surface)
                self.surfaces.add(crate)  # crates can be surfaces
                
                if crate not in self.crate_on_facts:
                    self.crate_on_facts[crate] = []
                self.crate_on_facts[crate].append((surface, fact.global_id))
                
                if surface not in self.surface_topped_by:
                    self.surface_topped_by[surface] = []
                self.surface_topped_by[surface].append((crate, fact.global_id))
                
            elif predicate == "at" and len(args) == 2:
                obj, place = args[0], args[1]
                self.at_fact_ids[(obj, place)] = fact.global_id
                self.places.add(place)
                
                if obj.startswith("crate"):
                    self.crates.add(obj)
                    if obj not in self.crate_at_facts:
                        self.crate_at_facts[obj] = []
                    self.crate_at_facts[obj].append((place, fact.global_id))
                elif obj.startswith("pallet"):
                    self.pallets.add(obj)
                    if obj not in self.surface_at_facts:
                        self.surface_at_facts[obj] = []
                    self.surface_at_facts[obj].append((place, fact.global_id))
                elif obj.startswith("truck"):
                    self.trucks.add(obj)
                elif obj.startswith("hoist"):
                    self.hoists.add(obj)
                    
            elif predicate == "in" and len(args) == 2:
                crate, truck = args[0], args[1]
                self.in_fact_ids[(crate, truck)] = fact.global_id
                self.crates.add(crate)
                self.trucks.add(truck)
                
            elif predicate == "clear" and len(args) == 1:
                surface = args[0]
                self.clear_fact_ids[surface] = fact.global_id
                self.surfaces.add(surface)
                
            elif predicate == "lifting" and len(args) == 2:
                hoist, crate = args[0], args[1]
                self.lifting_fact_ids[(hoist, crate)] = fact.global_id
                self.hoists.add(hoist)
                self.crates.add(crate)
                
            elif predicate == "available" and len(args) == 1:
                hoist = args[0]
                self.available_fact_ids[hoist] = fact.global_id
                self.hoists.add(hoist)

        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            predicate, args = self._parse_fact_name(gname)
            if predicate == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                gid = self.on_fact_ids.get((crate, surface))
                if gid is not None:
                    self.goal_on_pairs.append((crate, surface, gid))
                    self.goal_facts_set.add(gid)

        # Build mapping: surface -> fixed location (for pallets)
        self.surface_location = {}
        for surface in self.pallets:
            if surface in self.surface_at_facts:
                for place, gid in self.surface_at_facts[surface]:
                    # Pallets don't move, so check initial state
                    if (initial_node.state >> gid) & 1:
                        self.surface_location[surface] = place
        
        # Precompute hoist locations (hoists don't move in depots)
        self.hoist_location = {}
        for hoist in self.hoists:
            for place in self.places:
                key = (hoist, place)
                if key in self.at_fact_ids:
                    gid = self.at_fact_ids[key]
                    if (initial_node.state >> gid) & 1:
                        self.hoist_location[hoist] = place
        
        # Places that have hoists
        self.places_with_hoists = set(self.hoist_location.values())
        
        # Identify operator types
        for op in model.operators:
            task_type, args = self._parse_task_name(op.name)
            lower_type = task_type.lower() if task_type else ""
            if lower_type == "nop":
                self.nop_op_names.add(op.global_id)
            elif lower_type == "drive":
                self.drive_op_names.add(op.global_id)
            elif lower_type == "lift":
                self.lift_op_names.add(op.global_id)
            elif lower_type == "drop":
                self.drop_op_names.add(op.global_id)
            elif lower_type == "load":
                self.load_op_names.add(op.global_id)
            elif lower_type == "unload":
                self.unload_op_names.add(op.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_crate_location(self, crate, state):
        """Get current location of a crate, or None if in a truck."""
        if crate in self.crate_at_facts:
            for place, gid in self.crate_at_facts[crate]:
                if (state >> gid) & 1:
                    return place
        return None

    def _get_crate_support(self, crate, state):
        """Get what this crate is currently on, or None."""
        if crate in self.crate_on_facts:
            for surface, gid in self.crate_on_facts[crate]:
                if (state >> gid) & 1:
                    return surface
        return None

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        if surface in self.clear_fact_ids:
            return (state >> self.clear_fact_ids[surface]) & 1
        return True  # If no clear fact exists, assume clear

    def _is_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for truck in self.trucks:
            key = (crate, truck)
            if key in self.in_fact_ids:
                if (state >> self.in_fact_ids[key]) & 1:
                    return True
        return False

    def _is_being_lifted(self, crate, state):
        """Check if crate is being lifted by any hoist."""
        for hoist in self.hoists:
            key = (hoist, crate)
            if key in self.lifting_fact_ids:
                if (state >> self.lifting_fact_ids[key]) & 1:
                    return True
        return False

    def _count_crates_above(self, surface, state, visited=None):
        """Count how many crates are stacked above a surface."""
        if visited is None:
            visited = set()
        if surface in visited:
            return 0
        visited.add(surface)
        
        count = 0
        if surface in self.surface_topped_by:
            for crate, gid in self.surface_topped_by[surface]:
                if (state >> gid) & 1:
                    count += 1
                    # Recursively count what's above this crate
                    count += self._count_crates_above(crate, state, visited)
        return count

    def _get_surface_location(self, surface, state):
        """Get the fixed location of a surface (pallet or crate)."""
        if surface in self.surface_location:
            return self.surface_location[surface]
        # For crates acting as surfaces, find their current location
        return self._get_crate_location(surface, state)

    def _goal_displacement_cost(self, state):
        """Compute lower bound based on unsatisfied goals with displacement analysis.
        
        For each unsatisfied on(c, s) goal:
        - If c is already at same place as s: at least 2 actions (lift + drop)
        - If c is at different place: at least 5 actions (lift + load + drive + unload + drop)
        - If c is in a truck: at least 2 actions (unload + drop)
        - If c is being lifted: at least 1 action (drop)
        
        This is admissible because these are minimum action counts.
        """
        total_cost = 0
        for crate, surface, gid in self.goal_on_pairs:
            # Check if goal is already satisfied
            if (state >> gid) & 1:
                continue
            
            # Goal is unsatisfied - estimate minimum cost
            crate_loc = self._get_crate_location(crate, state)
            
            if self._is_being_lifted(crate, state):
                # Just need to drop it (if at right place) or more
                total_cost += 1
                continue
            
            if self._is_in_truck(crate, state):
                # Need at least: unload + drop = 2
                total_cost += 2
                continue
            
            if crate_loc is None:
                # Crate location unknown, conservative estimate
                total_cost += 2
                continue
            
            surface_loc = self._get_surface_location(surface, state)
            
            # Crate is on some surface at crate_loc
            # Need to check if crate is clear
            if not self._is_clear(crate, state):
                # Need to clear crate first: each crate above costs at least 2 (lift+load)
                crates_above = self._count_crates_above(crate, state)
                total_cost += crates_above * 2
            
            if surface_loc is not None and crate_loc == surface_loc:
                # Same place: lift + drop = 2
                total_cost += 2
            elif surface_loc is not None:
                # Different place: lift + load + drive + unload + drop = 5
                total_cost += 5
            else:
                # Surface location unknown (it's a crate that might move)
                total_cost += 2
        
        return total_cost

    def _target_clearing_cost(self, state):
        """Compute lower bound based on clearing target surfaces for goals.
        
        For each unsatisfied on(c, s) goal where s is not clear,
        we need to remove what's on s. Each removal costs at least 2 actions.
        This is admissible because clearing is unavoidable.
        """
        total_cost = 0
        surfaces_counted = set()
        
        for crate, surface, gid in self.goal_on_pairs:
            if (state >> gid) & 1:
                continue
            
            if surface in surfaces_counted:
                continue
            surfaces_counted.add(surface)
            
            if not self._is_clear(surface, state):
                # Check if what's on the surface is the goal crate
                # (in which case no clearing needed for the surface)
                current_top = None
                if surface in self.surface_topped_by:
                    for c, fid in self.surface_topped_by[surface]:
                        if (state >> fid) & 1:
                            current_top = c
                            break
                
                if current_top != crate:
                    # Need to clear surface: at least 2 per crate on top
                    crates_on_surface = self._count_crates_above(surface, state)
                    total_cost += crates_on_surface * 2
        
        return total_cost

    def _task_network_cost(self, task_network):
        """Estimate minimum cost from remaining task network.
        
        Each primitive action except nop costs 1.
        Each abstract task has a minimum decomposition cost.
        This is admissible because it lower-bounds actual actions needed.
        """
        cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task - estimate minimum cost
                task_type, args = self._parse_task_name(task.name)
                lower_type = task_type.lower() if task_type else ""
                
                if lower_type == "do_put_on":
                    # Minimum decomposition: m0 (already on) = 0 (nop)
                    # But if not already satisfied, minimum is:
                    # m1: clear + clear + lift + drop = at least 2
                    # We'll count 0 here as minimum since m0 might apply
                    # But we add context from goals
                    cost += 0  # Conservative - covered by goal analysis
                elif lower_type == "do_clear":
                    # m4: already clear = 0 (nop)
                    # m5: clear + lift + get_truck + load = at least 2
                    cost += 0
                elif lower_type == "do_get_truck":
                    # m6: already there = 0, m7: drive = 1
                    cost += 0
                elif lower_type == "do_lift_crate":
                    # m8: get_truck + unload = at least 1
                    # m9: lift = 1
                    cost += 1
                elif lower_type == "do_load_truck":
                    # get_truck + clear + lift + load = at least 2
                    cost += 2
                elif lower_type == "do_unload_truck":
                    # clear + unload + drop = at least 2
                    cost += 2
            else:
                # Primitive operator
                if task.global_id not in self.nop_op_names:
                    cost += 1
        
        return cost

    def _unsatisfied_goals_with_interference(self, state):
        """Count unsatisfied goals, penalizing goals that are currently true
        but will be disrupted by other goals (false-trivial detection).
        
        Admissible because each truly unsatisfied goal needs >= 2 actions.
        """
        unsatisfied = 0
        satisfied_goals = []
        unsatisfied_goals = []
        
        for crate, surface, gid in self.goal_on_pairs:
            if (state >> gid) & 1:
                satisfied_goals.append((crate, surface, gid))
            else:
                unsatisfied_goals.append((crate, surface, gid))
                unsatisfied += 1
        
        # Check for false-trivial: a satisfied goal on(c, s) where
        # s needs to be cleared for another goal (something else needs to go on s)
        # or c needs to be moved for another goal
        disrupted = 0
        
        # Crates that need to be moved (appear as crate in unsatisfied goals)
        crates_to_move = set()
        for crate, surface, gid in unsatisfied_goals:
            crates_to_move.add(crate)
        
        # Surfaces that need something placed on them
        surfaces_needing_placement = set()
        for crate, surface, gid in unsatisfied_goals:
            surfaces_needing_placement.add(surface)
        
        for crate, surface, gid in satisfied_goals:
            # If crate is currently on surface (goal satisfied) but
            # crate is also a surface that needs to be cleared for another goal
            if crate in surfaces_needing_placement:
                # Check if what needs to go on 'crate' is not already there
                # This means crate might need to be accessible, possibly disrupting
                pass  # Complex interaction, skip for now
            
            # If something needs to go on 'surface' but 'crate' is there AND
            # another crate also needs to go on surface - contradiction
            # This shouldn't happen in valid problems
            
            # If crate needs to move for another goal
            if crate in crates_to_move:
                disrupted += 1
        
        # Each truly unsatisfied goal needs at least 2 actions
        # Each disrupted goal needs at least 4 actions (remove + re-place)
        return unsatisfied * 2 + disrupted * 2

    def _compute(self, state, task_network):
        """Compute the heuristic value as max of multiple lower bounds."""
        # Term 1: Goal displacement cost (per-goal state-aware estimation)
        t1 = self._goal_displacement_cost(state)
        
        # Term 2: Target clearing cost
        t2 = self._target_clearing_cost(state)
        
        # Term 3: Task network structural cost
        t3 = self._task_network_cost(task_network)
        
        # Term 4: Unsatisfied goals with interference detection
        t4 = self._unsatisfied_goals_with_interference(state)
        
        return max(t1, t2, t3, t4)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Called for every node expansion."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
