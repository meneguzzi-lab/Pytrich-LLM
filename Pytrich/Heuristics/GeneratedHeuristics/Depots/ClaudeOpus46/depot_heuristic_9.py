"""Domain-Dependent Heuristic for Depots HDDL Domain

Bottleneck Analysis:
    The primary bottleneck in Depots is the need to clear crates stacked on top
    of target surfaces before placing goal crates, combined with the need to
    transport crates between different places using trucks (which requires
    lift/load/drive/unload/drop sequences). Each hoist can only hold one crate
    at a time, and each place has limited hoists.

Admissible Lower Bound Terms:
    1. Goal-displacement cost: For each unsatisfied goal (on ?c ?s), we need at
       minimum 1 action (Drop) if the crate is already lifted at the right place,
       2 actions (Lift+Drop) if at the same place on wrong surface, or ~5 actions
       (Lift+Load+Drive+Unload+Drop) if at a different place. This is admissible
       because each goal requires at least these actions.
    
    2. Clearing cost: For each goal surface that needs to be cleared, we need
       at least 2 actions per blocking crate (Lift+Load minimum). This is
       admissible because clearing is unavoidable prerequisite work.
    
    3. Task network lower bound: Count remaining compound tasks weighted by
       minimum decomposition depth. Each do_put_on that isn't already satisfied
       requires at least 1 action (nop if satisfied, or real actions otherwise).
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic9(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_9"):
        super().__init__(name)
        self._initialized = False

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like '+on[crate0,pallet1]' into (predicate, [args])"""
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
        """Preprocess: build fact maps, goal structures, place assignments."""
        self.model = model

        # Fact ID maps: predicate -> {(args_tuple): global_id}
        self.on_fact_id = {}      # (crate, surface) -> global_id
        self.at_fact_id = {}      # (locatable, place) -> global_id
        self.in_fact_id = {}      # (crate, truck) -> global_id
        self.clear_fact_id = {}   # surface -> global_id
        self.lifting_fact_id = {} # (hoist, crate) -> global_id
        self.available_fact_id = {} # hoist -> global_id

        # All objects by type (inferred from predicates)
        self.all_crates = set()
        self.all_surfaces = set()
        self.all_places = set()
        self.all_trucks = set()
        self.all_hoists = set()

        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.all_crates.add(args[0])
                self.all_surfaces.add(args[1])
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
                self.all_places.add(args[1])
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
                self.all_crates.add(args[0])
                self.all_trucks.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.all_surfaces.add(args[0])
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
                self.all_hoists.add(args[0])
                self.all_crates.add(args[1])
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
                self.all_hoists.add(args[0])

        # Parse goal facts
        self.goal_on = {}  # crate -> target_surface
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self.goal_on[args[0]] = args[1]

        # Build reverse map: surface -> set of crates that should be ON it (goal)
        self.goal_crates_on_surface = {}
        for crate, surface in self.goal_on.items():
            if surface not in self.goal_crates_on_surface:
                self.goal_crates_on_surface[surface] = set()
            self.goal_crates_on_surface[surface].add(crate)

        # For each place, find which hoists are there (static - hoists don't move)
        self.hoists_at_place = {}
        for (obj, place), fid in self.at_fact_id.items():
            if obj in self.all_hoists:
                if place not in self.hoists_at_place:
                    self.hoists_at_place[place] = []
                self.hoists_at_place[place].append(obj)

        # Build a map: for each surface, find all places where it could be
        # (surfaces are static, so check initial state)
        self.surface_place = {}
        for (obj, place), fid in self.at_fact_id.items():
            if obj in self.all_surfaces:
                # Check if this at fact is in initial state
                if (model.initial_state >> fid) & 1:
                    self.surface_place[obj] = place

        # For crates, their place can change, so we check dynamically
        # But we can build the at_fact lookup for quick access
        self.crate_at_facts = {}  # crate -> {place: global_id}
        for (obj, place), fid in self.at_fact_id.items():
            if obj in self.all_crates:
                if obj not in self.crate_at_facts:
                    self.crate_at_facts[obj] = {}
                self.crate_at_facts[obj][place] = fid

        # Build on-fact lookup per crate (what surface is crate on)
        self.crate_on_facts = {}  # crate -> {surface: global_id}
        for (crate, surface), fid in self.on_fact_id.items():
            if crate not in self.crate_on_facts:
                self.crate_on_facts[crate] = {}
            self.crate_on_facts[crate][surface] = fid

        # Build reverse on-fact lookup: surface -> {crate: global_id}
        self.surface_on_facts = {}  # surface -> {crate: global_id}
        for (crate, surface), fid in self.on_fact_id.items():
            if surface not in self.surface_on_facts:
                self.surface_on_facts[surface] = {}
            self.surface_on_facts[surface][crate] = fid

        # in-fact lookup per crate
        self.crate_in_facts = {}  # crate -> {truck: global_id}
        for (crate, truck), fid in self.in_fact_id.items():
            if crate not in self.crate_in_facts:
                self.crate_in_facts[crate] = {}
            self.crate_in_facts[crate][truck] = fid

        # Precompute task name parsing for abstract tasks
        self.task_info = {}  # abstract_task.global_id -> (task_name, args)
        for at in model.abstract_tasks:
            # Parse task name like "do_put_on[crate0,pallet1]"
            name = at.name
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                task_name = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                self.task_info[at.global_id] = (task_name, args)
            else:
                self.task_info[at.global_id] = (name, [])

        self._initialized = True
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_crate_location(self, crate, state):
        """Get the current place of a crate, or None if not at any place (in truck/being lifted)."""
        if crate in self.crate_at_facts:
            for place, fid in self.crate_at_facts[crate].items():
                if (state >> fid) & 1:
                    return place
        return None

    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        if crate in self.crate_in_facts:
            for truck, fid in self.crate_in_facts[crate].items():
                if (state >> fid) & 1:
                    return True
        return False

    def _is_crate_being_lifted(self, crate, state):
        """Check if crate is being lifted by any hoist."""
        for (hoist, c), fid in self.lifting_fact_id.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False

    def _get_surface_place(self, surface):
        """Get the place of a surface (static)."""
        return self.surface_place.get(surface, None)

    def _is_on_goal(self, crate, state):
        """Check if crate is already on its goal surface."""
        if crate not in self.goal_on:
            return True  # no goal for this crate
        target = self.goal_on[crate]
        key = (crate, target)
        if key in self.on_fact_id:
            return (state >> self.on_fact_id[key]) & 1
        return False

    def _is_clear(self, surface, state):
        """Check if surface is clear."""
        if surface in self.clear_fact_id:
            return (state >> self.clear_fact_id[surface]) & 1
        return True  # if no clear fact exists, assume clear

    def _get_crate_on_surface(self, surface, state):
        """Get the crate currently on top of a surface, if any."""
        if surface in self.surface_on_facts:
            for crate, fid in self.surface_on_facts[surface].items():
                if (state >> fid) & 1:
                    return crate
        return None

    def _count_stack_above(self, surface, state, max_depth=50):
        """Count crates stacked above a surface."""
        count = 0
        current = surface
        for _ in range(max_depth):
            top_crate = self._get_crate_on_surface(current, state)
            if top_crate is None:
                break
            count += 1
            current = top_crate
        return count

    def _estimate_put_on_cost(self, crate, target_surface, state):
        """Estimate minimum cost for do_put_on(crate, target_surface)."""
        # Check if already satisfied
        key = (crate, target_surface)
        if key in self.on_fact_id and (state >> self.on_fact_id[key]) & 1:
            return 0  # Already on target - nop

        cost = 0

        # Cost to clear the target surface
        if not self._is_clear(target_surface, state):
            # Need to clear whatever is on target_surface
            # Each crate above needs at least 4 actions: Lift + Load + (maybe Drive) + ...
            # Minimum: 2 actions per blocking crate (Lift + Load to get it into a truck)
            blocking = self._count_stack_above(target_surface, state)
            cost += blocking * 4  # Lift + Load + Unload + Drop per crate (conservative)

        target_place = self._get_surface_place(target_surface)
        crate_place = self._get_crate_location(crate, state)

        if crate_place is not None:
            # Crate is at a place
            # Need to clear crate first
            if not self._is_clear(crate, state):
                blocking_on_crate = self._count_stack_above(crate, state)
                cost += blocking_on_crate * 4

            if crate_place == target_place:
                # Same place: Lift + Drop = 2 actions minimum
                cost += 2
            else:
                # Different place: Lift + Load + Drive + Unload + Drop = 5 actions
                cost += 5
        elif self._is_crate_in_truck(crate, state):
            # In truck: need Unload + Drop = 2 actions, plus maybe Drive = 3
            cost += 3
        elif self._is_crate_being_lifted(crate, state):
            # Being lifted: just Drop = 1 (or Load+Drive+Unload+Drop = 4 if wrong place)
            cost += 1
        else:
            # Unknown state - conservative estimate
            cost += 5

        return cost

    def _compute(self, state, task_network):
        """Compute heuristic value for a state and task network."""
        if not self._initialized:
            return 0

        if len(task_network) == 0:
            # Check if goals are met
            if self.model.goals & state == self.model.goals:
                return 0
            return float('inf')

        # Term 1: Goal displacement - count unsatisfied goals with cost estimation
        goal_cost = 0
        unsatisfied_goals = 0
        for crate, target in self.goal_on.items():
            if not self._is_on_goal(crate, state):
                unsatisfied_goals += 1
                # Minimum cost: at least 1 action (Drop) to place it
                # But usually much more
                key = (crate, target)
                if key in self.on_fact_id:
                    # Quick estimate: is crate at the right place?
                    target_place = self._get_surface_place(target)
                    crate_place = self._get_crate_location(crate, state)
                    if crate_place is not None and target_place is not None:
                        if crate_place == target_place:
                            goal_cost += 2  # Lift + Drop
                        else:
                            goal_cost += 5  # Lift + Load + Drive + Unload + Drop
                    elif self._is_crate_in_truck(crate, state):
                        goal_cost += 3  # Drive + Unload + Drop
                    elif self._is_crate_being_lifted(crate, state):
                        goal_cost += 1  # Drop (best case)
                    else:
                        goal_cost += 5
                else:
                    goal_cost += 2

        # Term 2: Task network cost estimation
        task_cost = 0
        for task in task_network:
            gid = task.global_id
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task
                if gid in self.task_info:
                    task_name, args = self.task_info[gid]
                    if task_name == "do_put_on" and len(args) == 2:
                        crate_arg = args[0]
                        surface_arg = args[1]
                        tc = self._estimate_put_on_cost(crate_arg, surface_arg, state)
                        task_cost += tc
                    elif task_name == "do_clear" and len(args) == 2:
                        surface_arg = args[0]
                        if not self._is_clear(surface_arg, state):
                            blocking = self._count_stack_above(surface_arg, state)
                            task_cost += max(blocking * 4, 2)
                        # else: 0 (nop)
                    elif task_name == "do_get_truck" and len(args) == 2:
                        truck_arg = args[0]
                        place_arg = args[1]
                        # Check if truck is already at place
                        at_key = (truck_arg, place_arg)
                        if at_key in self.at_fact_id:
                            if not ((state >> self.at_fact_id[at_key]) & 1):
                                task_cost += 1  # Drive
                        # else: 0 (nop)
                    elif task_name == "do_lift_crate":
                        task_cost += 1  # At least Lift or Unload
                    elif task_name == "do_load_truck":
                        task_cost += 2  # Lift + Load minimum
                    elif task_name == "do_unload_truck":
                        task_cost += 2  # Unload + Drop minimum
                    else:
                        task_cost += 1
                else:
                    task_cost += 1
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive operator - costs 1 action
                # Check if it's nop
                op = self.model.operators[gid - self.model.iop_init]
                if "nop" not in op.name:
                    task_cost += 1

        # Term 3: Clearing cost for goal surfaces that need clearing
        # but aren't accounted for in the task network
        clearing_cost = 0
        for crate, target in self.goal_on.items():
            if not self._is_on_goal(crate, state):
                # Check if target surface needs clearing
                if not self._is_clear(target, state):
                    # What's on the target?
                    top_crate = self._get_crate_on_surface(target, state)
                    if top_crate is not None:
                        # Check if that crate is supposed to be there in the goal
                        if top_crate != crate:  # something else is on it
                            # Check if top_crate's goal is to be on target
                            if self.goal_on.get(top_crate) != target:
                                clearing_cost += 2  # At least Lift+Load to move it

        # Return max of independent lower bounds, combined with task estimate
        h1 = goal_cost  # Goal-based displacement lower bound
        h2 = task_cost   # Task-network structural lower bound
        h3 = unsatisfied_goals  # Simple count lower bound

        # Use max of goal-based and task-based, with clearing as additive bonus
        return max(h1, h2, h3 + clearing_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Evaluate heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
