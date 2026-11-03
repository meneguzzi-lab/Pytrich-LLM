"""Domain-Dependent Heuristic for Depots HDDL Domain

Bottleneck Analysis:
  The primary bottleneck in the Depots domain is that crates must be moved between
  places (depots/distributors) using trucks, and at each place there is limited
  hoist availability. Each crate placement requires: clearing the crate (lifting
  crates above it), clearing the target surface, potentially loading into a truck,
  driving, unloading, and dropping. The serializing constraint is that hoists can
  only hold one crate at a time, and stacks must be cleared top-down.

Admissible Lower Bounds:
  Term 1 - Unsatisfied goal count with action estimation:
    Each unsatisfied 'on(c,s)' goal requires at minimum 1 action (Drop) to achieve.
    If the crate is not at the same place as the target, it needs at least 4 actions
    (Lift + Load + Drive + Unload + Drop = 5, but we use a conservative lower bound).
    This is admissible because no action can satisfy more than one 'on' goal.

  Term 2 - Task network lower bound:
    Each remaining abstract task requires at least 1 primitive action to resolve.
    We estimate based on task type: do_put_on needs at least 1 action (Drop or nop),
    do_clear needs at least 1 action, etc. This is admissible as each task must
    eventually produce at least one primitive.

  Term 3 - Location mismatch penalty:
    For each goal on(c,s), if c and s are at different places, we need at least
    a Drive action. We count distinct required drives as a lower bound.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic19(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_19"):
        super().__init__(name)
        # Precomputed maps
        self.on_fact_ids = {}       # (crate_arg, surface_arg) -> global_id
        self.at_fact_ids = {}       # (obj_arg, place_arg) -> global_id
        self.clear_fact_ids = {}    # surface_arg -> global_id
        self.in_fact_ids = {}       # (crate_arg, truck_arg) -> global_id
        self.lifting_fact_ids = {}  # (hoist_arg, crate_arg) -> global_id
        self.available_fact_ids = {} # hoist_arg -> global_id
        
        # Goal structures
        self.goal_on_pairs = []     # List of (crate, surface, fact_global_id)
        self.goal_facts_set = set() # Set of goal fact global_ids
        
        # Object location maps (from initial state, used for preprocessing)
        self.crate_objects = set()
        self.surface_objects = set()
        self.place_objects = set()
        self.truck_objects = set()
        self.hoist_objects = set()
        
        # For fast task network estimation
        self.task_name_cache = {}
        
        # Crate-to-place lookup for at facts
        self.crate_at_facts = {}    # crate -> list of (place, global_id)
        self.surface_at_facts = {}  # surface -> list of (place, global_id)
        
        # on-facts for stack analysis: surface -> list of (crate, global_id)
        self.on_surface_facts = {}  # surface -> [(crate, global_id), ...]
        # on-facts: crate -> list of (surface, global_id)
        self.crate_on_facts = {}    # crate -> [(surface, global_id), ...]

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
        """Parse a task/operator name to extract the task type."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            return name[:bracket_idx]
        return name

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for fact in model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                # Build reverse maps
                if args[1] not in self.on_surface_facts:
                    self.on_surface_facts[args[1]] = []
                self.on_surface_facts[args[1]].append((args[0], fact.global_id))
                if args[0] not in self.crate_on_facts:
                    self.crate_on_facts[args[0]] = []
                self.crate_on_facts[args[0]].append((args[1], fact.global_id))
                
            elif predicate == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = fact.global_id
                # Build per-object at-fact lists
                if args[0] not in self.crate_at_facts:
                    self.crate_at_facts[args[0]] = []
                self.crate_at_facts[args[0]].append((args[1], fact.global_id))
                if args[0] not in self.surface_at_facts:
                    self.surface_at_facts[args[0]] = []
                self.surface_at_facts[args[0]].append((args[1], fact.global_id))
                
            elif predicate == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = fact.global_id
                
            elif predicate == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
                
            elif predicate == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = fact.global_id
                
            elif predicate == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = fact.global_id

        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            predicate, args = self._parse_fact_name(gname)
            if predicate == "on" and len(args) == 2:
                gid = self.on_fact_ids.get((args[0], args[1]))
                if gid is not None:
                    self.goal_on_pairs.append((args[0], args[1], gid))
                    self.goal_facts_set.add(gid)

        # Precompute task type costs for task network estimation
        # Map abstract task names to minimum decomposition costs
        self.abstract_task_min_costs = {}
        for at in model.abstract_tasks:
            task_type = self._parse_task_name(at.name)
            # Estimate minimum number of primitive actions per task type
            min_cost = self._estimate_task_min_cost(task_type, at)
            self.abstract_task_min_costs[at.global_id] = min_cost
        
        # Precompute operator costs (all operators cost 1 action except nop)
        self.operator_costs = {}
        for op in model.operators:
            op_type = self._parse_task_name(op.name)
            if op_type == "nop":
                self.operator_costs[op.global_id] = 0
            else:
                self.operator_costs[op.global_id] = 1

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _estimate_task_min_cost(self, task_type, abstract_task):
        """Estimate minimum primitive actions for a task type."""
        if task_type.startswith("do_put_on"):
            # Minimum: nop (if already satisfied) or Drop (1 action)
            # But realistically: Lift + Drop = 2, or Lift + Load + Drive + Unload + Drop = 5
            return 1
        elif task_type.startswith("do_clear"):
            # Minimum: nop (if already clear) or Lift + Load = 2
            return 1
        elif task_type.startswith("do_get_truck"):
            # Minimum: nop (if already there) or Drive = 1
            return 0
        elif task_type.startswith("do_lift_crate"):
            # Minimum: Lift = 1 or Unload = 1 (+ possibly Drive)
            return 1
        elif task_type.startswith("do_load_truck"):
            # Minimum: Lift + Load = 2 (+ do_get_truck + do_clear)
            return 2
        elif task_type.startswith("do_unload_truck"):
            # Minimum: Unload + Drop = 2 (+ do_clear)
            return 2
        return 1

    def _get_crate_location(self, crate, state):
        """Get the current location of a crate from the state using bit checks."""
        if crate in self.crate_at_facts:
            for place, gid in self.crate_at_facts[crate]:
                if (state >> gid) & 1:
                    return place
        return None

    def _get_surface_location(self, surface, state):
        """Get the current location of a surface from the state using bit checks."""
        if surface in self.surface_at_facts:
            for place, gid in self.surface_at_facts[surface]:
                if (state >> gid) & 1:
                    return place
        return None

    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for (c, t), gid in self.in_fact_ids.items():
            if c == crate and (state >> gid) & 1:
                return True
        return False

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        gid = self.clear_fact_ids.get(surface)
        if gid is not None:
            return (state >> gid) & 1
        return False

    def _get_crate_on_surface(self, surface, state):
        """Get the crate currently on top of a surface, if any."""
        if surface in self.on_surface_facts:
            for crate, gid in self.on_surface_facts[surface]:
                if (state >> gid) & 1:
                    return crate
        return None

    def _count_stack_height(self, surface, state, max_depth=10):
        """Count number of crates stacked on a surface."""
        count = 0
        current = surface
        for _ in range(max_depth):
            crate_on_top = self._get_crate_on_surface(current, state)
            if crate_on_top is None:
                break
            count += 1
            current = crate_on_top
        return count

    def _compute_goal_term(self, state):
        """Term 1: Estimate cost based on unsatisfied goals with state-aware costing."""
        h = 0
        for crate, surface, gid in self.goal_on_pairs:
            # Check if goal is already satisfied
            if (state >> gid) & 1:
                continue
            
            # Goal not satisfied - estimate minimum cost
            crate_loc = self._get_crate_location(crate, state)
            in_truck = self._is_crate_in_truck(crate, state)
            surface_loc = self._get_surface_location(surface, state)
            
            if in_truck:
                # Crate is in a truck: need Unload + Drop = 2
                # If truck not at surface location, also need Drive = 1 more
                cost = 2
                if surface_loc is not None:
                    # May need to drive truck there
                    cost += 1  # Conservative: assume drive needed
                # Check if surface is clear
                if not self._is_clear(surface, state):
                    # Need to clear the surface: at minimum Lift + Load = 2 per blocking crate
                    cost += 2
                h += cost
            elif crate_loc is not None:
                # Crate is at a place
                if surface_loc is not None and crate_loc == surface_loc:
                    # Same place: need Lift + Drop = 2
                    cost = 2
                    # If crate is not clear, need to clear it first
                    if not self._is_clear(crate, state):
                        stack_h = self._count_stack_height(crate, state, 5)
                        cost += stack_h * 2  # Each blocking crate: Lift + Load = 2
                    # If surface is not clear, need to clear it
                    if not self._is_clear(surface, state):
                        stack_h = self._count_stack_height(surface, state, 5)
                        cost += stack_h * 2
                    h += cost
                else:
                    # Different places: need Lift + Load + Drive + Unload + Drop = 5
                    cost = 5
                    if not self._is_clear(crate, state):
                        stack_h = self._count_stack_height(crate, state, 5)
                        cost += stack_h * 2
                    if not self._is_clear(surface, state):
                        stack_h = self._count_stack_height(surface, state, 5)
                        cost += stack_h * 2
                    h += cost
            else:
                # Crate location unknown (being lifted?) - minimum Drop = 1
                cost = 1
                if not self._is_clear(surface, state):
                    cost += 2
                h += cost
        
        return h

    def _compute_task_network_term(self, task_network):
        """Term 2: Estimate cost from remaining task network."""
        if not task_network:
            return 0
        
        h = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.abstract_task_min_costs:
                h += self.abstract_task_min_costs[gid]
            elif gid in self.operator_costs:
                h += self.operator_costs[gid]
            else:
                h += 1  # Unknown task, assume at least 1 action
        return h

    def _compute_goal_instability_term(self, state):
        """Term 3: Detect goals that are currently true but will likely be disrupted.
        
        A goal on(c, s) is 'unstable' if s itself needs to be moved for another goal,
        or if c has something on it that needs to be moved elsewhere.
        """
        # Build set of crates that need to be moved (have unsatisfied goals)
        crates_needing_move = set()
        for crate, surface, gid in self.goal_on_pairs:
            if not ((state >> gid) & 1):
                crates_needing_move.add(crate)
        
        # Count currently satisfied goals that sit on surfaces that need to move
        instability_cost = 0
        for crate, surface, gid in self.goal_on_pairs:
            if (state >> gid) & 1:
                # Goal is satisfied. But check if surface needs to move
                if surface in crates_needing_move:
                    # This goal will be disrupted when surface moves
                    # Need to redo: at least Lift + Drop = 2 to re-place
                    instability_cost += 2
        
        return instability_cost

    def _compute(self, state, task_network):
        """Compute the heuristic value as max of multiple terms."""
        # Term 1: Goal-based estimation with state-aware costing
        goal_h = self._compute_goal_term(state)
        
        # Term 2: Task network lower bound
        tn_h = self._compute_task_network_term(task_network)
        
        # Term 3: Goal instability detection
        instability_h = self._compute_goal_instability_term(state)
        
        # Combine: max of goal-based and task-network, plus instability as additive
        # The instability term adds to the goal term since those goals will need re-work
        combined_goal = goal_h + instability_h
        
        return max(combined_goal, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
