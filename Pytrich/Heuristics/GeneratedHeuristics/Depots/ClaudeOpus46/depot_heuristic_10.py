"""
Depots Domain Heuristic - DepotsAnthropicOpus46Heuristic10

Bottleneck Analysis:
  The primary bottleneck in the Depots domain is the need to clear surfaces (unstacking
  crates) before placing target crates, combined with the need to transport crates between
  places using trucks (which requires loading/unloading via hoists). Each place has limited
  hoists, and each hoist can only lift one crate at a time. Moving a crate that is not at
  the target location requires: clearing it, lifting it, loading onto truck, driving,
  unloading, and dropping - minimum 4-6 actions per displaced crate.

Admissible Lower Bounds:
  Term 1 - Unsatisfied goal count (weighted): For each goal fact (on ?c ?s) not currently
    true, we need at minimum 1 action (Drop). This is a trivial lower bound but admissible.
  
  Term 2 - Per-crate displacement cost: For each unsatisfied goal (on ?c ?s), estimate the
    minimum actions needed based on current crate state:
    - If crate is at the right place and on some surface: need to clear it + lift + drop = 3+ actions
    - If crate is in a truck: need to unload + drop = 2 actions minimum
    - If crate is at wrong place: need clear + lift + load + drive + unload + drop = 6 actions minimum
    - If crate needs clearing (something on top): add 2+ actions per blocking crate
    This sum divided by the number of hoists gives a lower bound (hoists work in parallel).

  Term 3 - Task network lower bound: Count remaining non-nop tasks in the task network,
    where each primitive action contributes 1 and each abstract task contributes its minimum
    decomposition depth cost.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic10(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_10"):
        super().__init__(name)
        # Fact lookup maps
        self.on_fact_id = {}      # (crate, surface) -> global_id
        self.at_fact_id = {}      # (locatable, place) -> global_id
        self.in_fact_id = {}      # (crate, truck) -> global_id
        self.clear_fact_id = {}   # surface -> global_id
        self.lifting_fact_id = {} # (hoist, crate) -> global_id
        self.available_fact_id = {} # hoist -> global_id
        
        # Goal info
        self.goal_on_facts = []   # list of ((crate, surface), global_id)
        
        # Domain object sets
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        
        # Crate location tracking: crate -> list of (place, fact_id)
        self.crate_at_facts = {}  # crate -> [(place, global_id), ...]
        
        # Surface location: surface -> place (static)
        self.surface_place = {}
        
        # on facts indexed by what's on top: surface -> [(crate, global_id)]
        self.on_top_of = {}  # surface -> list of (crate, fact_id) where on(crate, surface)
        
        # on facts indexed by crate: crate -> [(surface, global_id)]
        self.crate_on = {}   # crate -> list of (surface, fact_id)
        
        # Number of hoists per place
        self.hoists_at_place = {}
        self.total_hoists = 0
        
        # Operator info for task network estimation
        self.nop_op_ids = set()
        
        # Abstract task minimum costs
        self.abstract_task_min_cost = {}

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
        """Parse operator/task name to get the action name and arguments."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_name = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_name, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[0])  # crates are surfaces
                self.surfaces.add(args[1])
                # Index by what's underneath
                if args[1] not in self.on_top_of:
                    self.on_top_of[args[1]] = []
                self.on_top_of[args[1]].append((args[0], f.global_id))
                # Index by crate
                if args[0] not in self.crate_on:
                    self.crate_on[args[0]] = []
                self.crate_on[args[0]].append((args[1], f.global_id))
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
                if args[0] not in self.crate_at_facts:
                    self.crate_at_facts[args[0]] = []
                self.crate_at_facts[args[0]].append((args[1], f.global_id))
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
                self.trucks.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
                self.hoists.add(args[0])
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
                self.hoists.add(args[0])
        
        self.total_hoists = max(len(self.hoists), 1)
        
        # Build static surface->place mapping from initial state
        # For pallets and crates, find their initial location
        for (obj, place), fid in self.at_fact_id.items():
            if obj in self.surfaces and obj not in self.crates:
                # This is a pallet - check if it's at this place in initial state
                if (initial_node.state >> fid) & 1:
                    self.surface_place[obj] = place
        
        # Also map hoists to places from initial state (hoists don't move)
        for (obj, place), fid in self.at_fact_id.items():
            if obj in self.hoists:
                if (initial_node.state >> fid) & 1:
                    if place not in self.hoists_at_place:
                        self.hoists_at_place[place] = 0
                    self.hoists_at_place[place] += 1
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_id.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_facts.append((args[0], args[1], fid))
        
        # Identify nop operators
        for op in model.operators:
            tname, _ = self._parse_task_name(op.name)
            if tname == "nop":
                self.nop_op_ids.add(op.global_id)
        
        # Precompute minimum decomposition costs for abstract tasks
        # Use iterative computation to handle dependencies
        self._precompute_abstract_task_costs()
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _precompute_abstract_task_costs(self):
        """Compute minimum action cost for each abstract task type (by name prefix)."""
        # We'll compute a fixed-point minimum cost for each abstract task
        # Initialize all abstract tasks with a large cost
        for at in self.model.abstract_tasks:
            self.abstract_task_min_cost[at.global_id] = 1
        
        # Iterate until convergence
        changed = True
        iterations = 0
        max_iterations = 20  # Prevent infinite loops
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            for at in self.model.abstract_tasks:
                min_cost = 999999
                for decomp in at.decompositions:
                    cost = 0
                    for subtask in decomp.task_network:
                        if subtask.global_id in self.nop_op_ids:
                            continue
                        elif subtask.global_id >= self.model.iop_init and subtask.global_id <= self.model.iop_end:
                            # Primitive operator
                            cost += 1
                        elif subtask.global_id >= self.model.iabt_init and subtask.global_id <= self.model.iabt_end:
                            # Abstract task
                            cost += self.abstract_task_min_cost.get(subtask.global_id, 1)
                    if cost < min_cost:
                        min_cost = cost
                if min_cost < self.abstract_task_min_cost[at.global_id]:
                    self.abstract_task_min_cost[at.global_id] = min_cost
                    changed = True
                elif min_cost > self.abstract_task_min_cost[at.global_id]:
                    self.abstract_task_min_cost[at.global_id] = min_cost
                    changed = True

    def _get_crate_location(self, crate, state):
        """Get current location of a crate, or None if not at any place (in truck or being lifted)."""
        if crate in self.crate_at_facts:
            for place, fid in self.crate_at_facts[crate]:
                if (state >> fid) & 1:
                    return place
        return None

    def _get_target_place(self, surface):
        """Get the place where a surface (pallet) is located. For crates, returns None (dynamic)."""
        return self.surface_place.get(surface, None)

    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for (c, t), fid in self.in_fact_id.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        fid = self.clear_fact_id.get(surface)
        if fid is not None:
            return (state >> fid) & 1
        return True  # If no clear fact, assume clear

    def _is_being_lifted(self, crate, state):
        """Check if crate is being lifted by any hoist."""
        for (h, c), fid in self.lifting_fact_id.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False

    def _count_blocks_on_top(self, surface, state, depth=0):
        """Count how many crates are stacked on top of this surface."""
        if depth > 50:  # Safety limit
            return 0
        if self._is_clear(surface, state):
            return 0
        # Find what's on top
        if surface in self.on_top_of:
            for crate, fid in self.on_top_of[surface]:
                if (state >> fid) & 1:
                    return 1 + self._count_blocks_on_top(crate, state, depth + 1)
        return 0

    def _compute_goal_displacement_cost(self, state):
        """Compute displacement-based lower bound for unsatisfied goals."""
        total_cost = 0
        
        for crate, surface, fid in self.goal_on_facts:
            # Check if this goal is already satisfied
            if (state >> fid) & 1:
                continue
            
            # Goal not satisfied - estimate minimum cost to achieve it
            cost = 0
            
            crate_loc = self._get_crate_location(crate, state)
            
            # Determine where the target surface is
            target_place = self._get_target_place(surface)
            if target_place is None:
                # Target is a crate - find its current location
                target_place = self._get_crate_location(surface, state)
            
            if self._is_being_lifted(crate, state):
                # Crate is being lifted - need at minimum Drop (1 action)
                # But may need to drive truck + load + unload if at wrong place
                if target_place is not None and crate_loc is None:
                    cost = 1  # Just drop
                else:
                    cost = 1
            elif self._is_crate_in_truck(crate, state):
                # Crate in truck - minimum: unload + drop = 2
                # If truck not at target place, add drive = 3
                cost = 2
            elif crate_loc is not None:
                # Crate is at some place
                # Need to: clear crate (if not clear) + lift + place
                if not self._is_clear(crate, state):
                    # Need to clear everything on top
                    blocks_on_top = self._count_blocks_on_top(crate, state)
                    # Each block needs: lift + load = 2 actions minimum
                    cost += blocks_on_top * 2
                
                if crate_loc == target_place:
                    # Same place - just need lift + drop = 2
                    cost += 2
                else:
                    # Different place - need lift + load + drive + unload + drop = 5
                    cost += 5
            else:
                # Crate location unknown (shouldn't happen normally)
                cost = 2
            
            # Target surface must also be clear
            if not self._is_clear(surface, state):
                # Check if the thing on top is the crate we want to place
                # (which would mean we don't need extra clearing)
                need_clear = True
                if surface in self.on_top_of:
                    for c, on_fid in self.on_top_of[surface]:
                        if (state >> on_fid) & 1 and c == crate:
                            need_clear = False
                            break
                if need_clear:
                    blocks_on_surface = self._count_blocks_on_top(surface, state)
                    cost += blocks_on_surface * 2
            
            total_cost += cost
        
        return total_cost

    def _compute_unsatisfied_goals(self, state):
        """Count unsatisfied goal facts."""
        count = 0
        for crate, surface, fid in self.goal_on_facts:
            if not ((state >> fid) & 1):
                count += 1
        return count

    def _compute_task_network_cost(self, task_network):
        """Estimate minimum cost from remaining task network."""
        total = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.nop_op_ids:
                continue
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # Primitive operator
                total += 1
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                # Abstract task - use precomputed minimum cost
                total += self.abstract_task_min_cost.get(gid, 1)
        return total

    def _compute_goal_instability(self, state):
        """Detect goals that are currently true but will likely need to be undone.
        
        A goal on(c, s) is 'unstable' if s has a goal on(c2, s) where c2 != c,
        meaning the current stacking must change.
        """
        # Build a map of what each surface needs on top (goal state)
        goal_on_surface = {}  # surface -> crate (goal)
        for crate, surface, fid in self.goal_on_facts:
            goal_on_surface[surface] = crate
        
        instability_cost = 0
        for crate, surface, fid in self.goal_on_facts:
            if (state >> fid) & 1:
                # Goal is currently true
                # Check if crate itself needs something specific on top
                if crate in goal_on_surface:
                    needed_on_top = goal_on_surface[crate]
                    # Check if the needed crate is currently on top
                    on_fid = self.on_fact_id.get((needed_on_top, crate))
                    if on_fid is not None and not ((state >> on_fid) & 1):
                        # Something else might be on top, or nothing
                        # If not clear and wrong thing on top, this goal may need undoing
                        if not self._is_clear(crate, state):
                            # Check what's actually on top
                            if crate in self.on_top_of:
                                for c_on, c_fid in self.on_top_of[crate]:
                                    if (state >> c_fid) & 1 and c_on != needed_on_top:
                                        # Wrong crate on top - will need to move it
                                        # which means this goal will be temporarily undone
                                        instability_cost += 2  # undo + redo = at least 2 extra actions
                                        break
        
        return instability_cost

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        # Term 1: Goal displacement cost (detailed per-crate cost estimation)
        displacement_cost = self._compute_goal_displacement_cost(state)
        
        # Term 2: Task network lower bound
        tn_cost = self._compute_task_network_cost(task_network)
        
        # Term 3: Simple unsatisfied goals (very fast)
        unsatisfied = self._compute_unsatisfied_goals(state)
        
        # Term 4: Goal instability
        instability = self._compute_goal_instability(state)
        
        # The displacement cost divided by number of hoists gives a parallelism-aware bound
        # But since actions are sequential in our plan, we use the raw sum
        
        # Return max of independent lower bounds
        h = max(displacement_cost, tn_cost, unsatisfied + instability)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
