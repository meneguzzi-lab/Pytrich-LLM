"""Depots Domain Heuristic for Pytrich Planner

Bottleneck: Crates must be transported between places via trucks using
fixed-location hoists. Moving a crate to a different place requires at
minimum 5 actions (Lift, Load, Drive, Unload, Drop). Even same-place
rearrangement needs 2 actions (Lift, Drop) plus clearing work.
Stacking dependencies create cascading clear operations.

Admissible Lower Bounds:
1. Goal-distance term: For each unsatisfied goal fact (on c s), estimate
   minimum actions needed. If crate is at wrong place: >= 5 actions.
   If at right place but wrong surface: >= 2 actions. Admissible because
   each goal requires at least these actions independently.

2. Task-network term: Count remaining abstract tasks weighted by minimum
   decomposition depth. Each do_put_on that isn't already satisfied needs
   at least 1 action (nop excluded since precondition won't match for
   unsatisfied goals). Admissible as lower bound on remaining actions.

3. Clear-chain term: For goals requiring a surface to be clear, count
   the minimum clearing work needed. Each crate blocking a target surface
   needs at least 4 actions to move away (lift, load, drive away or
   lift, drop elsewhere). Admissible as each clear is independent work.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic0(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_0"):
        super().__init__(name)
        self._on_fact_ids = {}       # (crate, surface) -> global_id
        self._at_fact_ids = {}       # (locatable, place) -> global_id
        self._in_fact_ids = {}       # (crate, truck) -> global_id
        self._clear_fact_ids = {}    # surface -> global_id
        self._lifting_fact_ids = {}  # (hoist, crate) -> global_id
        self._available_fact_ids = {} # hoist -> global_id
        
        self._goal_on_pairs = []     # list of (crate, surface, fact_global_id)
        self._crate_locations = {}   # crate -> list of (place, fact_global_id)
        self._surface_locations = {} # surface -> list of (place, fact_global_id)
        self._crate_in_truck = {}    # crate -> list of (truck, fact_global_id)
        self._surface_at_place = {}  # surface -> place (static for pallets)
        
        # For task network analysis
        self._do_put_on_task_ids = set()  # global_ids of do_put_on abstract tasks
        self._nop_op_ids = set()     # global_ids of nop operators
        
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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == "on" and len(args) == 2:
                self._on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self._at_fact_ids[(args[0], args[1])] = f.global_id
                # Build crate/surface location lookups
                obj, place = args[0], args[1]
                if obj not in self._crate_locations:
                    self._crate_locations[obj] = []
                self._crate_locations[obj].append((place, f.global_id))
                if obj not in self._surface_locations:
                    self._surface_locations[obj] = []
                self._surface_locations[obj].append((place, f.global_id))
            elif pred == "in" and len(args) == 2:
                self._in_fact_ids[(args[0], args[1])] = f.global_id
                crate = args[0]
                if crate not in self._crate_in_truck:
                    self._crate_in_truck[crate] = []
                self._crate_in_truck[crate].append((args[1], f.global_id))
            elif pred == "clear" and len(args) == 1:
                self._clear_fact_ids[args[0]] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self._lifting_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self._available_fact_ids[args[0]] = f.global_id
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                fid = self._on_fact_ids.get((crate, surface), -1)
                if fid >= 0:
                    self._goal_on_pairs.append((crate, surface, fid))
        
        # Identify do_put_on abstract tasks and nop operators
        for at in model.abstract_tasks:
            if "do_put_on" in at.name:
                self._do_put_on_task_ids.add(at.global_id)
        for op in model.operators:
            if "nop" in op.name:
                self._nop_op_ids.add(op.global_id)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_crate_place(self, crate, state):
        """Return the place where a crate currently is, or None if not at any place (in truck/being lifted)."""
        if crate in self._crate_locations:
            for place, fid in self._crate_locations[crate]:
                if (state >> fid) & 1:
                    return place
        return None

    def _get_surface_place(self, surface, state):
        """Return the place where a surface currently is."""
        if surface in self._surface_locations:
            for place, fid in self._surface_locations[surface]:
                if (state >> fid) & 1:
                    return place
        return None

    def _is_crate_in_some_truck(self, crate, state):
        """Check if crate is in any truck."""
        if crate in self._crate_in_truck:
            for truck, fid in self._crate_in_truck[crate]:
                if (state >> fid) & 1:
                    return True
        return False

    def _is_crate_being_lifted(self, crate, state):
        """Check if crate is being lifted by any hoist."""
        for (hoist, c), fid in self._lifting_fact_ids.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False

    def _is_clear(self, surface, state):
        """Check if surface is clear."""
        fid = self._clear_fact_ids.get(surface, -1)
        if fid >= 0:
            return (state >> fid) & 1
        return False

    def _count_crates_on_top(self, surface, state):
        """Count how many crates are stacked on top of this surface (recursively).
        Returns the height of the stack above this surface."""
        count = 0
        current = surface
        while True:
            found_on_top = None
            for (crate, surf), fid in self._on_fact_ids.items():
                if surf == current and (state >> fid) & 1:
                    found_on_top = crate
                    break
            if found_on_top is None:
                break
            count += 1
            current = found_on_top
        return count

    def _compute(self, state, task_network):
        # Term 1: Goal-based distance estimation
        goal_cost = self._goal_distance_term(state)
        
        # Term 2: Task network based estimation
        task_cost = self._task_network_term(state, task_network)
        
        return max(goal_cost, task_cost)

    def _goal_distance_term(self, state):
        """Estimate minimum actions needed to achieve all unsatisfied goal facts.
        
        For each goal (on crate surface):
        - If already true: 0
        - If crate is at same place as surface: need at least 2 (lift + drop)
          plus clearing work
        - If crate is at different place: need at least 5 (lift + load + drive + unload + drop)
          plus clearing work  
        - If crate is in a truck: need at least 2 (unload + drop) plus possible drive
        - If crate is being lifted: need at least 1 (drop) plus possible transport
        """
        total_cost = 0
        
        for crate, surface, fid in self._goal_on_pairs:
            # Check if goal already satisfied
            if (state >> fid) & 1:
                continue
            
            # Base cost for placing crate on surface
            crate_place = self._get_crate_place(crate, state)
            surface_place = self._get_surface_place(surface, state)
            
            if self._is_crate_being_lifted(crate, state):
                # Crate is being lifted - at minimum need drop (1)
                # But may need load+drive+unload+drop if at wrong place
                if crate_place is None and surface_place is not None:
                    # Being lifted, we don't know where exactly
                    # Conservative: assume same place, just need drop
                    base_cost = 1
                else:
                    base_cost = 1
            elif self._is_crate_in_some_truck(crate, state):
                # Crate is in truck - need unload + drop = 2
                # May also need drive = 3
                base_cost = 2
            elif crate_place is not None and surface_place is not None:
                if crate_place == surface_place:
                    # Same place: lift + drop = 2
                    base_cost = 2
                else:
                    # Different place: lift + load + drive + unload + drop = 5
                    base_cost = 5
            else:
                # Fallback
                base_cost = 2
            
            # Add clearing cost for the target surface
            if not self._is_clear(surface, state):
                # Need to clear the surface first
                # Each crate on top needs at least 4 actions to remove
                # (lift + load + possibly more)
                # But conservatively, at least 2 per crate (lift + load or lift + drop)
                stack_height = self._count_crates_on_top(surface, state)
                base_cost += stack_height * 2
            
            # Add clearing cost for the crate itself (if something is on top of it)
            if crate_place is not None and not self._is_clear(crate, state):
                stack_height = self._count_crates_on_top(crate, state)
                base_cost += stack_height * 2
            
            total_cost += base_cost
        
        return total_cost

    def _task_network_term(self, state, task_network):
        """Estimate cost from remaining task network.
        
        For each task in the network:
        - nop operators: 0
        - primitive operators: 1  
        - do_put_on tasks: estimate based on state
        - other abstract tasks: 1 (conservative)
        """
        total = 0
        
        for task in task_network:
            gid = task.global_id
            
            if gid in self._nop_op_ids:
                continue
            
            # Check if it's an operator (primitive)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                total += 1
                continue
            
            # Check if it's an abstract task
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                if gid in self._do_put_on_task_ids:
                    # Estimate cost for do_put_on
                    total += self._estimate_put_on_cost(task, state)
                else:
                    # Other abstract tasks (do_clear, do_get_truck, do_lift_crate, etc.)
                    total += self._estimate_abstract_task_cost(task, state)
                continue
            
            # Fallback for decompositions or unknown
            total += 1
        
        return total

    def _estimate_put_on_cost(self, task, state):
        """Estimate minimum cost for a do_put_on task.
        
        Parse task name to get crate and surface arguments,
        then estimate based on current state.
        """
        # Parse task name to extract arguments
        # Task names look like: do_put_on[crate0,pallet1] or similar
        name = task.name
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            if len(args) >= 2:
                crate = args[0].strip()
                surface = args[1].strip()
                
                # Check if already on target
                on_fid = self._on_fact_ids.get((crate, surface), -1)
                if on_fid >= 0 and (state >> on_fid) & 1:
                    return 0  # Will use m0 (nop)
                
                crate_place = self._get_crate_place(crate, state)
                surface_place = self._get_surface_place(surface, state)
                
                cost = 0
                
                if self._is_crate_in_some_truck(crate, state):
                    # Method m2: do_get_truck + do_clear + Unload + Drop
                    # Minimum: 2 actions (Unload + Drop) + possible drive
                    cost = 2
                    if surface_place is not None and not self._is_clear(surface, state):
                        cost += self._count_crates_on_top(surface, state) * 2
                elif crate_place is not None:
                    if surface_place is not None and crate_place == surface_place:
                        # Method m1: do_clear(crate) + do_clear(surface) + do_lift_crate + Drop
                        # Minimum: Lift + Drop = 2
                        cost = 2
                        if not self._is_clear(crate, state):
                            cost += self._count_crates_on_top(crate, state) * 2
                        if not self._is_clear(surface, state):
                            cost += self._count_crates_on_top(surface, state) * 2
                    else:
                        # Method m3: do_load_truck + Drive + do_unload_truck
                        # Minimum: Lift + Load + Drive + Unload + Drop = 5
                        cost = 5
                        if not self._is_clear(crate, state):
                            cost += self._count_crates_on_top(crate, state) * 2
                        if surface_place is not None and not self._is_clear(surface, state):
                            cost += self._count_crates_on_top(surface, state) * 2
                elif self._is_crate_being_lifted(crate, state):
                    # Crate is being lifted, need at minimum drop
                    cost = 1
                    if surface_place is not None and not self._is_clear(surface, state):
                        cost += self._count_crates_on_top(surface, state) * 2
                else:
                    # Fallback
                    cost = 2
                
                return max(cost, 1)  # At least 1 if not already done
        
        # Fallback: can't parse, assume moderate cost
        return 2

    def _estimate_abstract_task_cost(self, task, state):
        """Estimate cost for non-do_put_on abstract tasks."""
        name = task.name
        
        if "do_clear" in name:
            # Parse to get surface and place
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                if len(args) >= 2:
                    surface = args[0].strip()
                    # Check if already clear
                    if self._is_clear(surface, state):
                        return 0  # nop
                    # Need to clear: at least lift + load = 2 per crate on top
                    stack = self._count_crates_on_top(surface, state)
                    return max(stack * 4, 2)  # lift+load+get_truck+possible drive per crate
            return 1
        
        elif "do_get_truck" in name:
            # Parse to get truck and place
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                if len(args) >= 2:
                    truck = args[0].strip()
                    place = args[1].strip()
                    at_fid = self._at_fact_ids.get((truck, place), -1)
                    if at_fid >= 0 and (state >> at_fid) & 1:
                        return 0  # Already there, nop
                    return 1  # Drive
            return 1
        
        elif "do_lift_crate" in name:
            # At minimum: 1 action (Lift or Unload)
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                if len(args) >= 1:
                    crate = args[0].strip()
                    if self._is_crate_being_lifted(crate, state):
                        return 0
            return 1
        
        elif "do_load_truck" in name:
            # Lift + Load = 2 minimum, plus possible truck driving and clearing
            return 2
        
        elif "do_unload_truck" in name:
            # Unload + Drop = 2 minimum, plus possible clearing
            return 2
        
        # Generic fallback
        return 1
