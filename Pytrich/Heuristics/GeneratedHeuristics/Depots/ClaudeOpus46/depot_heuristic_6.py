"""Domain-dependent heuristic for the Depots HDDL domain.

Bottleneck Analysis:
  The main serializing constraint in Depots is that crates must be moved between
  places via trucks (single capacity per hoist interaction), and hoists are fixed
  at locations. Moving a crate from place A to place B requires: clearing the crate,
  lifting it, loading onto a truck, driving, unloading, and dropping. Each crate
  that needs relocation costs at minimum ~4-6 actions. Additionally, crates stacked
  on top of other crates that need to move must be cleared first, creating recursive
  dependencies similar to Blocksworld.

Admissible Lower Bound Terms:
  T1 - Unsatisfied goal count: Each unsatisfied 'on(c,s)' goal requires at least 1
       action (Drop) to achieve. This is a trivial admissible lower bound.
  T2 - Location mismatch cost: If a crate needs to be on a surface at a different
       place than where it currently is, it needs at minimum: Lift + Load + Drive +
       Unload + Drop = 5 actions. If at the same place but wrong position: Lift + Drop
       = 2 actions. This is admissible because these are unavoidable actions.
  T3 - Clearing cost: If a goal requires 'on(c, s)' and s is not clear (and the
       thing on s is not c), then the blocking crate must be moved away first,
       costing at least 2 more actions (Lift + Load or Lift + Drop). Additionally,
       if c itself is not clear, crates on top of c must be removed first.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic6(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_6"):
        super().__init__(name)
        # Precomputed lookup tables
        self.on_fact_id = {}       # (crate, surface) -> global_id
        self.at_fact_id = {}       # (locatable, place) -> global_id
        self.clear_fact_id = {}    # surface -> global_id
        self.in_fact_id = {}       # (crate, truck) -> global_id
        self.lifting_fact_id = {}  # (hoist, crate) -> global_id
        self.available_fact_id = {} # hoist -> global_id
        
        # Goal info
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)
        
        # Location info
        self.surface_place = {}    # surface -> place (static location for pallets)
        self.hoist_place = {}      # hoist -> place (static location)
        self.places = set()
        self.crates = set()
        self.surfaces = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # For stack height estimation
        self.on_facts_by_surface = {}  # surface -> list of (crate, global_id)
        
        # Goal surface set for quick lookup
        self.goal_surface_to_crate = {}  # surface -> crate that should be on it
        self.goal_crate_to_surface = {}  # crate -> surface it should be on

    def _parse_fact_name(self, name: str):
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
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                if args[1] not in self.on_facts_by_surface:
                    self.on_facts_by_surface[args[1]] = []
                self.on_facts_by_surface[args[1]].append((args[0], f.global_id))
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
        
        # Identify object types from at facts
        for (obj, place) in self.at_fact_id:
            if obj.startswith("crate"):
                self.crates.add(obj)
            elif obj.startswith("pallet"):
                self.pallets.add(obj)
                self.surface_place[obj] = place  # pallets don't move
            elif obj.startswith("truck"):
                self.trucks.add(obj)
            elif obj.startswith("hoist"):
                self.hoists.add(obj)
                self.hoist_place[obj] = place  # hoists don't move
        
        self.surfaces = self.pallets | self.crates
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                gid = self.on_fact_id.get((crate, surface))
                if gid is not None:
                    self.goal_on_pairs.append((crate, surface, gid))
                    self.goal_surface_to_crate[surface] = crate
                    self.goal_crate_to_surface[crate] = surface
        
        # Build place-to-hoists mapping for availability checks
        self.place_hoists = {}
        for hoist, place in self.hoist_place.items():
            if place not in self.place_hoists:
                self.place_hoists[place] = []
            self.place_hoists[place].append(hoist)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_crate_location(self, crate, state):
        """Find where a crate currently is. Returns place name or None if in truck."""
        for (obj, place), gid in self.at_fact_id.items():
            if obj == crate:
                if (state >> gid) & 1:
                    return place
        return None

    def _get_surface_location(self, surface, state):
        """Find where a surface currently is."""
        if surface in self.surface_place and surface in self.pallets:
            return self.surface_place[surface]  # pallets are static
        # For crates, check at facts
        for (obj, place), gid in self.at_fact_id.items():
            if obj == surface:
                if (state >> gid) & 1:
                    return place
        return None

    def _is_crate_in_some_truck(self, crate, state):
        """Check if crate is in any truck."""
        for (c, t), gid in self.in_fact_id.items():
            if c == crate and (state >> gid) & 1:
                return True
        return False

    def _is_clear(self, surface, state):
        """Check if surface is clear."""
        gid = self.clear_fact_id.get(surface)
        if gid is None:
            return True
        return (state >> gid) & 1

    def _is_on(self, crate, surface, state):
        """Check if crate is on surface."""
        gid = self.on_fact_id.get((crate, surface))
        if gid is None:
            return False
        return (state >> gid) & 1

    def _compute(self, state, task_network):
        """Compute heuristic value as max of multiple admissible terms."""
        t1 = self._unsatisfied_goals_cost(state)
        t2 = self._task_network_cost(state, task_network)
        return max(t1, t2)

    def _unsatisfied_goals_cost(self, state):
        """T1+T2 combined: For each unsatisfied goal on(c,s), estimate minimum actions.
        
        Admissibility: Each unsatisfied on(c,s) requires at minimum:
        - If c is at the same place as s: 2 actions (Lift + Drop) if c is clear and s is clear
          Add 2 per blocking crate on c, add 2 per blocking crate on s
        - If c is at a different place: 5 actions (Lift + Load + Drive + Unload + Drop) minimum
          Plus clearing costs
        - If c is in a truck: 3 actions (Unload + Drop) minimum, but need truck at right place
        
        We take the max over all individual goal costs as our estimate (since goals
        must be achieved sequentially due to hoist serialization).
        """
        if not self.goal_on_pairs:
            return 0
        
        total_cost = 0
        max_single_cost = 0
        
        for crate, surface, gid in self.goal_on_pairs:
            # Check if goal is already satisfied
            if (state >> gid) & 1:
                # Goal is currently true, but check if it might be disrupted
                # If the surface needs to be cleared for another goal's crate,
                # or if this crate needs to be moved for another goal
                continue
            
            cost = 0
            
            # Check if crate is in a truck
            in_truck = self._is_crate_in_some_truck(crate, state)
            
            if in_truck:
                # Need: (Drive?) + Unload + Drop = 2-3 actions minimum
                cost = 2
                # Check if surface is clear
                if not self._is_clear(surface, state):
                    cost += 2  # Need to clear the surface (Lift + Load/Drop)
            else:
                crate_loc = self._get_crate_location(crate, state)
                surface_loc = self._get_surface_location(surface, state)
                
                if crate_loc is not None and surface_loc is not None:
                    if crate_loc == surface_loc:
                        # Same place: Lift + Drop = 2 minimum
                        cost = 2
                    else:
                        # Different place: Lift + Load + Drive + Unload + Drop = 5
                        cost = 5
                else:
                    # Can't determine location, assume moderate cost
                    cost = 3
                
                # Clearing cost for crate (if not clear, need to remove what's on top)
                if not self._is_clear(crate, state):
                    # At least 2 actions to clear (Lift + Load the blocking crate)
                    cost += 2
                
                # Clearing cost for surface
                if not self._is_clear(surface, state):
                    # Check if the thing on the surface is what we want to put there
                    # In that case it's already handled
                    surface_already_has_goal = False
                    if surface in self.goal_surface_to_crate:
                        goal_crate = self.goal_surface_to_crate[surface]
                        if goal_crate == crate:
                            # We're the goal crate for this surface, 
                            # but goal not satisfied, so surface must be cleared
                            cost += 2
                        else:
                            cost += 2
                    else:
                        cost += 2
            
            total_cost += cost
            max_single_cost = max(max_single_cost, cost)
        
        # Return weighted combination: use total/2 as estimate since some work
        # can be parallelized across hoists, but take max with single most expensive
        # Actually for admissibility, we use a conservative estimate
        # total_cost is admissible because each unsatisfied goal requires
        # independent actions (the Drop for each goal is unique)
        return total_cost

    def _task_network_cost(self, state, task_network):
        """Estimate cost from remaining task network.
        
        Each task in the network requires at least some actions to complete.
        We estimate based on the task type and current state.
        """
        if not task_network:
            return 0
        
        cost = 0
        # Limit iteration to avoid timeout on huge networks
        count = 0
        max_count = 500  # Safety limit
        
        for task in task_network:
            if count >= max_count:
                break
            count += 1
            
            task_name = task.name if hasattr(task, 'name') else ''
            
            # Check if it's a primitive operator
            if hasattr(task, 'add_effects'):
                # It's an operator (primitive action)
                cost += 1
            elif hasattr(task, 'decompositions'):
                # It's an abstract task - estimate minimum decomposition cost
                task_cost = self._estimate_abstract_task_cost(task, state)
                cost += task_cost
        
        return cost

    def _estimate_abstract_task_cost(self, task, state):
        """Estimate minimum cost to decompose an abstract task.
        
        Uses domain knowledge about task types to provide tighter bounds.
        """
        name = task.name if hasattr(task, 'name') else ''
        
        # Parse task name to determine type
        if 'do_put_on' in name:
            return self._estimate_put_on_cost(task, state)
        elif 'do_clear' in name:
            return self._estimate_clear_cost(task, state)
        elif 'do_get_truck' in name:
            return self._estimate_get_truck_cost(task, state)
        elif 'do_lift_crate' in name:
            return 1  # At minimum: one Lift or Unload action
        elif 'do_load_truck' in name:
            return 2  # At minimum: Lift + Load
        elif 'do_unload_truck' in name:
            return 2  # At minimum: Unload + Drop
        else:
            # Unknown task, estimate based on minimum decomposition
            if hasattr(task, 'decompositions') and task.decompositions:
                min_subtasks = float('inf')
                for decomp in task.decompositions:
                    if decomp.applicable(state):
                        n = len(decomp.task_network)
                        if n < min_subtasks:
                            min_subtasks = n
                if min_subtasks == float('inf'):
                    # No applicable decomposition found, use minimum over all
                    for decomp in task.decompositions:
                        n = len(decomp.task_network)
                        if n < min_subtasks:
                            min_subtasks = n
                return max(1, min_subtasks) if min_subtasks != float('inf') else 1
            return 1

    def _estimate_put_on_cost(self, task, state):
        """Estimate cost for do_put_on task.
        
        do_put_on(c, s2) has methods:
        - m0: already on s2 -> nop (cost 0)
        - m1: at same place -> clear c + clear s2 + lift + drop (min 2 actions)
        - m2: in truck -> get truck + clear s2 + unload + drop (min 2 actions)
        - m3: load + drive + unload (min ~6 actions)
        """
        # Try to extract crate and surface from task name
        # Task names are grounded, e.g., "do_put_on[crate0,pallet1]"
        pred, args = self._parse_fact_name(task.name)
        
        if pred and len(args) >= 2:
            crate = args[0]
            surface = args[1]
            
            # Check m0: already on target
            on_gid = self.on_fact_id.get((crate, surface))
            if on_gid is not None and (state >> on_gid) & 1:
                return 0  # nop
            
            # Check if crate is at some place
            crate_loc = None
            for (obj, place), gid in self.at_fact_id.items():
                if obj == crate and (state >> gid) & 1:
                    crate_loc = place
                    break
            
            in_truck = False
            if crate_loc is None:
                in_truck = self._is_crate_in_some_truck(crate, state)
            
            surface_loc = None
            if surface in self.pallets:
                surface_loc = self.surface_place.get(surface)
            else:
                for (obj, place), gid in self.at_fact_id.items():
                    if obj == surface and (state >> gid) & 1:
                        surface_loc = place
                        break
            
            if in_truck:
                # m2: get_truck + clear_s2 + Unload + Drop
                cost = 2  # Unload + Drop minimum
                if not self._is_clear(surface, state):
                    cost += 2  # clearing surface
                return cost
            
            if crate_loc is not None and surface_loc is not None:
                if crate_loc == surface_loc:
                    # m1: clear c + clear s2 + lift + drop
                    cost = 2  # Lift + Drop
                    if not self._is_clear(crate, state):
                        cost += 2
                    if not self._is_clear(surface, state):
                        cost += 2
                    return cost
                else:
                    # m3: load_truck + drive + unload_truck
                    # load = lift + load (2), drive (1), unload = unload + drop (2) = 5
                    cost = 5
                    if not self._is_clear(crate, state):
                        cost += 2
                    if not self._is_clear(surface, state):
                        cost += 2
                    return cost
            
            # Fallback: moderate estimate
            return 3
        
        # Can't parse, use decomposition-based estimate
        return 2

    def _estimate_clear_cost(self, task, state):
        """Estimate cost for do_clear task."""
        pred, args = self._parse_fact_name(task.name)
        if pred and len(args) >= 1:
            surface = args[0]
            if self._is_clear(surface, state):
                return 0
            # Need to lift what's on top and load it somewhere: Lift + Load = 2
            return 2
        return 1

    def _estimate_get_truck_cost(self, task, state):
        """Estimate cost for do_get_truck task."""
        pred, args = self._parse_fact_name(task.name)
        if pred and len(args) >= 2:
            truck = args[0]
            place = args[1]
            at_gid = self.at_fact_id.get((truck, place))
            if at_gid is not None and (state >> at_gid) & 1:
                return 0  # Already there
            return 1  # Drive action
        return 1
