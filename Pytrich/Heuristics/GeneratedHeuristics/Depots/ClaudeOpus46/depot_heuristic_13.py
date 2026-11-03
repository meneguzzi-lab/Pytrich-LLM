# Depots Domain Heuristic
# Bottleneck: Stack clearing and inter-place transport of crates.
# Each crate placement goal requires clearing source stack, clearing target,
# and potentially transporting between places (lift+load+drive+unload+drop).
#
# Admissible lower bounds:
# T1: Per unsatisfied on-goal, minimum actions needed based on crate location vs target location
#     - same place & crate accessible: 2 (lift+drop)
#     - different place: 5 (lift+load+drive+unload+drop)
#     - crate in truck: 2 (unload+drop) or 3 if truck not at target place
#     Admissible: these are the minimum possible action sequences.
# T2: Stack clearing cost - for each crate that must move, count blocking crates above it
#     Each blocker needs at least 2 actions (lift+load) to remove.
#     Admissible: can't clear a blocker in fewer than 2 actions.
# T3: Task network lower bound - remaining compound tasks estimate
#     Each non-nop primitive task costs 1; each compound task costs at least 1.
#     Admissible: every task needs at least 1 action.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic13(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_13"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_ids = {}      # (crate, surface) -> global_id
        self.at_fact_ids = {}      # (locatable, place) -> global_id
        self.in_fact_ids = {}      # (crate, truck) -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.lifting_fact_ids = {} # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)
        
        # Object collections
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # Location maps for objects
        self.crate_at_facts = {}   # crate -> list of (place, fact_global_id)
        self.surface_at_facts = {} # surface -> list of (place, fact_global_id)
        self.truck_at_facts = {}   # truck -> list of (place, fact_global_id)
        self.hoist_at_facts = {}   # hoist -> list of (place, fact_global_id)
        
        # on-relationship maps
        self.on_crate_facts = {}   # crate -> list of (surface, fact_global_id)  what crate is ON
        self.on_surface_facts = {} # surface -> list of (crate, fact_global_id)  what is ON surface
        
        # in-relationship maps
        self.in_crate_facts = {}   # crate -> list of (truck, fact_global_id)
        
        # Hoist at place map
        self.hoists_at_place = {}  # place -> list of hoist names
        
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
    
    def _get_crate_location(self, state, crate):
        """Get current place of a crate, or None if not at any place (in truck or being lifted)"""
        if crate in self.crate_at_facts:
            for place, fid in self.crate_at_facts[crate]:
                if (state >> fid) & 1:
                    return place
        return None
    
    def _get_surface_location(self, state, surface):
        """Get current place of a surface"""
        if surface in self.surface_at_facts:
            for place, fid in self.surface_at_facts[surface]:
                if (state >> fid) & 1:
                    return place
        return None
    
    def _get_truck_location(self, state, truck):
        """Get current place of a truck"""
        if truck in self.truck_at_facts:
            for place, fid in self.truck_at_facts[truck]:
                if (state >> fid) & 1:
                    return place
        return None
    
    def _is_crate_in_truck(self, state, crate):
        """Check if crate is in any truck, return truck name or None"""
        if crate in self.in_crate_facts:
            for truck, fid in self.in_crate_facts[crate]:
                if (state >> fid) & 1:
                    return truck
        return None
    
    def _is_crate_being_lifted(self, state, crate):
        """Check if crate is being lifted by any hoist"""
        if crate in self.lifting_by_crate:
            for hoist, fid in self.lifting_by_crate[crate]:
                if (state >> fid) & 1:
                    return hoist
        return None
    
    def _is_clear(self, state, surface):
        """Check if surface is clear"""
        if surface in self.clear_fact_ids:
            return (state >> self.clear_fact_ids[surface]) & 1
        return False
    
    def _what_is_on(self, state, surface):
        """Find what crate is directly on this surface"""
        if surface in self.on_surface_facts:
            for crate, fid in self.on_surface_facts[surface]:
                if (state >> fid) & 1:
                    return crate
        return None
    
    def _count_stack_above(self, state, surface, max_depth=50):
        """Count how many crates are stacked above this surface"""
        count = 0
        current = surface
        for _ in range(max_depth):
            crate_above = self._what_is_on(state, current)
            if crate_above is None:
                break
            count += 1
            current = crate_above
        return count
    
    def _crate_on_what(self, state, crate):
        """Find what surface this crate is on"""
        if crate in self.on_crate_facts:
            for surface, fid in self.on_crate_facts[crate]:
                if (state >> fid) & 1:
                    return surface
        return None
    
    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build lifting by crate map
        self.lifting_by_crate = {}  # crate -> list of (hoist, fid)
        
        # Parse all facts and build maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                self.on_fact_ids[(crate, surface)] = f.global_id
                self.crates.add(crate)
                self.surfaces.add(surface)
                self.surfaces.add(crate)  # crates are also surfaces
                
                if crate not in self.on_crate_facts:
                    self.on_crate_facts[crate] = []
                self.on_crate_facts[crate].append((surface, f.global_id))
                
                if surface not in self.on_surface_facts:
                    self.on_surface_facts[surface] = []
                self.on_surface_facts[surface].append((crate, f.global_id))
                
            elif pred == "at" and len(args) == 2:
                obj, place = args[0], args[1]
                self.at_fact_ids[(obj, place)] = f.global_id
                self.places.add(place)
                
                # Categorize by object type based on naming
                if obj.startswith("crate"):
                    self.crates.add(obj)
                    if obj not in self.crate_at_facts:
                        self.crate_at_facts[obj] = []
                    self.crate_at_facts[obj].append((place, f.global_id))
                elif obj.startswith("truck"):
                    self.trucks.add(obj)
                    if obj not in self.truck_at_facts:
                        self.truck_at_facts[obj] = []
                    self.truck_at_facts[obj].append((place, f.global_id))
                elif obj.startswith("hoist"):
                    self.hoists.add(obj)
                    if obj not in self.hoist_at_facts:
                        self.hoist_at_facts[obj] = []
                    self.hoist_at_facts[obj].append((place, f.global_id))
                    # Also record static hoist locations
                    if place not in self.hoists_at_place:
                        self.hoists_at_place[place] = []
                    self.hoists_at_place[place].append(obj)
                elif obj.startswith("pallet"):
                    self.pallets.add(obj)
                    if obj not in self.surface_at_facts:
                        self.surface_at_facts[obj] = []
                    self.surface_at_facts[obj].append((place, f.global_id))
                    
            elif pred == "in" and len(args) == 2:
                crate, truck = args[0], args[1]
                self.in_fact_ids[(crate, truck)] = f.global_id
                self.crates.add(crate)
                self.trucks.add(truck)
                
                if crate not in self.in_crate_facts:
                    self.in_crate_facts[crate] = []
                self.in_crate_facts[crate].append((truck, f.global_id))
                
            elif pred == "clear" and len(args) == 1:
                surface = args[0]
                self.clear_fact_ids[surface] = f.global_id
                
            elif pred == "lifting" and len(args) == 2:
                hoist, crate = args[0], args[1]
                self.lifting_fact_ids[(hoist, crate)] = f.global_id
                if crate not in self.lifting_by_crate:
                    self.lifting_by_crate[crate] = []
                self.lifting_by_crate[crate].append((hoist, f.global_id))
                
            elif pred == "available" and len(args) == 1:
                hoist = args[0]
                self.available_fact_ids[hoist] = f.global_id
        
        # Also add crates to surface_at_facts since crates are surfaces too
        for crate in self.crates:
            if crate not in self.surface_at_facts:
                self.surface_at_facts[crate] = []
            if crate in self.crate_at_facts:
                for place, fid in self.crate_at_facts[crate]:
                    self.surface_at_facts[crate].append((place, fid))
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                fid = self.on_fact_ids.get((crate, surface))
                if fid is not None:
                    self.goal_on_pairs.append((crate, surface, fid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)
    
    def _compute(self, state, task_network):
        # Term 1: Goal-based placement cost
        t1 = self._goal_placement_cost(state)
        
        # Term 2: Task network estimate
        t2 = self._task_network_cost(state, task_network)
        
        return max(t1, t2)
    
    def _goal_placement_cost(self, state):
        """Compute admissible lower bound based on unsatisfied goal on-predicates.
        
        For each unsatisfied on(crate, surface) goal:
        - If crate is being lifted at the target place: 1 (just Drop)
        - If crate is at the same place as target surface and accessible: 2 (Lift+Drop)
        - If crate is at same place but not clear: 2 + 2*blockers (clear blockers + Lift+Drop)
        - If crate is in a truck: 2 (Unload+Drop) if truck at target, 3 (Drive+Unload+Drop) otherwise
        - If crate is at a different place: 4 (Lift+Load+Drive+Unload+Drop = 5, but min 4 with overlap)
        We also need to clear the target surface if it's not clear.
        """
        total_cost = 0
        
        for crate, surface, fid in self.goal_on_pairs:
            # Check if goal already satisfied
            if (state >> fid) & 1:
                # Goal is currently true, but check if it will need to be undone
                # (goal stability check - if something above crate needs to move for another goal)
                continue
            
            cost = 0
            
            # Check current state of crate
            lifting_hoist = self._is_crate_being_lifted(state, crate)
            if lifting_hoist is not None:
                # Crate is being lifted
                # Need to check if hoist is at target place
                target_place = self._get_surface_location(state, surface)
                if target_place is not None:
                    hoist_at_target = False
                    if (lifting_hoist, target_place) in self.at_fact_ids:
                        hfid = self.at_fact_ids[(lifting_hoist, target_place)]
                        hoist_at_target = (state >> hfid) & 1
                    if hoist_at_target:
                        cost = 1  # Just Drop
                    else:
                        # Need to load into truck, drive, unload, drop = 4
                        cost = 4
                else:
                    cost = 1  # conservative
            else:
                in_truck = self._is_crate_in_truck(state, crate)
                if in_truck is not None:
                    # Crate is in a truck
                    target_place = self._get_surface_location(state, surface)
                    if target_place is not None:
                        truck_place = self._get_truck_location(state, in_truck)
                        if truck_place == target_place:
                            cost = 2  # Unload + Drop
                        else:
                            cost = 3  # Drive + Unload + Drop
                    else:
                        cost = 2
                else:
                    # Crate is on some surface at some place
                    crate_place = self._get_crate_location(state, crate)
                    target_place = self._get_surface_location(state, surface)
                    
                    if crate_place is None or target_place is None:
                        cost = 2  # minimum: lift+drop or similar
                    elif crate_place == target_place:
                        # Same place - need Lift + Drop = 2
                        # But may need to clear crate first
                        if not self._is_clear(state, crate):
                            blockers = self._count_stack_above(state, crate, max_depth=20)
                            cost = 2 + 2 * blockers  # each blocker: Lift+Load
                        else:
                            cost = 2  # Lift + Drop
                    else:
                        # Different place - need at minimum Lift+Load+Drive+Unload+Drop = 5
                        if not self._is_clear(state, crate):
                            blockers = self._count_stack_above(state, crate, max_depth=20)
                            cost = 5 + 2 * blockers
                        else:
                            cost = 5
            
            # Also check if target surface needs clearing
            if not self._is_clear(state, surface):
                # Check if what's on the surface is supposed to be there per goals
                thing_on = self._what_is_on(state, surface)
                if thing_on is not None:
                    # Check if goal says this thing should be on this surface
                    goal_fid = self.on_fact_ids.get((thing_on, surface))
                    if goal_fid is not None and (state >> goal_fid) & 1:
                        # This is a goal that's currently true but needs to be undone
                        # to place our crate here - but wait, maybe another crate goes on top
                        # Actually, if on(thing, surface) is a goal AND on(crate, surface) is also a goal,
                        # that can't happen (surface can only have one thing directly on it)
                        # So thing_on needs to move: at least 2 more actions
                        cost += 2
                    else:
                        # thing_on is not in goal position, needs to move anyway
                        # Don't double count - it will have its own goal cost
                        # But we still need to account for clearing work
                        # Minimum 2 actions to clear (Lift+Load or Lift+Drop elsewhere)
                        cost += 2
            
            total_cost += cost
        
        return total_cost
    
    def _task_network_cost(self, state, task_network):
        """Estimate minimum actions from remaining task network.
        
        For each task in the network:
        - Primitive operator (not nop): 1 action
        - nop: 0 actions
        - Abstract task: estimate based on task type and state
        """
        if not task_network:
            return 0
        
        cost = 0
        # Limit how many tasks we examine for performance
        max_examine = min(len(task_network), 200)
        
        for i in range(max_examine):
            task = task_network[i]
            task_gid = task.global_id
            
            # Check if it's an operator
            if self.model.iop_init <= task_gid <= self.model.iop_end:
                # It's a primitive operator
                if "nop" in task.name:
                    cost += 0
                else:
                    cost += 1
            elif self.model.iabt_init <= task_gid <= self.model.iabt_end:
                # It's an abstract task - estimate based on type
                cost += self._estimate_abstract_task_cost(state, task)
        
        return cost
    
    def _estimate_abstract_task_cost(self, state, task):
        """Estimate minimum decomposition cost for an abstract task."""
        name = task.name
        
        # Parse task name to get type and arguments
        # Format: __do_put_on[crate0,pallet1] or similar
        bracket_idx = name.find("[")
        if bracket_idx < 0:
            return 1
        
        task_type = name[:bracket_idx]
        args_str = name[bracket_idx+1:-1] if name.endswith("]") else ""
        args = args_str.split(",") if args_str else []
        
        # Remove leading underscores from task type
        clean_type = task_type.lstrip("_")
        
        if clean_type == "do_put_on" and len(args) == 2:
            crate, surface = args[0], args[1]
            # Check if already done (m0)
            fid = self.on_fact_ids.get((crate, surface))
            if fid is not None and (state >> fid) & 1:
                return 0  # nop
            
            # Check if crate is at same place as surface
            crate_place = self._get_crate_location(state, crate)
            in_truck = self._is_crate_in_truck(state, crate)
            lifting = self._is_crate_being_lifted(state, crate)
            
            if lifting is not None:
                return 1  # just Drop
            elif in_truck is not None:
                # m2: do_get_truck + do_clear + Unload + Drop = at least 2
                return 2
            elif crate_place is not None:
                target_place = self._get_surface_location(state, surface)
                if target_place is not None and crate_place == target_place:
                    # m1: do_clear(c) + do_clear(s2) + do_lift_crate + Drop
                    min_cost = 1  # Drop
                    # do_lift_crate: at least Lift = 1
                    min_cost += 1
                    # do_clear(c): 0 if clear, else 2+ per blocker
                    if not self._is_clear(state, crate):
                        blockers = self._count_stack_above(state, crate, max_depth=10)
                        min_cost += 2 * blockers  # Lift+Load per blocker
                    # do_clear(s2): 0 if clear, else 2+ per blocker
                    if not self._is_clear(state, surface):
                        blockers_s = self._count_stack_above(state, surface, max_depth=10)
                        # But be careful: what's on surface might be the crate itself
                        top_of_surface = self._what_is_on(state, surface)
                        if top_of_surface != crate:
                            min_cost += 2 * max(1, blockers_s)
                    return min_cost
                else:
                    # m3: do_load_truck + Drive + do_unload_truck = at least 5
                    min_cost = 5
                    if not self._is_clear(state, crate):
                        blockers = self._count_stack_above(state, crate, max_depth=10)
                        min_cost += 2 * blockers
                    return min_cost
            else:
                return 2  # minimum
                
        elif clean_type == "do_clear" and len(args) == 2:
            surface, place = args[0], args[1]
            if self._is_clear(state, surface):
                return 0
            # Need to clear: at least Lift + Load = 2 per crate on top
            blockers = self._count_stack_above(state, surface, max_depth=10)
            return 2 * max(1, blockers)
            
        elif clean_type == "do_get_truck" and len(args) == 2:
            truck, place = args[0], args[1]
            truck_place = self._get_truck_location(state, truck)
            if truck_place == place:
                return 0
            return 1  # Drive
            
        elif clean_type == "do_lift_crate" and len(args) == 3:
            crate, place, hoist = args[0], args[1], args[2]
            in_truck = self._is_crate_in_truck(state, crate)
            if in_truck is not None:
                return 2  # do_get_truck + Unload (at least 1+1)
            return 1  # Lift
            
        elif clean_type == "do_load_truck" and len(args) == 4:
            # do_get_truck + do_clear + Lift + Load
            crate = args[0]
            min_cost = 2  # Lift + Load
            if not self._is_clear(state, crate):
                blockers = self._count_stack_above(state, crate, max_depth=10)
                min_cost += 2 * blockers
            return min_cost
            
        elif clean_type == "do_unload_truck" and len(args) == 4:
            # do_clear + Unload + Drop = at least 2
            surface = args[1]
            min_cost = 2  # Unload + Drop
            if not self._is_clear(state, surface):
                blockers_s = self._count_stack_above(state, surface, max_depth=10)
                min_cost += 2 * max(1, blockers_s)
            return min_cost
        
        return 1  # default: at least 1 action
    
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
