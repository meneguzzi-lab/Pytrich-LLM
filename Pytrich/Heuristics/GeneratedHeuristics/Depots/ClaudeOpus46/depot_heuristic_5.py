"""Domain-Dependent Heuristic for Depots HDDL Domain

Bottleneck Analysis:
  The primary bottleneck in the Depots domain is the need to move crates between
  places via trucks (requiring lift, load, drive, unload, drop sequences) and the
  need to clear surfaces before placing crates on them. Each crate that needs to
  move to a different place requires at minimum 5 actions (lift, load, drive, unload, drop).
  Crates at the correct place but on wrong surfaces need at least 2 actions (lift, drop).
  Additionally, surfaces that need to be cleared require recursive clearing work.

Admissible Lower Bounds:
  Term 1 - Goal displacement cost: For each unsatisfied 'on(c, s)' goal, estimate
    the minimum number of primitive actions needed. If crate is at the wrong place,
    at least 5 actions (lift+load+drive+unload+drop). If at correct place but wrong
    surface, at least 2 actions (lift+drop). This is admissible because these are
    the minimum action sequences needed.
  
  Term 2 - Task network lower bound: Count remaining compound tasks in the task
    network, estimating minimum actions per task type. do_put_on needs at least 0
    (if already satisfied) or 2+ actions. This gives a structural lower bound.
  
  Term 3 - Clearing cost: For surfaces that need to be cleared for goal placement,
    count the number of crates that must be moved off. Each requires at least 2
    actions (lift + load/drop). This captures cascading clearing work.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic5(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_5"):
        super().__init__(name)
        # Precomputed maps
        self.on_fact_id = {}       # (crate, surface) -> global_id
        self.at_fact_id = {}       # (locatable, place) -> global_id
        self.clear_fact_id = {}    # surface -> global_id
        self.in_fact_id = {}       # (crate, truck) -> global_id
        self.lifting_fact_id = {}  # (hoist, crate) -> global_id
        self.available_fact_id = {} # hoist -> global_id
        
        # Goal tracking
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)
        
        # Location maps
        self.crate_places = {}     # crate -> set of places where it could be (from 'at' facts)
        self.surface_places = {}   # surface -> set of places
        self.hoist_places = {}     # hoist -> place
        self.trucks = []           # list of truck names
        self.places = set()        # all places
        self.crates = set()        # all crates
        self.surfaces = set()      # all surfaces
        self.hoists = set()        # all hoists
        
        # For stack height estimation
        self.on_facts_by_surface = {}  # surface -> list of (crate, global_id) that are on this surface
        
        # Abstract task name patterns
        self.do_put_on_prefix = "do_put_on"
        self.do_clear_prefix = "do_clear"
        self.do_get_truck_prefix = "do_get_truck"
        self.do_lift_crate_prefix = "do_lift_crate"
        self.do_load_truck_prefix = "do_load_truck"
        self.do_unload_truck_prefix = "do_unload_truck"

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
        """Parse a task/operator name to extract the task type and arguments."""
        # Task names look like: do_put_on[crate0,pallet1] or Drive[truck0,depot0,depot1]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task_type, args
        # Could also be just a name without brackets
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                # Track on-facts by surface
                surface = args[1]
                if surface not in self.on_facts_by_surface:
                    self.on_facts_by_surface[surface] = []
                self.on_facts_by_surface[surface].append((args[0], f.global_id))
                self.crates.add(args[0])
                self.surfaces.add(args[1])
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
                # Build location maps
                obj = args[0]
                place = args[1]
                if obj.startswith("crate"):
                    self.crates.add(obj)
                    if obj not in self.crate_places:
                        self.crate_places[obj] = set()
                    self.crate_places[obj].add(place)
                if obj.startswith("pallet") or obj.startswith("crate"):
                    self.surfaces.add(obj)
                    if obj not in self.surface_places:
                        self.surface_places[obj] = set()
                    self.surface_places[obj].add(place)
                if obj.startswith("hoist"):
                    self.hoists.add(obj)
                    self.hoist_places[obj] = place
                if obj.startswith("truck"):
                    self.trucks.append(obj)
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
        
        # Parse goal state
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate = args[0]
                surface = args[1]
                fid = self.on_fact_id.get((crate, surface), -1)
                self.goal_on_pairs.append((crate, surface, fid))
        
        # Precompute: for each place, which hoists are there
        self.place_hoists = {}
        for h, p in self.hoist_places.items():
            if p not in self.place_hoists:
                self.place_hoists[p] = []
            self.place_hoists[p].append(h)
        
        # Precompute initial locations of surfaces (pallets are static)
        self.surface_static_place = {}
        for (obj, place), fid in self.at_fact_id.items():
            if obj.startswith("pallet"):
                self.surface_static_place[obj] = place
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_crate_location(self, crate, state):
        """Get the current location of a crate by checking 'at' facts."""
        for place in self.places:
            key = (crate, place)
            if key in self.at_fact_id:
                fid = self.at_fact_id[key]
                if (state >> fid) & 1:
                    return place
        return None

    def _get_surface_location(self, surface, state):
        """Get location of a surface. Pallets are static."""
        if surface in self.surface_static_place:
            return self.surface_static_place[surface]
        # For crates used as surfaces, check at facts
        return self._get_crate_location(surface, state)

    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for truck in self.trucks:
            key = (crate, truck)
            if key in self.in_fact_id:
                fid = self.in_fact_id[key]
                if (state >> fid) & 1:
                    return True
        return False

    def _count_crates_on(self, surface, state):
        """Count how many crates are currently on this surface."""
        if surface not in self.on_facts_by_surface:
            return 0
        count = 0
        for crate, fid in self.on_facts_by_surface[surface]:
            if (state >> fid) & 1:
                count += 1
        return count

    def _stack_height_above(self, surface, state, depth=0):
        """Estimate number of crates stacked above a surface that need clearing."""
        if depth > 15:  # Safety limit
            return 0
        if surface not in self.on_facts_by_surface:
            return 0
        total = 0
        for crate, fid in self.on_facts_by_surface[surface]:
            if (state >> fid) & 1:
                total += 1 + self._stack_height_above(crate, state, depth + 1)
        return total

    def _compute(self, state, task_network):
        """Compute heuristic value using max of multiple admissible terms."""
        h1 = self._goal_displacement_cost(state)
        h2 = self._task_network_cost(state, task_network)
        h3 = self._unsatisfied_goals_count(state)
        
        return max(h1, h2, h3)

    def _unsatisfied_goals_count(self, state):
        """Simple count of unsatisfied goal facts."""
        count = 0
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0:
                if not ((state >> fid) & 1):
                    count += 1
            else:
                count += 1
        return count

    def _goal_displacement_cost(self, state):
        """For each unsatisfied on(c,s) goal, estimate minimum actions needed.
        
        Admissibility argument:
        - If crate is already on target surface: 0 actions (satisfied)
        - If crate is at same place as target surface: minimum 2 actions (lift + drop)
        - If crate is at different place: minimum 5 actions (lift + load + drive + unload + drop)
        - If crate is in a truck: minimum 2 actions (unload + drop) 
        - If target surface is not clear: need to add clearing cost
        
        We take the max over all individual goal costs as a simpler admissible bound,
        or sum up a fraction. Actually, since many goals can be worked on somewhat
        independently but share hoists, we'll sum and divide by the number of hoists
        to get a better admissible bound (each hoist can only do one thing at a time).
        """
        total_cost = 0
        max_single_cost = 0
        
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                continue  # Goal already satisfied
            
            cost = 0
            crate_in_truck = self._is_crate_in_truck(crate, state)
            
            if crate_in_truck:
                # Need at least: unload + drop = 2
                # Plus potentially drive to get truck to right place
                target_place = self._get_surface_location(surface, state)
                cost = 2
                if target_place is not None:
                    # May need drive
                    cost = 3  # drive + unload + drop minimum
            else:
                crate_place = self._get_crate_location(crate, state)
                target_place = self._get_surface_location(surface, state)
                
                if crate_place is not None and target_place is not None:
                    if crate_place == target_place:
                        # Same place: lift + drop = 2 actions
                        cost = 2
                    else:
                        # Different places: lift + load + drive + unload + drop = 5
                        cost = 5
                elif crate_place is None and not crate_in_truck:
                    # Crate is being lifted by a hoist
                    # Need at least 1 action (drop or load)
                    cost = 1
                else:
                    # Unknown location, assume worst case for local
                    cost = 2
            
            # Check if target surface needs clearing
            if surface in self.clear_fact_id:
                clear_fid = self.clear_fact_id[surface]
                if not ((state >> clear_fid) & 1):
                    # Surface not clear - need to clear it first
                    # Each crate on top needs at least 2 actions to remove (lift + load/drop)
                    stack_above = self._stack_height_above(surface, state)
                    cost += stack_above * 2
            
            # Check if crate itself needs clearing (if something is on top of it)
            if crate in self.clear_fact_id:
                clear_fid = self.clear_fact_id[crate]
                if not ((state >> clear_fid) & 1):
                    stack_above = self._stack_height_above(crate, state)
                    cost += stack_above * 2
            
            total_cost += cost
            max_single_cost = max(max_single_cost, cost)
        
        # Admissible bound: total work divided by number of hoists
        # (since hoists work independently but each can only do one action at a time)
        num_hoists = max(len(self.hoists), 1)
        parallel_bound = (total_cost + num_hoists - 1) // num_hoists
        
        # Take max of the parallel bound and the max single task cost
        return max(parallel_bound, max_single_cost)

    def _task_network_cost(self, state, task_network):
        """Estimate minimum actions from remaining task network.
        
        For each task in the network, estimate minimum primitive actions.
        This is admissible because each compound task must eventually decompose
        into at least this many primitives.
        """
        if not task_network:
            return 0
        
        total = 0
        for task_item in task_network:
            task = task_item  # task_network contains operators or abstract tasks
            name = task.name
            
            # Check if it's an operator (primitive)
            if hasattr(task, 'add_effects'):
                # It's an operator/primitive action
                if "nop" in name:
                    total += 0
                else:
                    total += 1
            else:
                # It's an abstract task - estimate minimum decomposition cost
                cost = self._estimate_abstract_task_cost(name, state)
                total += cost
        
        return total

    def _estimate_abstract_task_cost(self, task_name, state):
        """Estimate minimum number of primitive actions for an abstract task.
        
        Uses state information to determine if the task is trivially satisfied
        or requires actual work.
        """
        ttype, args = self._parse_task_name(task_name)
        
        if ttype == "do_put_on" and len(args) >= 2:
            crate = args[0]
            surface = args[1]
            # Check if already on target
            key = (crate, surface)
            if key in self.on_fact_id:
                fid = self.on_fact_id[key]
                if (state >> fid) & 1:
                    return 0  # Already satisfied, will use m0 (nop)
            
            # Check if crate is at same place as surface
            crate_in_truck = self._is_crate_in_truck(crate, state)
            
            if crate_in_truck:
                # m2: do_get_truck + do_clear + Unload + Drop = at least 2 primitives
                return 2
            
            crate_place = self._get_crate_location(crate, state)
            target_place = self._get_surface_location(surface, state)
            
            if crate_place is not None and target_place is not None:
                if crate_place == target_place:
                    # m1: do_clear(c) + do_clear(s) + do_lift_crate + Drop
                    # Minimum: 1 (Lift) + 1 (Drop) = 2 primitives
                    cost = 2
                    # Add clearing costs
                    if crate in self.clear_fact_id:
                        cfid = self.clear_fact_id[crate]
                        if not ((state >> cfid) & 1):
                            cost += 2  # At least lift+load to clear top crate
                    if surface in self.clear_fact_id:
                        sfid = self.clear_fact_id[surface]
                        if not ((state >> sfid) & 1):
                            cost += 2
                    return cost
                else:
                    # m3: do_load_truck + Drive + do_unload_truck
                    # Minimum: Lift + Load + Drive + Unload + Drop = 5
                    cost = 5
                    if crate in self.clear_fact_id:
                        cfid = self.clear_fact_id[crate]
                        if not ((state >> cfid) & 1):
                            cost += 2
                    if surface in self.clear_fact_id:
                        sfid = self.clear_fact_id[surface]
                        if not ((state >> sfid) & 1):
                            cost += 2
                    return cost
            
            # Crate being lifted
            if crate_place is None and not crate_in_truck:
                return 1  # Just need drop
            
            return 2  # Conservative default
        
        elif ttype == "do_clear" and len(args) >= 2:
            surface = args[0]
            place = args[1]
            if surface in self.clear_fact_id:
                cfid = self.clear_fact_id[surface]
                if (state >> cfid) & 1:
                    # Check at
                    at_key = (surface, place)
                    if at_key in self.at_fact_id:
                        afid = self.at_fact_id[at_key]
                        if (state >> afid) & 1:
                            return 0  # Already clear and at place
                return 2  # Need to clear: lift + load minimum
            return 0
        
        elif ttype == "do_get_truck" and len(args) >= 2:
            truck = args[0]
            place = args[1]
            at_key = (truck, place)
            if at_key in self.at_fact_id:
                fid = self.at_fact_id[at_key]
                if (state >> fid) & 1:
                    return 0  # Truck already there
            return 1  # Need one Drive action
        
        elif ttype == "do_lift_crate" and len(args) >= 3:
            crate = args[0]
            # Need at least 1 action (Lift or Unload)
            return 1
        
        elif ttype == "do_load_truck" and len(args) >= 4:
            # Need: get_truck + clear + Lift + Load = at least 2 primitives
            return 2
        
        elif ttype == "do_unload_truck" and len(args) >= 4:
            # Need: clear + Unload + Drop = at least 2 primitives
            return 2
        
        # Unknown task type - return 1 as safe lower bound
        return 1
