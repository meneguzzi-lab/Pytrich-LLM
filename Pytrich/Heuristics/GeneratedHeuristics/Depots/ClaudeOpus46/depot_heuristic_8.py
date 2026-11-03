"""Depots Domain Heuristic

Bottleneck: Crates are stacked and must be cleared before access. Moving a crate
between places requires load/drive/unload (min 3 actions for transport, plus
clearing work). The dominant cost is stack-clearing at source and destination.

Admissible lower bounds:
1. Goal-displacement cost: For each goal (on c s) not satisfied, estimate minimum
   actions needed. If crate is at wrong place, need at least 4 actions (lift, load,
   drive, unload+drop). If at right place but wrong surface, need at least 2 (lift+drop).
   This is admissible because these are unavoidable actions.

2. Stack-clearing cost: For each goal crate that needs to be moved, count crates
   above it that must be cleared first. Each clearing needs at least 2 actions
   (lift + load). Admissible because each crate above truly must be moved.

3. Task network lower bound: Count remaining non-nop primitive actions in the
   task network decomposition. Each abstract task requires at minimum 1 action.
   Admissible as a lower bound on remaining plan length.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic8(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_8"):
        super().__init__(name)
        # Fact index maps built during initialize
        self.on_fact_ids = {}      # (crate, surface) -> global_id
        self.at_fact_ids = {}      # (locatable, place) -> global_id
        self.in_fact_ids = {}      # (crate, truck) -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.lifting_fact_ids = {} # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)
        
        # Object location maps
        self.crate_places = {}     # crate -> set of places it could be at (fact ids)
        self.surface_places = {}   # surface -> set of places
        
        # Stack structure: which crate is on which surface
        self.on_crate_surface = {} # crate -> list of (surface, fact_id)
        self.on_surface_crate = {} # surface -> list of (crate, fact_id) - what's on this surface
        
        # All crates, surfaces, places, trucks, hoists
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        
        # For fast at-place lookup
        self.crate_at_facts = {}   # crate -> dict of place -> fact_global_id
        self.surface_at_facts = {} # surface -> dict of place -> fact_global_id
        
        # Operator type detection
        self.nop_operators = set()  # global_ids of nop operators
        
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
        
        # Parse all facts and build index maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                # Track crate -> surface relationships
                crate, surface = args[0], args[1]
                self.crates.add(crate)
                self.surfaces.add(crate)  # crates are surfaces too
                self.surfaces.add(surface)
                if crate not in self.on_crate_surface:
                    self.on_crate_surface[crate] = []
                self.on_crate_surface[crate].append((surface, f.global_id))
                if surface not in self.on_surface_crate:
                    self.on_surface_crate[surface] = []
                self.on_surface_crate[surface].append((crate, f.global_id))
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                obj, place = args[0], args[1]
                self.places.add(place)
                # Build per-object at-place maps
                if obj.startswith("crate"):
                    self.crates.add(obj)
                    if obj not in self.crate_at_facts:
                        self.crate_at_facts[obj] = {}
                    self.crate_at_facts[obj][place] = f.global_id
                elif obj.startswith("pallet") or obj.startswith("crate"):
                    if obj not in self.surface_at_facts:
                        self.surface_at_facts[obj] = {}
                    self.surface_at_facts[obj][place] = f.global_id
                elif obj.startswith("truck"):
                    self.trucks.add(obj)
                elif obj.startswith("hoist"):
                    self.hoists.add(obj)
                # Also track surfaces
                if obj.startswith("pallet"):
                    self.surfaces.add(obj)
                    if obj not in self.surface_at_facts:
                        self.surface_at_facts[obj] = {}
                    self.surface_at_facts[obj][place] = f.global_id
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = f.global_id
        
        # Parse goal state
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate, surface = args[0], args[1]
                fid = self.on_fact_ids.get((crate, surface), -1)
                self.goal_on_pairs.append((crate, surface, fid))
        
        # Identify nop operators
        for op in model.operators:
            if "nop" in op.name.lower():
                self.nop_operators.add(op.global_id)
        
        # Precompute: for each surface, which place is it statically at?
        # Pallets are static - they don't move. So find their place.
        self.surface_static_place = {}
        for (obj, place), fid in self.at_fact_ids.items():
            if obj.startswith("pallet"):
                self.surface_static_place[obj] = place
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)
    
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
    
    def _compute(self, state, task_network):
        h1 = self._goal_displacement_cost(state)
        h2 = self._task_network_cost(state, task_network)
        return max(h1, h2)
    
    def _find_crate_place(self, crate, state):
        """Find which place a crate is currently at, or None if in a truck."""
        if crate in self.crate_at_facts:
            for place, fid in self.crate_at_facts[crate].items():
                if (state >> fid) & 1:
                    return place
        return None
    
    def _find_surface_place(self, surface, state):
        """Find which place a surface is at."""
        # Pallets have static places
        if surface in self.surface_static_place:
            return self.surface_static_place[surface]
        # Crates can move
        if surface in self.crate_at_facts:
            for place, fid in self.crate_at_facts[surface].items():
                if (state >> fid) & 1:
                    return place
        # Check surface_at_facts
        if surface in self.surface_at_facts:
            for place, fid in self.surface_at_facts[surface].items():
                if (state >> fid) & 1:
                    return place
        return None
    
    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for (c, t), fid in self.in_fact_ids.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False
    
    def _count_crates_above(self, surface, state):
        """Count how many crates are stacked above a surface (recursively)."""
        count = 0
        current = surface
        # Check if something is on current
        if current in self.clear_fact_ids:
            clear_fid = self.clear_fact_ids[current]
            if (state >> clear_fid) & 1:
                return 0  # Already clear
        
        # Find what's on current
        if current in self.on_surface_crate:
            for crate, fid in self.on_surface_crate[current]:
                if (state >> fid) & 1:
                    count = 1 + self._count_crates_above(crate, state)
                    return count
        return 0
    
    def _goal_displacement_cost(self, state):
        """Compute admissible lower bound based on unsatisfied goal on-predicates.
        
        For each goal (on crate surface) not currently true:
        - If crate is being lifted: need at least 1 action (drop)
        - If crate is in truck: need at least 2 actions (unload + drop) 
        - If crate is at same place as target surface: need at least 2 (lift + drop)
          plus clearing costs
        - If crate is at different place: need at least 4 (lift + load + drive + unload + drop = 5,
          but lift+load might be combined, so conservatively 4)
        
        Additionally, if target surface is not clear, need to clear it first (2 actions per crate above).
        """
        total_cost = 0
        
        for crate, surface, fid in self.goal_on_pairs:
            if fid >= 0 and (state >> fid) & 1:
                continue  # Goal already satisfied
            
            cost = 0
            
            # Check if crate is being lifted by some hoist
            being_lifted = False
            for (h, c), lid in self.lifting_fact_ids.items():
                if c == crate and (state >> lid) & 1:
                    being_lifted = True
                    break
            
            if being_lifted:
                # Need at least drop (1 action), maybe more if wrong place
                cost = 1
            elif self._is_crate_in_truck(crate, state):
                # Need unload + drop = 2 minimum (maybe drive too)
                cost = 2
            else:
                crate_place = self._find_crate_place(crate, state)
                surface_place = self._find_surface_place(surface, state)
                
                if crate_place is not None and surface_place is not None:
                    if crate_place == surface_place:
                        # Same place: need lift + drop = 2
                        cost = 2
                    else:
                        # Different place: lift + load + drive + unload + drop = 5
                        cost = 5
                elif crate_place is not None:
                    # Surface place unknown (might be in truck too), estimate 4
                    cost = 4
                else:
                    # Crate location unknown, conservative estimate
                    cost = 2
            
            # Add clearing cost for the crate itself (crates above it must be moved)
            if not being_lifted and not self._is_crate_in_truck(crate, state):
                above_crate = self._count_crates_above_limited(crate, state, 5)
                # Each crate above needs at least 2 actions (lift + load/drop)
                cost += above_crate * 2
            
            # Add clearing cost for target surface
            if surface in self.clear_fact_ids:
                clear_fid = self.clear_fact_ids[surface]
                if not ((state >> clear_fid) & 1):
                    above_surface = self._count_crates_above_limited(surface, state, 5)
                    cost += above_surface * 2
            
            total_cost += cost
        
        return total_cost
    
    def _count_crates_above_limited(self, surface, state, limit):
        """Count crates above surface, with a limit to avoid deep recursion."""
        count = 0
        current = surface
        for _ in range(limit):
            if current in self.clear_fact_ids:
                clear_fid = self.clear_fact_ids[current]
                if (state >> clear_fid) & 1:
                    return count
            
            found = False
            if current in self.on_surface_crate:
                for crate, fid in self.on_surface_crate[current]:
                    if (state >> fid) & 1:
                        count += 1
                        current = crate
                        found = True
                        break
            if not found:
                return count
        return count
    
    def _task_network_cost(self, state, task_network):
        """Lower bound from task network: estimate minimum actions for each task.
        
        For primitive tasks (operators): cost 1 (except nop = 0)
        For abstract tasks: estimate based on task type and state.
        """
        if not task_network:
            return 0
        
        total = 0
        # Limit iteration to avoid timeout on huge networks
        count = 0
        max_tasks = 2000  # Safety limit
        
        for task in task_network:
            if count >= max_tasks:
                # For remaining tasks, add 1 each as conservative estimate
                total += len(task_network) - count
                break
            count += 1
            
            gid = task.global_id
            
            # Check if it's an operator
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                if gid not in self.nop_operators:
                    total += 1
            # Check if it's an abstract task
            elif gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                total += self._estimate_abstract_task_cost(task, state)
        
        return total
    
    def _estimate_abstract_task_cost(self, task, state):
        """Estimate minimum cost to complete an abstract task.
        
        Uses task name to determine type and state to estimate cost.
        """
        name = task.name
        
        # Parse task name to get type
        # Format like: __do_put_on[crate0,pallet1] or similar
        if "do_put_on" in name:
            return self._estimate_do_put_on(name, state)
        elif "do_clear" in name:
            return self._estimate_do_clear(name, state)
        elif "do_get_truck" in name:
            return self._estimate_do_get_truck(name, state)
        elif "do_lift_crate" in name:
            return self._estimate_do_lift_crate(name, state)
        elif "do_load_truck" in name:
            return self._estimate_do_load_truck(name, state)
        elif "do_unload_truck" in name:
            return self._estimate_do_unload_truck(name, state)
        else:
            return 1  # Unknown task, at least 1 action
    
    def _parse_task_args(self, name):
        """Extract arguments from a task name like __do_put_on[crate0,pallet1]."""
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            args_str = name[bracket_idx+1:-1]
            return args_str.split(",") if args_str else []
        return []
    
    def _estimate_do_put_on(self, name, state):
        """Estimate cost of do_put_on(crate, surface).
        
        Methods:
        - m0: already on target -> 0 (nop)
        - m1: crate at same place -> clear crate + clear surface + lift + drop >= 2
        - m2: crate in truck -> get_truck + clear surface + unload + drop >= 2  
        - m3: general -> load + drive + unload >= 5
        """
        args = self._parse_task_args(name)
        if len(args) < 2:
            return 1
        
        crate, surface = args[0], args[1]
        
        # Check if already on target (m0)
        on_key = (crate, surface)
        if on_key in self.on_fact_ids:
            fid = self.on_fact_ids[on_key]
            if (state >> fid) & 1:
                return 0
        
        # Check if crate is in a truck (m2 path)
        if self._is_crate_in_truck(crate, state):
            # get_truck (0 or 1) + clear_surface + unload + drop = min 2
            cost = 2
            # Add clearing cost for surface
            if surface in self.clear_fact_ids:
                sfid = self.clear_fact_ids[surface]
                if not ((state >> sfid) & 1):
                    cost += 1 + self._count_crates_above_limited(surface, state, 3) * 2
            return cost
        
        # Check if crate is at a place
        crate_place = self._find_crate_place(crate, state)
        surface_place = self._find_surface_place(surface, state)
        
        if crate_place is not None and surface_place is not None:
            if crate_place == surface_place:
                # m1 path: clear crate + clear surface + lift + drop
                cost = 2  # lift + drop minimum
                # Clearing crate
                if crate in self.clear_fact_ids:
                    cfid = self.clear_fact_ids[crate]
                    if not ((state >> cfid) & 1):
                        cost += 1 + self._count_crates_above_limited(crate, state, 3) * 2
                # Clearing surface
                if surface in self.clear_fact_ids:
                    sfid = self.clear_fact_ids[surface]
                    if not ((state >> sfid) & 1):
                        cost += 1 + self._count_crates_above_limited(surface, state, 3) * 2
                return cost
            else:
                # m3 path: load_truck + drive + unload_truck
                # load_truck = get_truck + clear_crate + lift + load = min 2
                # drive = 1
                # unload_truck = clear_surface + unload + drop = min 2
                cost = 5
                # Add clearing costs
                if crate in self.clear_fact_ids:
                    cfid = self.clear_fact_ids[crate]
                    if not ((state >> cfid) & 1):
                        cost += self._count_crates_above_limited(crate, state, 3) * 2
                if surface in self.clear_fact_ids:
                    sfid = self.clear_fact_ids[surface]
                    if not ((state >> sfid) & 1):
                        cost += self._count_crates_above_limited(surface, state, 3) * 2
                return cost
        
        # Default: assume some transport needed
        return 3
    
    def _estimate_do_clear(self, name, state):
        """Estimate cost of do_clear(surface, place).
        
        m4: already clear -> 0
        m5: not clear -> clear top + lift + get_truck + load >= 3
        """
        args = self._parse_task_args(name)
        if len(args) < 2:
            return 1
        
        surface = args[0]
        
        if surface in self.clear_fact_ids:
            fid = self.clear_fact_ids[surface]
            if (state >> fid) & 1:
                return 0
            # Not clear: need to clear recursively
            above = self._count_crates_above_limited(surface, state, 5)
            # Each crate needs: clear itself (recursive) + lift + get_truck + load = min 3 per crate
            return above * 3
        
        return 0
    
    def _estimate_do_get_truck(self, name, state):
        """Estimate cost of do_get_truck(truck, place).
        
        m6: truck already there -> 0
        m7: need to drive -> 1
        """
        args = self._parse_task_args(name)
        if len(args) < 2:
            return 0
        
        truck, place = args[0], args[1]
        at_key = (truck, place)
        if at_key in self.at_fact_ids:
            fid = self.at_fact_ids[at_key]
            if (state >> fid) & 1:
                return 0
        return 1
    
    def _estimate_do_lift_crate(self, name, state):
        """Estimate cost of do_lift_crate(crate, place, hoist).
        
        m8: crate in truck -> get_truck + unload >= 1
        m9: crate on surface -> lift = 1
        """
        return 1
    
    def _estimate_do_load_truck(self, name, state):
        """do_load_truck: get_truck + clear + lift + load >= 2."""
        args = self._parse_task_args(name)
        if len(args) < 4:
            return 2
        
        crate = args[0]
        cost = 2  # lift + load minimum
        
        if crate in self.clear_fact_ids:
            cfid = self.clear_fact_ids[crate]
            if not ((state >> cfid) & 1):
                cost += self._count_crates_above_limited(crate, state, 3) * 2
        
        return cost
    
    def _estimate_do_unload_truck(self, name, state):
        """do_unload_truck: clear_surface + unload + drop >= 2."""
        args = self._parse_task_args(name)
        if len(args) < 4:
            return 2
        
        surface = args[1]
        cost = 2  # unload + drop minimum
        
        if surface in self.clear_fact_ids:
            sfid = self.clear_fact_ids[surface]
            if not ((state >> sfid) & 1):
                cost += self._count_crates_above_limited(surface, state, 3) * 2
        
        return cost
