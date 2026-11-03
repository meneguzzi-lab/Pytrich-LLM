from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic19(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_19=None):
        super().__init__(model)
        self.model = model
        
        # Precompute indices for relevant predicates to speed up state parsing.
        # Storing tuples of (global_id, arg0, arg1) for fast iteration.
        self.on_map = []      # list of (bit_id, crate, surface)
        self.at_map = []      # list of (bit_id, object, place)
        self.in_map = []      # list of (bit_id, crate, truck)
        self.lifting_map = [] # list of (bit_id, hoist, crate)
        
        # Iterate over all facts in the model to populate maps based on predicate names
        for i, fact in enumerate(self.model.facts):
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on' and len(args) == 2:
                # args: [crate, surface]
                self.on_map.append((fact.global_id, args[0], args[1]))
            elif pred == 'at' and len(args) == 2:
                # args: [locatable, place]
                self.at_map.append((fact.global_id, args[0], args[1]))
            elif pred == 'in' and len(args) == 2:
                # args: [crate, truck]
                self.in_map.append((fact.global_id, args[0], args[1]))
            elif pred == 'lifting' and len(args) == 2:
                # args: [hoist, crate]
                self.lifting_map.append((fact.global_id, args[0], args[1]))
                
        # Parse goals from the model
        # model.goals is a bitwise integer representing the goal state.
        self.goal_requirements = []
        if self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_names:
                pred, args = self._parse_fact_name(name)
                if pred == 'on' and len(args) == 2:
                    self.goal_requirements.append((args[0], args[1]))

    def _parse_fact_name(self, name: str):
        """Parses a grounded fact string like +on[crate1,pallet1] into predicate and args."""
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

    def h(self, node, state, task_network):
        """
        Calculates the heuristic value based on the cost to achieve remaining goals.
        Uses a relaxed estimation considering clearing stacks and moving trucks.
        """
        # 1. Reconstruct minimal necessary state representation using dictionaries for O(1) lookups
        parent_of = {} # crate -> surface (what is crate on?)
        child_of = {}  # surface -> crate (what is on surface?)
        loc_of = {}    # thing -> place (where is thing?)
        inside = {}    # crate -> truck (is crate inside truck?)
        held_by = {}   # crate -> hoist (is crate being lifted?)

        # Scan state bitmask using precomputed maps for efficiency
        # Checking (state >> bit_id) & 1 is faster than string lookups
        
        # Process ON facts
        for bit_id, c, s in self.on_map:
            if (state >> bit_id) & 1:
                parent_of[c] = s
                child_of[s] = c
        
        # Process AT facts
        for bit_id, obj, loc in self.at_map:
            if (state >> bit_id) & 1:
                loc_of[obj] = loc
        
        # Process IN facts
        for bit_id, c, t in self.in_map:
            if (state >> bit_id) & 1:
                inside[c] = t
                
        # Process LIFTING facts
        for bit_id, h, c in self.lifting_map:
            if (state >> bit_id) & 1:
                held_by[c] = h

        # 2. Helper functions for state analysis
        
        def get_location(obj):
            """Returns the place where an object is located, traversing hierarchy if needed."""
            # Case 1: Directly at a place (most common for pallets, parked trucks, bottom crates)
            if obj in loc_of:
                return loc_of[obj]
            # Case 2: Inside a truck
            if obj in inside:
                truck = inside[obj]
                return loc_of.get(truck)
            # Case 3: Being lifted by a hoist
            if obj in held_by:
                hoist = held_by[obj]
                return loc_of.get(hoist)
            # Case 4: On another surface (recursive look-up)
            if obj in parent_of:
                return get_location(parent_of[obj])
            return None

        def get_stack_depth(obj):
            """Returns number of items stacked above obj."""
            depth = 0
            curr = obj
            # Traverse up using child_of (surface -> crate)
            while curr in child_of:
                curr = child_of[curr]
                depth += 1
                if depth > 50: # Safety break to prevent infinite loops in invalid cyclic states
                    break
            return depth

        # 3. Calculate Heuristic Cost
        total_cost = 0
        
        for c_goal, s_goal in self.goal_requirements:
            # Check if goal is already satisfied in current state
            if parent_of.get(c_goal) == s_goal:
                continue
                
            # Base cost for the 'Put On' action sequence (approx 1 unit)
            total_cost += 1
            
            # --- Clearance Costs ---
            # To move C, we must clear everything above it
            total_cost += get_stack_depth(c_goal)
            
            # To put on S, we must clear everything above S
            total_cost += get_stack_depth(s_goal)
            
            # --- Accessibility Costs ---
            # If C is in a truck, we must unload it first
            if c_goal in inside:
                total_cost += 2 # Unload cost approximation
                
            # If S is in a truck (implies S must be unloaded to be a surface)
            if s_goal in inside:
                total_cost += 2
                
            # --- Location Costs ---
            loc_c = get_location(c_goal)
            loc_s = get_location(s_goal)
            
            if loc_c and loc_s:
                if loc_c != loc_s:
                    # Different locations: Requires Truck Load + Drive + Unload sequence
                    total_cost += 5 
            else:
                # Unknown location (e.g., partial state or error), penalize slightly
                total_cost += 1
                
        # 4. Task Network Penalty
        # Add a small tie-breaking penalty for the size of the remaining task network
        # to prefer states that are closer to having an empty task network (solved).
        if task_network:
            total_cost += len(task_network) * 0.1
            
        return total_cost
