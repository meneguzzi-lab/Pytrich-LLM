from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic3(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_3):
        super().__init__(depots_gemini3pro_heuristic_3)
        self.model = depots_gemini3pro_heuristic_3
        
        # Precompute lookups for fast state query
        self.fact_name_to_id = {}
        self.at_indices = {}      # obj -> [(loc, global_id)]
        self.in_indices = {}      # crate -> [(truck, global_id)]
        self.on_indices = {}      # surface -> [(crate_on_top, global_id)]
        self.is_on_indices = {}   # crate -> [(surface, global_id)] 
        self.clear_indices = {}   # surface -> global_id
        
        # We parse all facts from the model initialization
        for fact in self.model.facts:
            self.fact_name_to_id[fact.name] = fact.global_id
            
            # Parse fact name: +predicate[arg1,arg2] or -predicate
            # We strip + and parse content inside []
            name = fact.name.strip()
            if name.startswith("+"):
                name = name[1:]
            
            # Basic parsing logic for predicate[arg1,arg2]
            predicate = None
            args = []
            
            if "[" in name and name.endswith("]"):
                idx = name.find("[")
                predicate = name[:idx]
                args_content = name[idx+1:-1]
                if args_content:
                    args = args_content.split(",")
            
            if predicate and args:
                if predicate == "at" and len(args) == 2:
                    obj, loc = args
                    if obj not in self.at_indices: self.at_indices[obj] = []
                    self.at_indices[obj].append((loc, fact.global_id))
                    
                elif predicate == "in" and len(args) == 2:
                    crate, truck = args
                    if crate not in self.in_indices: self.in_indices[crate] = []
                    self.in_indices[crate].append((truck, fact.global_id))
                    
                elif predicate == "on" and len(args) == 2:
                    crate, surface = args
                    # Index what is on a surface
                    if surface not in self.on_indices: self.on_indices[surface] = []
                    self.on_indices[surface].append((crate, fact.global_id))
                    # Index what a crate is on
                    if crate not in self.is_on_indices: self.is_on_indices[crate] = []
                    self.is_on_indices[crate].append((surface, fact.global_id))
                    
                elif predicate == "clear" and len(args) == 1:
                    surf = args[0]
                    self.clear_indices[surf] = fact.global_id

    def _get_location(self, obj, state):
        """Recursively find the location (Place) of an object."""
        # 1. Check direct 'at' (e.g., trucks, pallets, sometimes loose crates)
        if obj in self.at_indices:
            for loc, fid in self.at_indices[obj]:
                if (state >> fid) & 1:
                    return loc
        
        # 2. Check 'in' truck (obj is inside truck, truck is at location)
        if obj in self.in_indices:
            for truck, fid in self.in_indices[obj]:
                if (state >> fid) & 1:
                    return self._get_location(truck, state)
        
        # 3. Check 'on' surface (obj is on crate/pallet)
        if obj in self.is_on_indices:
            for surf, fid in self.is_on_indices[obj]:
                if (state >> fid) & 1:
                    return self._get_location(surf, state)
                    
        return None

    def _is_clear(self, surface, state):
        """Check if a surface is clear."""
        if surface in self.clear_indices:
            fid = self.clear_indices[surface]
            return (state >> fid) & 1
        # If no clear predicate tracks it, assume true (safe default for non-stackable or irrelevant objects)
        return True

    def _is_on(self, crate, surface, state):
        """Check if crate is directly on surface."""
        if crate in self.is_on_indices:
            for s, fid in self.is_on_indices[crate]:
                if s == surface and ((state >> fid) & 1):
                    return True
        return False

    def _parse_task(self, task_name):
        """Parse task name and arguments from grounded string."""
        name = task_name.strip()
        args = []
        base_name = name
        
        # Try bracket format: task[arg1,arg2]
        if "[" in name and name.endswith("]"):
            idx = name.find("[")
            base_name = name[:idx]
            args_str = name[idx+1:-1]
            if args_str:
                args = args_str.split(",")
        # Try space format: task arg1 arg2
        elif " " in name:
            parts = name.split()
            base_name = parts[0]
            args = parts[1:]
        
        return base_name, args

    def h(self, node):
        """
        Heuristic estimate for Depots domain.
        Aggregates cost of primitive actions and estimates cost for 'do_put_on' tasks.
        """
        state = node.state
        tn = node.task_network
        
        if not tn:
            return 0
            
        total_cost = 0
        
        for task in tn:
            t_name, args = self._parse_task(task.name)
            
            if t_name == "do_put_on" and len(args) >= 2:
                crate = args[0]
                surface = args[1]
                
                # If the goal (on crate surface) is already met, cost is 0
                if self._is_on(crate, surface, state):
                    continue
                
                # Determine locations
                loc_c = self._get_location(crate, state)
                loc_s = self._get_location(surface, state)
                
                # If locations are unknown (partial state or parsing issue), apply a penalty
                if not loc_c or not loc_s:
                    total_cost += 5
                    continue
                
                step_cost = 0
                
                # Transport Cost
                if loc_c == loc_s:
                    # Same location: Lift + Drop = 2
                    step_cost = 2
                else:
                    # Different location: Lift + Load + Drive + Unload + Drop = 5
                    step_cost = 5
                    
                # Setup/Clearance Cost
                # If crate or surface are not clear, we must move at least one item (Lift+Drop=2)
                if not self._is_clear(crate, state):
                    step_cost += 2
                if not self._is_clear(surface, state):
                    step_cost += 2
                
                total_cost += step_cost
                
            # Subtasks and Primitives Cost Estimation
            elif t_name == "do_clear":
                total_cost += 2 # Assume minimal Lift+Drop to clear
            elif t_name == "do_get_truck":
                total_cost += 1 # Assume Drive
            elif t_name == "do_load_truck":
                total_cost += 2 # Lift + Load
            elif t_name == "do_unload_truck":
                total_cost += 2 # Unload + Drop
            elif t_name == "do_lift_crate":
                total_cost += 1 # Lift
                
            # Primitive Actions
            elif t_name == "Drive":
                total_cost += 1
            elif t_name == "Lift":
                total_cost += 1
            elif t_name == "Drop":
                total_cost += 1
            elif t_name == "Load":
                total_cost += 1
            elif t_name == "Unload":
                total_cost += 1
            elif t_name == "nop":
                total_cost += 0
            else:
                # Default small cost for unknown tasks to encourage processing
                total_cost += 1
                
        return total_cost
