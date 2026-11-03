from typing import List, Union, Set, Dict, Tuple
from pytrich.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic11(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_11, model):
        super().__init__(model)
        self.model = model
        self.depots_gemini3pro_heuristic_11 = depots_gemini3pro_heuristic_11
        
        # Precompute fact lookups for O(1) access
        self.fact_name_to_id = {}
        self.on_map = {}        # bottom_obj -> list of (top_obj, global_id)
        self.at_map = {}        # obj -> list of (location, global_id)
        self.in_map = {}        # crate -> list of (truck, global_id)
        self.lifting_map = {}   # hoist -> list of (crate, global_id)
        self.reverse_lifting_map = {} # crate -> list of (hoist, global_id)
        self.clear_map = {}     # surface -> global_id
        self.available_map = {} # hoist -> global_id
        
        self._preprocess_facts()
        
        # Precompute task parsing for O(1) access
        self.task_registry = {} # global_id -> (task_name, args_list)
        self._preprocess_tasks()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Handle grounded fact format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess_facts(self):
        for fact in self.model.facts:
            self.fact_name_to_id[fact.name] = fact.global_id
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            fid = fact.global_id
            
            if pred == "on" and len(args) == 2:
                top, bottom = args
                if bottom not in self.on_map: self.on_map[bottom] = []
                self.on_map[bottom].append((top, fid))
                
            elif pred == "at" and len(args) == 2:
                obj, loc = args
                if obj not in self.at_map: self.at_map[obj] = []
                self.at_map[obj].append((loc, fid))
                
            elif pred == "in" and len(args) == 2:
                crate, truck = args
                if crate not in self.in_map: self.in_map[crate] = []
                self.in_map[crate].append((truck, fid))
                
            elif pred == "lifting" and len(args) == 2:
                hoist, crate = args
                if hoist not in self.lifting_map: self.lifting_map[hoist] = []
                self.lifting_map[hoist].append((crate, fid))
                # Add reverse map for efficiently finding location of lifted crate
                if crate not in self.reverse_lifting_map: self.reverse_lifting_map[crate] = []
                self.reverse_lifting_map[crate].append((hoist, fid))
                
            elif pred == "available" and len(args) == 1:
                self.available_map[args[0]] = fid
                
            elif pred == "clear" and len(args) == 1:
                self.clear_map[args[0]] = fid

    def _preprocess_tasks(self):
        def parse_task_str(s):
            s = s.strip()
            # Format: taskname(arg1, arg2)
            if "(" in s and s.endswith(")"):
                idx = s.find("(")
                name = s[:idx]
                args_part = s[idx+1:-1]
                args = [a.strip() for a in args_part.split(",") if a.strip()]
                return name, args
            return s, []

        # Process Abstract Tasks
        for task in self.model.abstract_tasks:
            t_name, t_args = parse_task_str(task.name)
            self.task_registry[task.global_id] = (t_name, t_args)
            
        # Process Primitive Operators
        for op in self.model.operators:
            op_name, op_args = parse_task_str(op.name)
            self.task_registry[op.global_id] = (op_name, op_args)

    def _is_true(self, state, fid):
        return (state >> fid) & 1

    def _get_location(self, state, obj):
        """Recursively find the location (Place) of an object (Crate/Truck/Hoist)."""
        # Check direct 'at' facts
        if obj in self.at_map:
            for loc, fid in self.at_map[obj]:
                if self._is_true(state, fid):
                    return loc
        
        # Check if inside a truck
        if obj in self.in_map:
            for truck, fid in self.in_map[obj]:
                if self._is_true(state, fid):
                    return self._get_location(state, truck)
                    
        # Check if being lifted by a hoist
        if obj in self.reverse_lifting_map:
            for hoist, fid in self.reverse_lifting_map[obj]:
                if self._is_true(state, fid):
                    return self._get_location(state, hoist)
                    
        return None

    def _count_above(self, state, obj):
        """Estimate stacking depth above an object."""
        count = 0
        current = obj
        visited = set()
        
        # Iteratively climb the stack
        while True:
            if current in visited: break 
            visited.add(current)
            found_above = False
            
            if current in self.on_map:
                for top, fid in self.on_map[current]:
                    if self._is_true(state, fid):
                        count += 1
                        current = top
                        found_above = True
                        break
            
            if not found_above:
                break
        return count

    def compute(self, state, task_network):
        h_val = 0
        
        if not task_network:
            return 0
            
        for task in task_network:
            # Primitive Operator Cost
            if hasattr(task, 'cost'): 
                h_val += getattr(task, 'cost', 1)
                continue
                
            # Abstract Task Estimation
            tid = task.global_id
            if tid not in self.task_registry:
                h_val += 1 
                continue
                
            t_name, t_args = self.task_registry[tid]
            
            if t_name == "do_put_on" and len(t_args) == 2:
                crate, target = t_args
                
                # Check if crate is already on target in current state
                satisfied = False
                if target in self.on_map:
                    for top, fid in self.on_map[target]:
                        if top == crate and self._is_true(state, fid):
                            satisfied = True
                            break
                
                if satisfied:
                    h_val += 0
                else:
                    # Resolve locations
                    loc_c = self._get_location(state, crate)
                    loc_target = self._get_location(state, target)
                    
                    # Estimate unstacking costs (Clear C + Clear Target)
                    # Each unstack operation is Lift + Drop (2 steps)
                    cost_clear_c = self._count_above(state, crate) * 2 
                    cost_clear_target = self._count_above(state, target) * 2
                    
                    # Estimate transport/move cost
                    transport_cost = 0
                    if loc_c and loc_target and loc_c != loc_target:
                        # Load(1) + Drive(1) + Unload(1) + drop overhead -> approx 4
                        transport_cost = 4 
                    else:
                        # Lift(1) + Drop(1) -> 2
                        transport_cost = 2 
                        
                    h_val += cost_clear_c + cost_clear_target + transport_cost

            elif t_name == "do_clear" and len(t_args) == 2:
                surf, place = t_args
                h_val += self._count_above(state, surf) * 2
                
            elif t_name == "do_get_truck" and len(t_args) == 2:
                truck, place = t_args
                loc_t = self._get_location(state, truck)
                if loc_t != place:
                    h_val += 1 # Drive cost
                    
            elif t_name == "do_load_truck" and len(t_args) == 4:
                h_val += 2 # Lift + Load
                
            elif t_name == "do_unload_truck" and len(t_args) == 4:
                h_val += 2 # Unload + Drop
                
            elif t_name == "do_lift_crate":
                h_val += 1
                
            else:
                # Default fallback for other abstract tasks
                h_val += 1

        return h_val
