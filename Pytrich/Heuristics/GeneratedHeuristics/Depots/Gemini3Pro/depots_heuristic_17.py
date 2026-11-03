from pytrich_planner.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic17(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_17=None):
        super().__init__(model)
        self.at_map = {} 
        self.on_map = {} 
        self.in_map = {} 
        self.clear_map = {} 
        self.truck_at_map = {}
        self._map_facts()

    def _map_facts(self):
        # Build lookup tables for fast state querying using bitwise operations
        for fact in self.model.facts:
            name = fact.name
            fid = fact.global_id
            
            # Fact names are expected in format +pred[arg1,arg2]
            if not name.startswith('+'): continue
            
            content = name[1:]
            if '[' not in content: continue
            
            pred, args_part = content.split('[', 1)
            args_part = args_part.rstrip(']')
            args = args_part.split(',')
            
            if pred == 'at':
                if len(args) == 2:
                    obj, loc = args
                    if 'truck' in obj:
                        if obj not in self.truck_at_map: self.truck_at_map[obj] = {}
                        self.truck_at_map[obj][loc] = fid
                    else:
                        if obj not in self.at_map: self.at_map[obj] = {}
                        self.at_map[obj][loc] = fid
            elif pred == 'on':
                if len(args) == 2:
                    obj, surf = args
                    if obj not in self.on_map: self.on_map[obj] = {}
                    self.on_map[obj][surf] = fid
            elif pred == 'in':
                if len(args) == 2:
                    obj, truck = args
                    if obj not in self.in_map: self.in_map[obj] = {}
                    self.in_map[obj][truck] = fid
            elif pred == 'clear':
                if len(args) == 1:
                    obj = args[0]
                    self.clear_map[obj] = fid

    def get_loc(self, obj, state):
        # Returns (location, is_in_truck_boolean)
        # Check if object is at a specific location
        if obj in self.at_map:
            for loc, fid in self.at_map[obj].items():
                if (state >> fid) & 1:
                    return loc, False 
        
        # Check if object is in a truck
        if obj in self.in_map:
            for truck, fid in self.in_map[obj].items():
                if (state >> fid) & 1:
                    return self.get_truck_loc(truck, state), True 
        return None, False

    def get_truck_loc(self, truck, state):
        if truck in self.truck_at_map:
             for loc, fid in self.truck_at_map[truck].items():
                 if (state >> fid) & 1:
                     return loc
        return None

    def h(self, node, state, goals=None, task_network=None):
        # Domain-dependent heuristic for Depots
        # Estimates cost based on hierarchical task network
        cost = 0
        if task_network is None: return 0
        
        for task in task_network:
            # Primitive tasks (Operators) just add their cost (usually 1)
            if hasattr(task, 'cost'):
                cost += task.cost
                continue
            
            name = task.name
            # Robust parsing of task name and arguments
            parts = name.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
            if not parts: continue
            
            task_type = parts[0]
            args = parts[1:]
            
            if task_type == 'do_put_on' and len(args) >= 2:
                c, s = args[0], args[1]
                
                # 1. Check if goal is already satisfied
                is_done = False
                if c in self.on_map and s in self.on_map[c]:
                     fid = self.on_map[c][s]
                     if (state >> fid) & 1:
                         is_done = True
                
                if not is_done:
                    # 2. Location Analysis
                    loc_c, c_in_truck = self.get_loc(c, state)
                    loc_s, s_in_truck = self.get_loc(s, state)
                    
                    if loc_c is None or loc_s is None:
                        # Penalty for state uncertainty
                        cost += 5
                    else:
                        # 3. Transport Cost Estimation
                        if c_in_truck:
                            # Needs Unload + Drop
                            sub = 2 
                            if loc_c != loc_s:
                                sub += 1 # Needs Drive
                        else:
                            # Needs Lift + Drop
                            sub = 2 
                            if loc_c != loc_s:
                                sub += 3 # Needs Load + Drive + Unload
                        cost += sub
                    
                    # 4. Obstruction Cost (Clearance)
                    # Crate 'c' must be clear to be moved
                    if c in self.clear_map:
                        fid = self.clear_map[c]
                        if not ((state >> fid) & 1):
                            cost += 2 # Penalty: at least Lift+Drop to clear top
                            
                    # Surface 's' must be clear to receive 'c'
                    if s in self.clear_map:
                        fid = self.clear_map[s]
                        if not ((state >> fid) & 1):
                            cost += 2 # Penalty: at least Lift+Drop to clear top

            elif task_type == 'do_clear' and len(args) >= 1:
                s = args[0]
                if s in self.clear_map:
                     fid = self.clear_map[s]
                     if not ((state >> fid) & 1):
                         cost += 2 # Minimum cost to clear something
            
            elif task_type == 'do_get_truck' and len(args) >= 2:
                t, p = args[0], args[1]
                t_loc = self.get_truck_loc(t, state)
                if t_loc != p:
                    cost += 1 # Drive action
            
            elif task_type == 'do_load_truck' or task_type == 'do_unload_truck':
                cost += 1 # Base cost

        return cost