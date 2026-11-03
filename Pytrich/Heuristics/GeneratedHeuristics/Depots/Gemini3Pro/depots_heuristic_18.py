from Pytrich.Heuristics.heuristic import Heuristic
import sys

class DepotsGemini3ProHeuristic18(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_18=None, model=None):
        super().__init__(model=model)
        self.depots_gemini3pro_heuristic_18 = depots_gemini3pro_heuristic_18
        
        # Precompute lookups for fast state evaluation
        self.fact_name_to_id = {}
        self.at_map = {}      # obj -> [(place, global_id)]
        self.in_map = {}      # crate -> [(truck, global_id)]
        self.on_map = {}      # (crate, surface) -> global_id
        self.clear_map = {}   # surface -> global_id
        self.truck_at_map = {} # truck -> [(place, global_id)]

        if self.model:
            for fact in self.model.facts:
                self.fact_name_to_id[fact.name] = fact.global_id
                pred, args = self._parse_fact_name(fact.name)
                if not pred:
                    continue
                
                if pred == 'at':
                    # at ?x ?y
                    if len(args) == 2:
                        obj, place = args[0], args[1]
                        if obj not in self.at_map:
                            self.at_map[obj] = []
                        self.at_map[obj].append((place, fact.global_id))
                        
                        # Track trucks separately for easier lookup
                        if "truck" in obj:
                            if obj not in self.truck_at_map:
                                self.truck_at_map[obj] = []
                            self.truck_at_map[obj].append((place, fact.global_id))
                        
                elif pred == 'in':
                    # in ?x ?y (crate in truck)
                    if len(args) == 2:
                        crate, truck = args[0], args[1]
                        if crate not in self.in_map:
                            self.in_map[crate] = []
                        self.in_map[crate].append((truck, fact.global_id))
                    
                elif pred == 'on':
                    # on ?x ?y
                    if len(args) == 2:
                        crate, surface = args[0], args[1]
                        self.on_map[(crate, surface)] = fact.global_id
                    
                elif pred == 'clear':
                    # clear ?x
                    if len(args) == 1:
                        surface = args[0]
                        self.clear_map[surface] = fact.global_id

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return None, []

    def _parse_task_args(self, task_name):
        # Handle "task arg1 arg2" or "task[arg1,arg2]"
        if "[" in task_name and "]" in task_name:
            base = task_name.split("[")[0]
            args = task_name.split("[")[1].strip("]").split(",")
            return base, [a.strip() for a in args]
        else:
            parts = task_name.split()
            if not parts:
                return "", []
            return parts[0], parts[1:]

    def _is_true(self, state, global_id):
        return (state >> global_id) & 1

    def _get_location(self, state, obj):
        # Check 'at' predicate directly
        if obj in self.at_map:
            for place, fid in self.at_map[obj]:
                if self._is_true(state, fid):
                    return place
        
        # Check 'in' predicate (crate inside truck)
        if obj in self.in_map:
            for truck, fid in self.in_map[obj]:
                if self._is_true(state, fid):
                    # Recursive lookup for truck location
                    return self._get_location(state, truck)
        return None

    def eval(self, node):
        state = node.state
        tn = node.task_network
        
        h_val = 0
        
        # Analyze Task Network
        for task in tn:
            base_name, args = self._parse_task_args(task.name)
            
            if base_name == "do_put_on":
                # do_put_on ?c ?s2
                if len(args) >= 2:
                    c, s = args[0], args[1]
                    
                    # 1. Check if goal is already achieved
                    on_fid = self.on_map.get((c, s))
                    if on_fid is not None and self._is_true(state, on_fid):
                        continue # Task effectively done
                    
                    # 2. Determine locations
                    loc_c = self._get_location(state, c)
                    loc_s = self._get_location(state, s)
                    
                    # Base cost for any move operation (Lift + Drop)
                    cost = 2
                    
                    # 3. Distance cost
                    if loc_c and loc_s:
                        if loc_c != loc_s:
                            cost += 5 # Load + Drive + Unload penalty
                    elif not loc_c or not loc_s:
                        # Penalty for unknown location (shouldn't happen in valid states)
                        cost += 1
                            
                    # 4. Obstruction cost (Clearance)
                    # Check if crate c is clear (if it's on a surface)
                    if loc_c and c in self.at_map: 
                        # If c is 'at' a place, it's on a surface/pallet. Check clear status.
                        clear_fid_c = self.clear_map.get(c)
                        if clear_fid_c is not None and not self._is_true(state, clear_fid_c):
                            cost += 2 # Penalty for unstacking items above c
                    
                    # Check if target surface s is clear
                    # Note: Pallets are always clear effectively in some models, but crates need checking
                    clear_fid_s = self.clear_map.get(s)
                    if clear_fid_s is not None and not self._is_true(state, clear_fid_s):
                         cost += 2 # Penalty for unstacking items above s
                         
                    h_val += cost

            elif base_name == "do_clear":
                 # do_clear ?s1 ?p1
                 if len(args) >= 1:
                     s = args[0]
                     clear_fid = self.clear_map.get(s)
                     if clear_fid is not None and not self._is_true(state, clear_fid):
                         h_val += 2 # Needs unstacking
            
            elif base_name == "do_get_truck":
                # do_get_truck ?t ?p
                if len(args) >= 2:
                    t, p = args[0], args[1]
                    at_fid = None
                    # Find specific fact id for at(t, p)
                    if t in self.truck_at_map:
                        for place, fid in self.truck_at_map[t]:
                            if place == p:
                                at_fid = fid
                                break
                    
                    if at_fid is not None and not self._is_true(state, at_fid):
                        h_val += 4 # Drive cost
                        
            elif base_name == "do_load_truck" or base_name == "Load":
                h_val += 1
            elif base_name == "do_unload_truck" or base_name == "Unload":
                h_val += 1
            elif base_name == "Drive":
                h_val += 4
            elif base_name == "Lift" or base_name == "do_lift_crate":
                h_val += 1
            elif base_name == "Drop":
                h_val += 1
            else:
                # Small default cost for other tasks to encourage processing
                h_val += 1
        
        return h_val