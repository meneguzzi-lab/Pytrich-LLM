from pytrich.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic0(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_0=None):
        super().__init__(model)
        self.model = model
        self.facts = model.facts
        self.goals_bitmask = model.goals
        
        self.goal_facts = []
        self.obj_state_lookup = {}
        self.supported_by_lookup = {}
        
        self._parse_goals()
        self._preprocess_facts()

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

    def _parse_goals(self):
        for f in self.model.facts:
            if (self.goals_bitmask >> f.global_id) & 1:
                pred, args = self._parse_fact_name(f.name)
                if pred == "on" and len(args) == 2:
                    self.goal_facts.append({"crate": args[0], "surface": args[1], "gid": f.global_id})

    def _preprocess_facts(self):
        for f in self.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            gid = f.global_id
            
            if pred == "on" and len(args) == 2:
                top, bottom = args[0], args[1]
                if top not in self.obj_state_lookup: self.obj_state_lookup[top] = []
                self.obj_state_lookup[top].append({"type": "on", "target": bottom, "gid": gid})
                if bottom not in self.supported_by_lookup: self.supported_by_lookup[bottom] = []
                self.supported_by_lookup[bottom].append({"top": top, "gid": gid})
                
            elif pred == "at" and len(args) == 2:
                obj, loc = args[0], args[1]
                if obj not in self.obj_state_lookup: self.obj_state_lookup[obj] = []
                self.obj_state_lookup[obj].append({"type": "at", "target": loc, "gid": gid})
                
            elif pred == "in" and len(args) == 2:
                crate, truck = args[0], args[1]
                if crate not in self.obj_state_lookup: self.obj_state_lookup[crate] = []
                self.obj_state_lookup[crate].append({"type": "in", "target": truck, "gid": gid})
                
            elif pred == "lifting" and len(args) == 2:
                hoist, crate = args[0], args[1]
                if crate not in self.obj_state_lookup: self.obj_state_lookup[crate] = []
                self.obj_state_lookup[crate].append({"type": "lifting", "target": hoist, "gid": gid})

    def _get_object_location(self, obj, state, visited=None):
        if visited is None: visited = set()
        if obj in visited: return None, 0, False
        visited.add(obj)
        
        if obj in self.obj_state_lookup:
            for entry in self.obj_state_lookup[obj]:
                if (state >> entry["gid"]) & 1:
                    etype = entry["type"]
                    target = entry["target"]
                    
                    if etype == "at":
                        return target, 0, False
                    elif etype == "in":
                        # In truck -> find truck loc
                        loc, _, _ = self._get_object_location(target, state, visited)
                        return loc, 0, True 
                    elif etype == "on":
                        loc, depth, _ = self._get_object_location(target, state, visited)
                        if loc: return loc, depth + 1, False
                    elif etype == "lifting":
                        # Crate is held by hoist. Find hoist location.
                        loc, _, _ = self._get_object_location(target, state, visited)
                        return loc, 0, True
        return None, 0, False

    def _get_stack_height_above(self, obj, state, visited=None):
        if visited is None: visited = set()
        if obj in visited: return 0
        visited.add(obj)
        
        if obj in self.supported_by_lookup:
            for entry in self.supported_by_lookup[obj]:
                if (state >> entry["gid"]) & 1:
                    return 1 + self._get_stack_height_above(entry["top"], state, visited)
        return 0

    def h(self, node):
        state = node.state
        h_val = 0
        
        for g in self.goal_facts:
            # If goal is met, cost is 0
            if (state >> g["gid"]) & 1:
                continue
                
            crate = g["crate"]
            surface = g["surface"]
            
            loc_c, depth_c, status_c = self._get_object_location(crate, state)
            loc_s, depth_s, _ = self._get_object_location(surface, state)
            
            # 1. Buried Cost (Unstacking above crate)
            # If held or in truck, depth is 0, height above is 0
            if not status_c:
                h_val += self._get_stack_height_above(crate, state) * 2
            
            # 2. Target Clearance Cost (Unstacking above target)
            h_val += self._get_stack_height_above(surface, state) * 2
            
            # 3. Transport Cost
            if loc_c is None or loc_s is None:
                # Missing info penalty
                h_val += 5
            elif loc_c != loc_s:
                # Different locations
                # If held/in_truck: Drive(1) + Unload(1) + Drop(1) = 3 (approx)
                # If on ground: Lift(1) + Load(1) + Drive(1) + Unload(1) + Drop(1) = 5
                if status_c: h_val += 3
                else: h_val += 5
            else:
                # Same location
                # If held: Drop(1) = 1
                # If on ground: Lift(1) + Drop(1) = 2
                if status_c: h_val += 1
                else: h_val += 2
                
        return h_val