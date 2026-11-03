from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic13(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_13=None):
        super().__init__(model)
        self.model = model
        self.obj_name_to_id = {}
        self.next_obj_id = 0
        # obj_id -> list of (type_code, target_id, global_id)
        # Type Codes: 1=AT, 2=IN, 3=COVERED_BY (target is the covering object)
        self.obj_lookups = {}
        # Goals: list of (crate_id, surface_id, global_id)
        self.goal_list = []
        self._init_metadata()

    def _get_id(self, name):
        if name not in self.obj_name_to_id:
            self.obj_name_to_id[name] = self.next_obj_id
            self.next_obj_id += 1
        return self.obj_name_to_id[name]

    def _parse(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            idx = name.find("[")
            pred = name[:idx]
            args = name[idx+1:-1].split(",") if name[idx+1:-1] else []
            return pred, args
        return None, []

    def _init_metadata(self):
        # 1. Parse Facts to build efficient lookups
        for fact in self.model.facts:
            pred, args = self._parse(fact.name)
            if not pred:
                continue
            gid = fact.global_id
            
            if pred == "at" and len(args) == 2:
                # at(obj, loc)
                oid = self._get_id(args[0])
                lid = self._get_id(args[1])
                if oid not in self.obj_lookups: self.obj_lookups[oid] = []
                self.obj_lookups[oid].append((1, lid, gid))
                
            elif pred == "in" and len(args) == 2:
                # in(crate, truck)
                oid = self._get_id(args[0])
                tid = self._get_id(args[1])
                if oid not in self.obj_lookups: self.obj_lookups[oid] = []
                self.obj_lookups[oid].append((2, tid, gid))
                
            elif pred == "on" and len(args) == 2:
                # on(top, bottom). We map bottom -> top to check for obstructions
                top = self._get_id(args[0])
                bot = self._get_id(args[1])
                if bot not in self.obj_lookups: self.obj_lookups[bot] = []
                self.obj_lookups[bot].append((3, top, gid)) # 3=COVERED_BY

        # 2. Identify Goal Requirements
        # Pre-filter goal facts to avoid parsing during eval
        gmask = self.model.goals
        for fact in self.model.facts:
            if (gmask >> fact.global_id) & 1:
                pred, args = self._parse(fact.name)
                if pred == "on" and len(args) == 2:
                    c = self._get_id(args[0])
                    s = self._get_id(args[1])
                    self.goal_list.append((c, s, fact.global_id))

    def _get_location(self, obj_id, state):
        # Returns place_id or None. Handles recursive locations (crate in truck)
        if obj_id not in self.obj_lookups:
            return None
        for code, target, gid in self.obj_lookups[obj_id]:
            if (state >> gid) & 1:
                if code == 1: # AT
                    return target
                elif code == 2: # IN truck
                    return self._get_location(target, state)
        return None

    def _is_covered(self, obj_id, state):
        # Returns True if object is covered by anything
        if obj_id not in self.obj_lookups:
            return False
        for code, target, gid in self.obj_lookups[obj_id]:
            if code == 3: # COVERED_BY
                if (state >> gid) & 1:
                    return True
        return False

    def eval(self, node):
        state = node.state
        h_val = 0
        
        # Calculate cost based on unsatisfied goals
        for c, s, gid in self.goal_list:
            # Check if goal satisfied (bitwise check)
            if (state >> gid) & 1:
                continue
            
            # Base penalty for unsatisfied goal
            h_val += 10
            
            # Location analysis for distance estimation
            loc_c = self._get_location(c, state)
            loc_s = self._get_location(s, state)
            
            if loc_c is not None and loc_s is not None:
                if loc_c != loc_s:
                    # Different locations: Needs Drive + Load + Unload
                    h_val += 5
                else:
                    # Same location: Needs stacking
                    h_val += 2
            else:
                 # Location unknown (should not happen in valid states), small penalty
                 h_val += 2
                 
            # Obstruction analysis (Blocksworld aspect)
            # If target surface 's' is covered by something else, extra cost
            if self._is_covered(s, state):
                 h_val += 3
            
            # If crate 'c' is covered (buried), extra cost to unstack
            if self._is_covered(c, state):
                 h_val += 3
                 
        return h_val