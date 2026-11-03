from Pytrich.heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic21(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_16, model):
        super().__init__(model)
        self.barman_gemini3pro_heuristic_16 = barman_gemini3pro_heuristic_16
        self.shakers = []
        self.shots = []
        self.target_goals = []
        self.shaker_props = {}
        self.shot_props = {}
        self._initialize_metadata()

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

    def _initialize_metadata(self):
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if "shaker" in fact.name.lower() and args:
                if args[0] not in self.shakers and "shaker" in args[0].lower():
                    self.shakers.append(args[0])
            if "shot" in fact.name.lower() and args:
                if args[0] not in self.shots and "shot" in args[0].lower():
                    self.shots.append(args[0])
        
        self.shakers = list(set(self.shakers))
        self.shots = list(set(self.shots))

        for s in self.shakers:
            self.shaker_props[s] = {'shaked': -1, 'unshaked': -1, 'clean': -1, 'empty': -1, 'contains': {}}
        for s in self.shots:
            self.shot_props[s] = {'clean': -1, 'empty': -1, 'contains': {}}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or not args:
                continue
            
            fid = fact.global_id
            obj = args[0]

            if obj in self.shaker_props:
                if pred == "shaked":
                    self.shaker_props[obj]['shaked'] = fid
                elif pred == "unshaked":
                    self.shaker_props[obj]['unshaked'] = fid
                elif pred == "clean":
                    self.shaker_props[obj]['clean'] = fid
                elif pred == "empty":
                    self.shaker_props[obj]['empty'] = fid
                elif pred == "contains" and len(args) > 1:
                    self.shaker_props[obj]['contains'][args[1]] = fid
            
            if obj in self.shot_props:
                if pred == "clean":
                    self.shot_props[obj]['clean'] = fid
                elif pred == "empty":
                    self.shot_props[obj]['empty'] = fid
                elif pred == "contains" and len(args) > 1:
                    self.shot_props[obj]['contains'][args[1]] = fid

        temp_goals = self.model.goals
        while temp_goals:
            idx = (temp_goals & -temp_goals).bit_length() - 1
            fact = self.model.facts[idx]
            pred, args = self._parse_fact_name(fact.name)
            if pred == "contains" and len(args) == 2:
                self.target_goals.append((idx, args[0], args[1]))
            temp_goals &= temp_goals - 1

    def compute(self, node):
        state = node.state
        goals = self.model.goals
        
        if (goals & state) == goals:
            return 0
            
        h_val = 0
        
        for gid, shot, cocktail in self.target_goals:
            if (state >> gid) & 1:
                continue

            cost = 14
            
            in_shaker = False
            shaker_ready = False
            
            for shaker in self.shakers:
                props = self.shaker_props.get(shaker)
                if not props:
                    continue
                
                c_map = props['contains']
                cid = c_map.get(cocktail)
                
                if cid is not None and ((state >> cid) & 1):
                    in_shaker = True
                    sid = props.get('shaked', -1)
                    if sid != -1 and ((state >> sid) & 1):
                        shaker_ready = True
                        break
            
            if shaker_ready:
                cost = 2
            elif in_shaker:
                cost = 5
            
            shot_data = self.shot_props.get(shot)
            if shot_data:
                clean_id = shot_data.get('clean', -1)
                is_clean = (clean_id != -1) and ((state >> clean_id) & 1)
                
                if not is_clean:
                    empty_id = shot_data.get('empty', -1)
                    is_empty = (empty_id != -1) and ((state >> empty_id) & 1)
                    if is_empty:
                        cost += 2
                    else:
                        cost += 3
            
            h_val += cost
            
        return h_val
