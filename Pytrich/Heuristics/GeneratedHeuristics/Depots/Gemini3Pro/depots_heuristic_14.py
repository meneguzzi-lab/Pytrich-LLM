from pytrich_planner.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic14(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_14):
        super().__init__(depots_gemini3pro_heuristic_14)
        self.goal_trackers = []
        self.truck_at_lookup = {}
        self._initialize_trackers()

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

    def _initialize_trackers(self):
        # Registry to quickly lookup fact IDs for objects
        # Structure: obj -> { 'at': [(loc, id)], 'in': [(truck, id)], 'clear': id }
        registry = {}

        def get_entry(obj):
            if obj not in registry:
                registry[obj] = {'at': [], 'in': [], 'clear': None}
            return registry[obj]

        # 1. Parse all facts in the model to build the registry
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            fid = fact.global_id
            
            if pred == 'at':
                # at[obj, loc]
                if len(args) == 2:
                    obj, loc = args
                    get_entry(obj)['at'].append((loc, fid))
                    # Track trucks separately for location lookup
                    if obj not in self.truck_at_lookup:
                        self.truck_at_lookup[obj] = []
                    self.truck_at_lookup[obj].append((loc, fid))

            elif pred == 'in':
                # in[crate, truck]
                if len(args) == 2:
                    crate, truck = args
                    get_entry(crate)['in'].append((truck, fid))

            elif pred == 'clear':
                # clear[surface]
                if len(args) == 1:
                    surf = args[0]
                    get_entry(surf)['clear'] = fid

        # 2. Identify Goal Facts from the bitmask
        goals_mask = self.model.goals
        # Iterate over all facts to check if they are in the goal set
        for fact in self.model.facts:
            if (goals_mask >> fact.global_id) & 1:
                pred, args = self._parse_fact_name(fact.name)
                # We primarily focus on 'on' goals for Depots (stacking crates)
                if pred == 'on' and len(args) == 2:
                    crate, surf = args
                    self.goal_trackers.append({
                        'goal_id': fact.global_id,
                        'crate': crate,
                        'surf': surf,
                        'c_info': registry.get(crate),
                        's_info': registry.get(surf)
                    })

    def compute(self, node):
        state = node.state
        h_val = 0
        
        for tracker in self.goal_trackers:
            # If goal is already achieved in current state, skip cost
            if (state >> tracker['goal_id']) & 1:
                continue
            
            # Base cost: To achieve on(x, y), we need at least Lift(x) + Drop(x) = 2 actions
            # If we are transporting, it will be Lift + Load + Drive + Unload + Drop
            sub_cost = 2
            
            c_info = tracker['c_info']
            s_info = tracker['s_info']
            
            # 1. Obstruction Penalties
            # If Crate to be moved is not clear, we must unstack things above it.
            # We add a fixed penalty since calculating exact depth is expensive.
            if c_info and c_info['clear'] is not None:
                if not ((state >> c_info['clear']) & 1):
                    sub_cost += 2
            
            # If Target Surface is not clear, we must clear it.
            if s_info and s_info['clear'] is not None:
                if not ((state >> s_info['clear']) & 1):
                    sub_cost += 2

            # 2. Determine Locations
            c_loc = None
            c_in_truck = None
            
            # Find Crate Location
            if c_info:
                # Check 'at' facts
                for loc, fid in c_info['at']:
                    if (state >> fid) & 1:
                        c_loc = loc
                        break
                
                # If not 'at', check 'in' (inside a truck)
                if c_loc is None:
                    for truck, fid in c_info['in']:
                        if (state >> fid) & 1:
                            c_in_truck = truck
                            break
            
            # Find Surface Location
            s_loc = None
            if s_info:
                for loc, fid in s_info['at']:
                    if (state >> fid) & 1:
                        s_loc = loc
                        break
            
            # 3. Distance & Transport Costs
            if c_in_truck:
                # Crate is inside a truck.
                # Determine truck location to see if we need to Drive.
                truck_loc = None
                t_lookup = self.truck_at_lookup.get(c_in_truck)
                if t_lookup:
                    for loc, fid in t_lookup:
                        if (state >> fid) & 1:
                            truck_loc = loc
                            break
                
                # If truck is not at the target surface location, add Drive cost.
                if s_loc and truck_loc != s_loc:
                     sub_cost += 1
                
                # Being in a truck effectively replaces the 'Lift' part of the base cost
                # with 'Unload'. Base=2 covers Unload+Drop. 
                # We assume no extra penalty unless driving is needed.
                pass
                
            elif c_loc:
                # Crate is at a specific place.
                if s_loc and c_loc != s_loc:
                    # Different locations: Needs Load + Drive + Unload.
                    # Base cost covers Lift + Drop.
                    # Sequence: Lift -> Load -> Drive -> Unload -> Drop.
                    # Total 5 actions. Base is 2. We add 3.
                    sub_cost += 3
            
            else:
                # Crate is neither 'at' a place nor 'in' a truck.
                # In this domain, this likely means it is currently being 'lifting' by a hoist.
                # If it's already lifted, we save the 'Lift' action cost.
                # Base 2 -> 1 (Just Drop).
                sub_cost -= 1

            h_val += sub_cost

        return h_val