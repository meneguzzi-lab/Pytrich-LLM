from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic0(Heuristic):
    """
    Depots Bottleneck: The serial manipulation of crates (Lift/Drop/Load/Unload) 
    and truck travel between locations.
    
    Lower Bounds:
    1. Manipulation Cost: Every crate not in its stable goal position must at least be dropped. 
       If it is currently at a different location or in a truck, it also requires Lift/Load/Unload.
    2. Transportation Cost: If a truck is carrying a crate or needs to pick up a crate for a 
       different location, it must Drive at least once.
    3. Stacking Constraint: If a crate is not clear, the crates on top must be moved first.
    """

    def __init__(self, depot_gemini3flash_heuristic_0="depot_gemini3flash_heuristic_0"):
        super().__init__(name=depot_gemini3flash_heuristic_0)
        self.obj_to_id = {}
        self.id_to_obj = {}
        self.crates = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()
        self.pallets = set()
        self.at_ids = {} # obj -> place -> gid
        self.on_ids = {} # crate -> surface -> gid
        self.in_ids = {} # crate -> truck -> gid
        self.lifting_ids = {} # hoist -> crate -> gid
        self.clear_ids = {} # surface -> gid
        self.goal_on = {} # crate_id -> surface_id
        self.target_place = {} # crate_id -> place_id
        self.pallet_to_place = {} # pallet_id -> place_id

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
        # 1. Parse objects and map to IDs
        all_objs = set()
        for f in model.facts:
            _, args = self._parse_fact_name(f.name)
            all_objs.update(args)
        
        for i, obj in enumerate(sorted(list(all_objs))):
            self.obj_to_id[obj] = i
            self.id_to_obj[i] = obj

        # 2. Parse facts and build maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            ids = [self.obj_to_id[a] for a in args]
            
            if pred == "at":
                # (at ?locatable - locatable ?place - place)
                loc, pla = ids
                if loc not in self.at_ids: self.at_ids[loc] = {}
                self.at_ids[loc][pla] = f.global_id
                self.places.add(pla)
            elif pred == "on":
                # (on ?crate - crate ?surface - surface)
                cra, sur = ids
                if cra not in self.on_ids: self.on_ids[cra] = {}
                self.on_ids[cra][sur] = f.global_id
                self.crates.add(cra)
            elif pred == "in":
                # (in ?crate - crate ?truck - truck)
                cra, tru = ids
                if cra not in self.in_ids: self.in_ids[cra] = {}
                self.in_ids[cra][tru] = f.global_id
                self.crates.add(cra)
                self.trucks.add(tru)
            elif pred == "lifting":
                # (lifting ?hoist - hoist ?crate - crate)
                hoi, cra = ids
                if hoi not in self.lifting_ids: self.lifting_ids[hoi] = {}
                self.lifting_ids[hoi][cra] = f.global_id
                self.hoists.add(hoi)
            elif pred == "clear":
                # (clear ?surface - surface)
                sur = ids[0]
                self.clear_ids[sur] = f.global_id

        # Distinguish pallets and places from initial state structure
        init_facts = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "at":
                # Identify pallets: they are at a place but are not trucks or hoists
                loc, pla = args
                l_id = self.obj_to_id[loc]
                p_id = self.obj_to_id[pla]
                if l_id not in self.trucks and l_id not in self.hoists and l_id not in self.crates:
                    self.pallets.add(l_id)
                    self.pallet_to_place[l_id] = p_id

        # 3. Parse Goals
        goal_facts = model.state_explicit_repr(model.goals)
        for g_name in goal_facts:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                cra, sur = [self.obj_to_id[a] for a in args]
                self.goal_on[cra] = sur

        # 4. Resolve Target Places
        for c_id in self.crates:
            curr = c_id
            visited = set()
            while curr in self.goal_on and curr not in visited:
                visited.add(curr)
                next_s = self.goal_on[curr]
                if next_s in self.pallets:
                    self.target_place[c_id] = self.pallet_to_place[next_s]
                    break
                curr = next_s

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _is_stable(self, c_id, state, memo):
        if c_id in memo: return memo[c_id]
        if c_id not in self.goal_on:
            memo[c_id] = True # Ignore non-goal crates
            return True
        target_s = self.goal_on[c_id]
        on_id = self.on_ids.get(c_id, {}).get(target_s)
        if on_id is None or not ((state >> on_id) & 1):
            memo[c_id] = False
            return False
        if target_s in self.pallets:
            memo[c_id] = True
            return True
        res = self._is_stable(target_s, state, memo)
        memo[c_id] = res
        return res

    def _compute(self, state):
        h = 0
        memo_stable = {}
        truck_needs_to_move = {t_id: False for t_id in self.trucks}
        
        for c_id in self.crates:
            if c_id not in self.goal_on:
                continue
            
            if not self._is_stable(c_id, state, memo_stable):
                h += 1 # Base cost: Drop
                target_p = self.target_place.get(c_id)
                
                # Manipulation Logic
                is_at_p = -1
                for p_id in self.places:
                    gid = self.at_ids.get(c_id, {}).get(p_id)
                    if gid is not None and (state >> gid) & 1:
                        is_at_p = p_id
                        break
                
                if is_at_p != -1:
                    h += 1 # Lift
                    if target_p is not None and is_at_p != target_p:
                        h += 2 # Load, Unload
                        # Flag truck necessity
                        for t_id in self.trucks:
                            t_at_gid = self.at_ids.get(t_id, {}).get(is_at_p)
                            if t_at_gid is not None and (state >> t_at_gid) & 1:
                                truck_needs_to_move[t_id] = True
                else:
                    is_in_t = -1
                    for t_id in self.trucks:
                        gid = self.in_ids.get(c_id, {}).get(t_id)
                        if gid is not None and (state >> gid) & 1:
                            is_in_t = t_id
                            break
                    
                    if is_in_t != -1:
                        h += 1 # Unload
                        t_at_target = False
                        if target_p is not None:
                            t_at_gid = self.at_ids.get(is_in_t, {}).get(target_p)
                            if t_at_gid is not None and (state >> t_at_gid) & 1:
                                t_at_target = True
                        if not t_at_target:
                            truck_needs_to_move[is_in_t] = True
                    # if lifting, cost is just Drop(1), which is already added
            
            # Stacking Penalty
            clear_gid = self.clear_ids.get(c_id)
            if clear_gid is not None and not ((state >> clear_gid) & 1):
                h += 1 # Something above must be moved

        # Add Drive costs
        for t_id in self.trucks:
            if truck_needs_to_move[t_id]:
                h += 1
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        # Add minor tie-breaker for task network progress
        h_value += 0.001 * len(node.task_network)
        super().update_info(h_value)
        return int(h_value) if h_value > 1000000 else h_value
