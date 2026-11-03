from typing import List, Tuple, Dict, Optional, Set
from pytrich.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic16(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_16, model=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.depots_gemini3pro_heuristic_16 = depots_gemini3pro_heuristic_16
        self.model = model
        
        self.parsed_goals: List[Tuple[int, str, str]] = []
        self.obj_at_indices: Dict[str, List[Tuple[int, str]]] = {}
        self.obj_in_indices: Dict[str, List[Tuple[int, str]]] = {}
        self.obj_lifting_indices: Dict[str, List[Tuple[int, str]]] = {}
        self.obj_clear_indices: Dict[str, int] = {}
        self.obj_lifting_reverse: Dict[str, List[int]] = {} # crate -> [fact_ids of lifting it]
        
        if self.model:
            # 1. Parse Goal Bits
            # We need to map the bitwise goals back to structured info to calculate distance.
            goal_repr = self.model.state_explicit_repr(self.model.goals)
            name_to_gid = {f.name: f.global_id for f in self.model.facts}
            
            for g_name in goal_repr:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 2:
                    if g_name in name_to_gid:
                        # Store: (global_id, crate, surface)
                        self.parsed_goals.append((name_to_gid[g_name], args[0], args[1]))

            # 2. Build Search Indices for state decoding
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if not pred: continue
                gid = fact.global_id
                
                if pred == "at" and len(args) == 2:
                    obj, place = args
                    if obj not in self.obj_at_indices: self.obj_at_indices[obj] = []
                    self.obj_at_indices[obj].append((gid, place))
                
                elif pred == "in" and len(args) == 2:
                    crate, truck = args
                    if crate not in self.obj_in_indices: self.obj_in_indices[crate] = []
                    self.obj_in_indices[crate].append((gid, truck))
                    
                elif pred == "lifting" and len(args) == 2:
                    hoist, crate = args
                    if crate not in self.obj_lifting_indices: self.obj_lifting_indices[crate] = []
                    self.obj_lifting_indices[crate].append((gid, hoist))
                    
                    if crate not in self.obj_lifting_reverse: self.obj_lifting_reverse[crate] = []
                    self.obj_lifting_reverse[crate].append(gid)
                    
                elif pred == "clear" and len(args) == 1:
                    obj = args[0]
                    self.obj_clear_indices[obj] = gid

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
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

    def get_object_location(self, obj: str, state_bitwise: int) -> Optional[str]:
        # Check 'at' directly (fastest)
        if obj in self.obj_at_indices:
            for gid, place in self.obj_at_indices[obj]:
                if (state_bitwise >> gid) & 1:
                    return place
        
        # Check 'in' truck (recursive)
        if obj in self.obj_in_indices:
            for gid, truck in self.obj_in_indices[obj]:
                if (state_bitwise >> gid) & 1:
                    return self.get_object_location(truck, state_bitwise)
                    
        # Check 'lifting' by hoist (recursive)
        if obj in self.obj_lifting_indices:
            for gid, hoist in self.obj_lifting_indices[obj]:
                if (state_bitwise >> gid) & 1:
                    return self.get_object_location(hoist, state_bitwise)
                    
        return None
        
    def is_lifting(self, obj: str, state_bitwise: int) -> bool:
        if obj in self.obj_lifting_reverse:
            for gid in self.obj_lifting_reverse[obj]:
                if (state_bitwise >> gid) & 1:
                    return True
        return False

    def h(self, node) -> float:
        state = node.state
        
        # Quick exit if goal state reached
        if (self.model.goals & state) == self.model.goals:
            return 0.0
            
        h_val = 0.0
        
        # Additive heuristic approximation based on unachieved subgoals
        for g_id, crate, surface in self.parsed_goals:
            # Check if this specific goal bit is set in current state
            if not ((state >> g_id) & 1):
                # Goal missing penalty
                h_val += 1.0
                
                # Determine locations
                loc_c = self.get_object_location(crate, state)
                loc_s = self.get_object_location(surface, state)
                
                if loc_c and loc_s:
                    # Transport penalty: If in different places, we need trucks/driving
                    if loc_c != loc_s:
                        h_val += 2.0
                    
                    # Unstacking penalty for Crate
                    # If crate is not clear, we must unstack above it.
                    # Exception: If crate is currently being lifted, it's effectively 'clear' for moving.
                    if crate in self.obj_clear_indices:
                        is_clear = (state >> self.obj_clear_indices[crate]) & 1
                        if not is_clear and not self.is_lifting(crate, state):
                            h_val += 1.0
                            
                    # Unstacking penalty for Surface (Target)
                    # If surface is not clear, we must unstack above it.
                    if surface in self.obj_clear_indices:
                        is_clear = (state >> self.obj_clear_indices[surface]) & 1
                        if not is_clear:
                             h_val += 1.0
                             
        return h_val
