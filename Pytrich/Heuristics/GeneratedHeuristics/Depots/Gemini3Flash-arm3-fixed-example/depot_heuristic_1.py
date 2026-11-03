from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class DepotsGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    Bottleneck Analysis:
    The primary bottleneck in Depots is the serial nature of hoist and truck operations. 
    A hoist can only handle one crate at a time, and stacking order constraints (BlocksWorld sub-problem) 
    combined with multi-location logistics (Logistics sub-problem) mean that crates must be moved 
    multiple times if they are in the wrong stack or location.

    Admissible Lower Bounds:
    1. Unstable Crate Cost (h_stability): Any crate not in its goal stack position 
       (either on the wrong surface or on a crate that itself must move) must be moved.
       - At minimum, a move requires a Lift and a Drop (2 actions).
       - If the current location and goal location differ, it must also be Loaded and Unloaded (2 more actions).
    2. Obstruction Clearance: If a goal surface is currently occupied by a crate that doesn't belong there, 
       that crate must be moved (at least 2 actions), which is often captured by the stability check.

    This heuristic combines these into a state-aware estimation of the remaining actions required.
    """

    def __init__(self, depot_gemini3flash_heuristic_1=None, name="DepotsGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.goal_on = {} # crate_id -> surface_id
        self.pallet_loc = {} # pallet_id -> place_id
        self.crate_at_facts = {} # crate_id -> [ (fact_id, place_id) ]
        self.crate_on_facts = {} # crate_id -> [ (fact_id, surface_id) ]
        self.crate_in_facts = {} # crate_id -> [ (fact_id, truck_id) ]
        self.truck_at_facts = {} # truck_id -> [ (fact_id, place_id) ]
        self.hoist_at = {} # hoist_id -> place_id
        self.is_pallet = set()
        self.crate_ids = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse Goal State
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                pred, args = self._parse_fact_name(goal_name)
                if pred == "on":
                    # Goal: crate args[0] on surface args[1]
                    cid = self._get_obj_id(args[0])
                    sid = self._get_obj_id(args[1])
                    if cid is not None and sid is not None:
                        self.goal_on[cid] = sid

        # 2. Pre-map all objects and facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            
            if pred == "at":
                locatable_id = self._get_obj_id(args[0])
                place_id = self._get_obj_id(args[1])
                if "pallet" in args[0]:
                    self.pallet_loc[locatable_id] = place_id
                    self.is_pallet.add(locatable_id)
                elif "truck" in args[0]:
                    if locatable_id not in self.truck_at_facts: self.truck_at_facts[locatable_id] = []
                    self.truck_at_facts[locatable_id].append((f.global_id, place_id))
                elif "crate" in args[0]:
                    self.crate_ids.append(locatable_id)
                    if locatable_id not in self.crate_at_facts: self.crate_at_facts[locatable_id] = []
                    self.crate_at_facts[locatable_id].append((f.global_id, place_id))
                elif "hoist" in args[0]:
                    self.hoist_at[locatable_id] = place_id
            
            elif pred == "on":
                cid = self._get_obj_id(args[0])
                sid = self._get_obj_id(args[1])
                if cid not in self.crate_on_facts: self.crate_on_facts[cid] = []
                self.crate_on_facts[cid].append((f.global_id, sid))
            
            elif pred == "in":
                cid = self._get_obj_id(args[0])
                tid = self._get_obj_id(args[1])
                if cid not in self.crate_in_facts: self.crate_in_facts[cid] = []
                self.crate_in_facts[cid].append((f.global_id, tid))

        self.crate_ids = list(set(self.crate_ids))
        h_initial = self._compute(initial_node.state)
        return super().initialize(model, h_initial)

    def _get_obj_id(self, name):
        # Model.facts objects are not directly indexed, but names are consistent
        return name

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def _get_current_loc(self, crate_id, state):
        # Check 'at' facts
        if crate_id in self.crate_at_facts:
            for fid, pid in self.crate_at_facts[crate_id]:
                if (state >> fid) & 1: return pid
        # Check 'in' facts
        if crate_id in self.crate_in_facts:
            for fid, tid in self.crate_in_facts[crate_id]:
                if (state >> fid) & 1:
                    for tfid, tpid in self.truck_at_facts[tid]:
                        if (state >> tfid) & 1: return tpid
        return None

    def _compute(self, state):
        if not self.goal_on:
            return 0
        
        stable = {} # obj_id -> bool
        current_on = {} # crate_id -> surface_id
        
        for cid in self.crate_ids:
            if cid in self.crate_on_facts:
                for fid, sid in self.crate_on_facts[cid]:
                    if (state >> fid) & 1:
                        current_on[cid] = sid
                        break
        
        def check_stable(obj_id):
            if obj_id in stable: return stable[obj_id]
            if obj_id in self.is_pallet: 
                return True
            # If it's a crate, it's stable if it's on its goal surface AND that surface is stable
            target_sid = self.goal_on.get(obj_id)
            if target_sid is None: 
                # If crate has no goal, we can assume it stays where it is unless it's blocking
                # For admissibility, let's treat no-goal crates as stable if they are on a pallet
                curr_s = current_on.get(obj_id)
                if curr_s is None: return False
                stable[obj_id] = check_stable(curr_s)
                return stable[obj_id]
            
            actual_sid = current_on.get(obj_id)
            if actual_sid == target_sid:
                stable[obj_id] = check_stable(actual_sid)
            else:
                stable[obj_id] = False
            return stable[obj_id]

        h = 0
        for cid in self.goal_on:
            if not check_stable(cid):
                # Crate must move: 2 for Lift/Drop
                h += 2
                # If it's in the wrong place, it needs 2 more for Load/Unload
                goal_s = self.goal_on[cid]
                goal_p = self.pallet_loc.get(goal_s)
                if not goal_p: # Goal is a crate, find its pallet's loc
                    curr = goal_s
                    while curr in self.goal_on:
                        curr = self.goal_on[curr]
                    goal_p = self.pallet_loc.get(curr)
                
                curr_p = self._get_current_loc(cid, state)
                if curr_p and goal_p and curr_p != goal_p:
                    h += 2
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
