from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# ANALYSIS:
# 1. Bottleneck: The single-hoist-per-location constraint and the stacking nature of crates.
#    Each hoist can only process one crate at a time, and a crate must be 'clear' and 'available' 
#    to be moved, creating serial dependencies like Blocksworld.
# 2. Independent Admissible Lower Bounds:
#    - Goal-Distance: Each crate not on its goal surface requires at least 2 actions (Lift, Drop) 
#      if the surface is at the same location, or 5 actions (Lift, Load, Drive, Unload, Drop) 
#      if it's at a different location.
#    - Clearance Cost: If a crate needs to be moved and is not 'clear', at least one more crate 
#      must be lifted and moved elsewhere (+2 actions).
#    - Task Network Cost: Every task in the network represents a commitment to at least one action.

class DepotsGemini3FlashHeuristic12(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_12="depot_gemini3flash_heuristic_12"):
        super().__init__(name=depot_gemini3flash_heuristic_12)
        self.goal_on_pairs = []  # (crate_id, surface_id, goal_fact_id)
        self.clear_fact_ids = {} # obj_id -> global_id
        self.on_fact_ids = {}    # crate_id -> {surface_id: global_id}
        self.at_fact_ids = {}    # locatable_id -> {place_id: global_id}
        self.in_truck_fact_ids = {} # crate_id -> {truck_id: global_id}
        
        self.crate_ids = set()
        self.surface_ids = set()
        self.place_ids = set()
        self.truck_ids = set()
        
        self.obj_to_place = {} # Fixed places for pallets/depots

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
        # 1. Map Predicates
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "on":
                c, s = args[0], args[1]
                if c not in self.on_fact_ids: self.on_fact_ids[c] = {}
                self.on_fact_ids[c][s] = f.global_id
                self.crate_ids.add(c)
                self.surface_ids.add(s)
            elif pred == "at":
                l, p = args[0], args[1]
                if l not in self.at_fact_ids: self.at_fact_ids[l] = {}
                self.at_fact_ids[l][p] = f.global_id
                self.place_ids.add(p)
                # Static analysis: find fixed locations (pallets are static places)
                if "pallet" in l or "depot" in l or "distributor" in l:
                    self.obj_to_place[l] = p
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "in":
                c, t = args[0], args[1]
                if c not in self.in_truck_fact_ids: self.in_truck_fact_ids[c] = {}
                self.in_truck_fact_ids[c][t] = f.global_id
                self.truck_ids.add(t)

        # 2. Index Goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                crate, surface = args[0], args[1]
                fact_id = self.on_fact_ids.get(crate, {}).get(surface)
                if fact_id is not None:
                    self.goal_on_pairs.append((crate, surface, fact_id))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _get_current_place(self, obj, state):
        # For static objects
        if obj in self.obj_to_place: return self.obj_to_place[obj]
        # For movable objects
        for p, fid in self.at_fact_ids.get(obj, {}).items():
            if (state >> fid) & 1: return p
        # For crates in trucks
        for t, fid in self.in_truck_fact_ids.get(obj, {}).items():
            if (state >> fid) & 1:
                # Crate is in truck, find truck's place
                return self._get_current_place(t, state)
        return None

    def _compute(self, state, task_network):
        # Term 1: Goal-based distance
        goal_h = 0
        moving_crates = set()
        
        for crate, target_surface, goal_fid in self.goal_on_pairs:
            # If goal is already met, cost 0
            if (state >> goal_fid) & 1: 
                continue
            
            moving_crates.add(crate)
            c_loc = self._get_current_place(crate, state)
            s_loc = self._get_current_place(target_surface, state)
            
            if c_loc == s_loc and c_loc is not None:
                goal_h += 2 # Lift + Drop
            else:
                goal_h += 5 # Lift + Load + Drive + Unload + Drop
            
            # Clearance check (P2: State-aware cost)
            if crate in self.clear_fact_ids:
                if not ((state >> self.clear_fact_ids[crate]) & 1):
                    goal_h += 2 # Need to move at least one crate on top
            if target_surface in self.clear_fact_ids:
                if not ((state >> self.clear_fact_ids[target_surface]) & 1):
                    goal_h += 2

        # Term 2: Task-network weight
        tn_h = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                tn_h += task.cost
            else:
                # Abstract task or nop
                if "nop" not in task.name:
                    tn_h += 1
                    # Nop bonus (P5/Hint): If commitment exists but nop impossible, cost is higher
                    if "do_put_on" in task.name:
                        tn_h += 1

        return max(goal_h, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
