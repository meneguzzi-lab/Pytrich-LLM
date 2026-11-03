from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS: 
# The Depots domain is a fusion of Logistics and Blocksworld. The primary bottleneck is the serial coordination 
# of trucks and hoists to move crates between locations while respecting stack constraints (clearance). 
# Transporting a crate between locations is high-cost, while stacking/unstacking is lower-cost but requires hoists.

# ADMISSIBLE LOWER BOUND SKETCHES:
# 1. Transport Cost: Every crate not at its goal location requires at least 5 actions (Lift, Load, Drive, Unload, Drop).
# 2. Local Stack Cost: Every crate at the correct location but not on its goal surface requires at least 2 actions (Lift, Drop).
# 3. Stability Cost: If a crate is on its goal surface, but the surface itself is not stable, the crate must eventually be moved.
# 4. Task Network size: Every non-nop task in the current network represents at least one future action.

class DepotsGemini3FlashHeuristic2(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_2="DepotsGemini3FlashHeuristic2"):
        super().__init__(depot_gemini3flash_heuristic_2)
        self.on_goal_dict = {}  # crate -> goal_surface
        self.crate_goal_place = {}  # crate -> place_name
        self.on_fact_ids = {}  # (crate, surface) -> global_id
        self.at_fact_ids = {}  # (obj, place) -> global_id
        self.in_fact_ids = {}  # (crate, truck) -> global_id
        self.lifting_fact_ids = {}  # (hoist, crate) -> global_id
        
        self.crate_at_facts = {}  # crate -> [(place, id)]
        self.crate_in_facts = {}  # crate -> [(truck, id)]
        self.truck_at_facts = {}  # truck -> [(place, id)]
        self.crate_lifting_facts = {}  # crate -> [(hoist, id)]
        self.goal_on_fact_ids = {} # crate -> global_id

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return prefix, predicate, args
        return None, None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map all facts
        obj_to_initial_place = {}
        for f in model.facts:
            _, pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                self.on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == 'at':
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.crate_at_facts: self.crate_at_facts[args[0]] = []
                self.crate_at_facts[args[0]].append((args[1], f.global_id))
                if (model.initial_state >> f.global_id) & 1:
                    obj_to_initial_place[args[0]] = args[1]
                if args[0].startswith('truck'):
                    if args[0] not in self.truck_at_facts: self.truck_at_facts[args[0]] = []
                    self.truck_at_facts[args[0]].append((args[1], f.global_id))
            elif pred == 'in':
                self.in_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.crate_in_facts: self.crate_in_facts[args[0]] = []
                self.crate_in_facts[args[0]].append((args[1], f.global_id))
            elif pred == 'lifting':
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
                if args[1] not in self.crate_lifting_facts: self.crate_lifting_facts[args[1]] = []
                self.crate_lifting_facts[args[1]].append((args[0], f.global_id))

        # 2. Parse Goals
        goal_names = model.state_explicit_repr(model.goals)
        for g in goal_names:
            _, pred, args = self._parse_fact_name(g)
            if pred == 'on':
                self.on_goal_dict[args[0]] = args[1]
                self.goal_on_fact_ids[args[0]] = self.on_fact_ids.get((args[0], args[1]))

        # 3. Determine Goal Place for every crate
        for c in self.on_goal_dict:
            curr = c
            visited = {c}
            while curr in self.on_goal_dict:
                curr = self.on_goal_dict[curr]
                if curr in visited: break
                visited.add(curr)
            if curr in obj_to_initial_place:
                self.crate_goal_place[c] = obj_to_initial_place[curr]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        h = 0
        stable = {}
        
        def check_stable(c):
            if c in stable: return stable[c]
            if c not in self.on_goal_dict: 
                stable[c] = True
                return True
            fid = self.goal_on_fact_ids.get(c)
            if fid is not None and (state >> fid) & 1:
                gs = self.on_goal_dict[c]
                # If gs is a pallet, it's inherently stable because it doesn't move
                if gs.startswith('pallet') or gs.startswith('depot') or gs.startswith('distributor'):
                    stable[c] = True
                    return True
                res = check_stable(gs)
                stable[c] = res
                return res
            stable[c] = False
            return False

        for c in self.on_goal_dict:
            if not check_stable(c):
                goal_p = self.crate_goal_place.get(c)
                # Find where crate c is
                found = False
                # Case: Lifting
                for _, fid in self.crate_lifting_facts.get(c, []):
                    if (state >> fid) & 1: h += 1; found = True; break
                if found: continue
                # Case: At a place
                for p_name, fid in self.crate_at_facts.get(c, []):
                    if (state >> fid) & 1:
                        h += (2 if p_name == goal_p else 5)
                        found = True; break
                if found: continue
                # Case: In a truck
                for t_name, fid in self.crate_in_facts.get(c, []):
                    if (state >> fid) & 1:
                        in_right_place = False
                        for p_name, tafid in self.truck_at_facts.get(t_name, []):
                            if (state >> tafid) & 1:
                                h += (2 if p_name == goal_p else 3)
                                in_right_place = True; break
                        if not in_right_place: h += 3
                        found = True; break
        
        # Guidance from task network
        h += sum(1 for t in task_network if t.name != 'nop')
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
