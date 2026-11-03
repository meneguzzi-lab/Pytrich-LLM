from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic2(Heuristic):
    """
    Depots Domain Heuristic
    
    Bottleneck Analysis:
    The core bottleneck in the Depots domain is the serialization of crate movement. 
    Crates must be cleared from stacks (Blocksworld), loaded onto trucks, transported, 
    unloaded, and then placed on target stacks (Logistics). The most expensive part 
    is the combination of stack depth and distance.

    Admissible Lower Bounds:
    1. Crate Displacement: Any crate not on its goal surface must be moved. 
       This requires at least a Lift and a Drop (2 actions).
    2. Logistics Cost: If a crate's current location differs from its goal surface location, 
       it must be loaded, driven, and unloaded (+3 actions). We use a safe +2 for Load/Unload.
    3. Task Network Requirement: Each non-trivial task in the network implies at least one action.
    4. Stability: A crate currently on its goal surface is not 'done' if the surface below it 
       still needs to move. These crates will eventually have to be unstacked and restacked.
    """

    def __init__(self, depot_gemini3flash_heuristic_2="depots_heuristic_2"):
        super().__init__(name=depot_gemini3flash_heuristic_2)
        self.goal_on = {} # crate_name -> surface_name
        self.on_fact_ids = {} # (crate, surface) -> id
        self.at_fact_ids = {} # (locatable, place) -> id
        self.in_fact_ids = {} # (crate, truck) -> id
        self.lifting_fact_ids = {} # (hoist, crate) -> id
        self.clear_fact_ids = {} # surface -> id
        
        self.pallets = set()
        self.crates = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()
        
        self.static_locs = {} # surface/hoist -> place
        self.crate_at_map = {} # crate -> [(fact_id, place_name)]
        self.crate_in_map = {} # crate -> [(fact_id, truck_name)]
        self.crate_lifting_map = {} # crate -> [(fact_id, hoist_name)]
        self.truck_at_map = {} # truck -> [(fact_id, place_name)]

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
            elif pred == "at":
                self.at_fact_ids[(args[0], args[1])] = fact.global_id
                self.places.add(args[1])
            elif pred == "in":
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "lifting":
                self.lifting_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = fact.global_id

        # Recover static locations from operator names (PANDA grounder fix)
        for op in model.operators:
            pred, args = self._parse_fact_name("+" + op.name)
            if pred in ["Lift", "Drop", "Load", "Unload"]:
                # Args: [hoist, crate, surface/truck, place]
                self.static_locs[args[0]] = args[3] # Hoist loc
                if pred in ["Lift", "Drop"]:
                    self.static_locs[args[2]] = args[3] # Pallet/Crate loc (Pallets are static)

        # Identify pallets (surfaces that aren't crates)
        for s_loc in self.static_locs:
            if s_loc not in self.crates and s_loc not in self.trucks and s_loc not in self.hoists:
                self.pallets.add(s_loc)

        # Map fact IDs to specific objects for fast checking
        for c in self.crates:
            self.crate_at_map[c] = [(self.at_fact_ids[(c, p)], p) for p in self.places if (c, p) in self.at_fact_ids]
            self.crate_in_map[c] = [(self.in_fact_ids[(c, t)], t) for t in model.facts if self._is_truck_fact(t, c)]
            self.crate_lifting_map[c] = [(self.lifting_fact_ids[(h, c)], h) for h in model.facts if self._is_hoist_fact(h, c)]
        
        # Re-map crate_in_map and crate_lifting_map to use actual truck/hoist names
        self.trucks = {t for f, t in self.in_fact_ids}
        self.hoists = {h for h, c in self.lifting_fact_ids}
        for c in self.crates:
            self.crate_in_map[c] = [(self.in_fact_ids[(c, t)], t) for t in self.trucks if (c, t) in self.in_fact_ids]
            self.crate_lifting_map[c] = [(self.lifting_fact_ids[(h, c)], h) for h in self.hoists if (h, c) in self.lifting_fact_ids]
            
        for t in self.trucks:
            self.truck_at_map[t] = [(self.at_fact_ids[(t, p)], p) for p in self.places if (t, p) in self.at_fact_ids]

        # Parse goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _is_truck_fact(self, t, c): return False # Placeholder for complex logic
    def _is_hoist_fact(self, h, c): return False # Placeholder for complex logic

    def _parse_fact_name(self, name):
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

    def _parse_task_name(self, name):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_type = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return task_type, args
        return name, []

    def _get_loc(self, crate, state):
        # Check direct location
        for fid, p in self.crate_at_map.get(crate, []):
            if (state >> fid) & 1: return p
        # Check truck
        for fid, t in self.crate_in_map.get(crate, []):
            if (state >> fid) & 1:
                for tfid, tp in self.truck_at_map.get(t, []):
                    if (state >> tfid) & 1: return tp
        # Check hoist
        for fid, h in self.crate_lifting_map.get(crate, []):
            if (state >> fid) & 1: return self.static_locs.get(h)
        return None

    def _is_stable(self, crate, state, current_on):
        curr = crate
        while True:
            if curr not in self.goal_on: return False
            target = self.goal_on[curr]
            if curr not in current_on or current_on[curr] != target: return False
            if target in self.pallets: return True
            if target not in self.crates: return False
            curr = target

    def _compute(self, state, task_network):
        # 1. Map current state stacks
        current_on = {}
        for (c, s), fid in self.on_fact_ids.items():
            if (state >> fid) & 1:
                current_on[c] = s

        # 2. Goal-based Heuristic (Admissible Term 1)
        goal_h = 0
        for c, g_s in self.goal_on.items():
            if not self._is_stable(c, state, current_on):
                goal_h += 2 # Needs at least Lift/Drop
                loc_c = self._get_loc(c, state)
                loc_g = self.static_locs.get(g_s) if g_s in self.static_locs else self._get_loc(g_s, state)
                if loc_c and loc_g and loc_c != loc_g:
                    goal_h += 2 # Needs at least Load/Unload

        # 3. Task-network-based Heuristic (Admissible Term 2)
        tn_h = 0
        for task in task_network:
            t_type, args = self._parse_task_name(task.name)
            if t_type == "do_put_on":
                c, s = args[0], args[1]
                on_fid = self.on_fact_ids.get((c, s))
                if not (on_fid and (state >> on_fid) & 1 and self._is_stable(c, state, current_on)):
                    tn_h += 2
                    # Nop bonus: if not even 'on' yet, it's definitely expensive
                    if not (on_fid and (state >> on_fid) & 1):
                        tn_h += 2
            elif t_type in ["do_clear", "do_load_truck", "do_unload_truck", "do_lift_crate"]:
                tn_h += 2

        return max(goal_h, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
