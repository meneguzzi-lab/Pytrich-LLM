import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic13(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    Bottleneck Analysis:
    The main bottleneck is the serial coordination of hoists and trucks to move crates 
    between locations and stack them in specific orders. Moving a crate involves 
    unstacking (clearing), lifting, transporting (loading/driving/unloading), and stacking.

    Admissibility Sketches:
    1. Task Network Cost (h_tasks): Each task in the frontier requires a minimum number 
       of physical actions (Drive, Lift, Drop, Load, Unload) to resolve. Since these tasks 
       represent a decomposition frontier of the goal, the sum of their minimum 
       completion costs is a lower bound on the remaining plan length.
    2. State-Aware Task Costing: The cost of a task like do_put_on is dynamically calculated 
       based on whether the target crate and surface are at the same location (min 2 actions) 
       or different locations (min 4 actions).
    """

    def __init__(self, depot_gemini3flash_heuristic_13=None, name="DepotsGemini3FlashHeuristic13"):
        super().__init__(name=name)
        self.on_id = {}  # (crate, surface) -> global_id
        self.at_id = {}  # locatable -> {place: global_id}
        self.clear_id = {}  # surface -> global_id
        self.in_id_by_crate = {}  # crate -> [(truck, global_id)]
        self.lifting_id_by_crate = {}  # crate -> [(hoist, global_id)]
        self.surface_to_place = {}  # pallet -> place (static)
        self.hoist_to_place = {}  # hoist -> place (static)
        self.crates = set()
        self.trucks = set()
        self.pallets = set()
        self.hoists = set()
        self.on_goal = {} # crate -> surface

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Precompute goal information
        if hasattr(model, 'goals') and model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 2:
                    self.on_goal[args[0]] = args[1]

        # Map facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == "on":
                self.on_id[(args[0], args[1])] = f.global_id
                self.crates.add(args[0])
            elif pred == "at":
                loc, place = args[0], args[1]
                if loc not in self.at_id: self.at_id[loc] = {}
                self.at_id[loc][place] = f.global_id
            elif pred == "clear":
                self.clear_id[args[0]] = f.global_id
            elif pred == "in":
                c, t = args[0], args[1]
                if c not in self.in_id_by_crate: self.in_id_by_crate[c] = []
                self.in_id_by_crate[c].append((t, f.global_id))
                self.trucks.add(t)
            elif pred == "lifting":
                h, c = args[0], args[1]
                if c not in self.lifting_id_by_crate: self.lifting_id_by_crate[c] = []
                self.lifting_id_by_crate[c].append((h, f.global_id))
                self.hoists.add(h)

        # Identify static locations for hoists and pallets
        # Initial state is used as ground truth for static locations
        init_facts = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "at":
                loc, p = args[0], args[1]
                if loc in self.hoists: self.hoist_to_place[loc] = p
                if "pallet" in loc.lower():
                    self.surface_to_place[loc] = p
                    self.pallets.add(loc)

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, [a.strip() for a in args]
        return None, []

    def _check(self, state, fact_id): 
        return (state >> fact_id) & 1 if fact_id is not None else False

    def _compute(self, state, task_network):
        if not task_network: return 0

        # 1. Update dynamic locations
        truck_at = {}
        for t in self.trucks:
            for p, f_id in self.at_id.get(t, {}).items():
                if self._check(state, f_id):
                    truck_at[t] = p
                    break

        curr_at = {}
        for p_id in self.pallets: curr_at[p_id] = self.surface_to_place[p_id]
        for c in self.crates:
            found = False
            for p, f_id in self.at_id.get(c, {}).items():
                if self._check(state, f_id):
                    curr_at[c] = p
                    found = True; break
            if found: continue
            for t, f_id in self.in_id_by_crate.get(c, []):
                if self._check(state, f_id):
                    curr_at[c] = truck_at.get(t)
                    found = True; break
            if found: continue
            for h, f_id in self.lifting_id_by_crate.get(c, []):
                if self._check(state, f_id):
                    curr_at[c] = self.hoist_to_place.get(h)
                    break

        # 2. Sum costs of tasks in network
        h_tasks = 0
        for t_id in task_network:
            task = self.model.get_component(t_id)
            t_type, args = self._parse_fact_name(task.name)
            if not t_type: continue
            
            if t_type == "do_put_on" and len(args) == 2:
                c, s = args[0], args[1]
                if not self._check(state, self.on_id.get((c, s))):
                    cost = 2
                    p_c, p_s = curr_at.get(c), curr_at.get(s)
                    if p_c and p_s and p_c != p_s: cost += 2
                    if not self._check(state, self.clear_id.get(c)): cost += 1
                    if not self._check(state, self.clear_id.get(s)): cost += 1
                    h_tasks += cost
            elif t_type == "do_clear":
                if not self._check(state, self.clear_id.get(args[0])): h_tasks += 2
            elif t_type == "do_get_truck":
                t, p = args[0], args[1]
                if truck_at.get(t) != p: h_tasks += 1
            elif t_type in ["do_load_truck", "do_unload_truck"]: h_tasks += 2
            elif t_type == "do_lift_crate": h_tasks += 1
            elif t_type in ["Lift", "Drop", "Load", "Unload", "Drive"]: h_tasks += 1
            elif t_type == "nop": h_tasks += 0
            else: h_tasks += 1

        return h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
