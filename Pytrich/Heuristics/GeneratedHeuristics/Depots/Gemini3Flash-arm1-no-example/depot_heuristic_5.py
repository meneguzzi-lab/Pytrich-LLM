from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The Depots domain combines Logistics and Blocksworld. The primary bottlenecks are:
# 1. Stack Clearance: To access a crate at the bottom of a stack or to place a crate on a pallet/surface, 
#    intervening crates must be moved (Lift + Drop/Load), requiring at least 2 actions per obstructing crate.
# 2. Serial Execution: Each location has a single hoist (resource) for lifting/dropping, and trucks must move between locations.
# 3. Transportation: Moving a crate from one location to another requires a specific sequence (Lift-Load-Drive-Unload-Drop).

# Admissible Lower Bounds:
# Term 1: Task Network Minimum Cost. For each task in the network, we estimate the minimum actions required to decompose it to primitives.
#         - do_put_on(c, s): 2 actions if c and s are at the same location, 5 actions if different locations.
#         - do_clear(s, p): 2 actions if not clear.
#         - do_get_truck(t, p): 1 action if t not at p.
# Term 2: Goal State Lower Bound. For every crate c whose goal surface s is not yet reached, we add the minimal movement cost (at least 2 actions).

class DepotsGemini3FlashHeuristic5(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_5="depot_gemini3flash_heuristic_5"):
        super().__init__(depot_gemini3flash_heuristic_5)
        self.at_facts = {}       # (obj, place) -> global_id
        self.on_facts = {}       # (crate, surface) -> global_id
        self.in_facts = {}       # (crate, truck) -> global_id
        self.clear_facts = {}    # surface -> global_id
        self.lifting_facts = {}  # (hoist, crate) -> global_id
        self.hoist_at = {}       # hoist -> place
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.task_data = {}      # task_id -> (predicate, args)
        self.goal_on = {}        # crate -> target_surface

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_facts[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
                # Hoists are fixed at places in Depots
                if "hoist" in args[0].lower() or any(h.name == args[0] for h in model.operators if "hoist" in args[0]):
                    self.hoist_at[args[0]] = args[1]
                    self.hoists.add(args[0])
                if "truck" in args[0].lower():
                    self.trucks.add(args[0])
            elif pred == "on":
                self.on_facts[(args[0], args[1])] = f.global_id
            elif pred == "in":
                self.in_facts[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_facts[args[0]] = f.global_id
            elif pred == "lifting":
                self.lifting_facts[(args[0], args[1])] = f.global_id

        for t in model.abstract_tasks:
            self.task_data[t.global_id] = self._parse_task_name(t.name)
        for op in model.operators:
            self.task_data[op.global_id] = ("OPERATOR", op.cost)

        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        term1 = 0
        for task in task_network:
            info = self.task_data.get(task.global_id)
            if not info: continue
            if info[0] == "OPERATOR":
                term1 += info[1]
                continue
            
            pred, args = info
            if pred == "do_put_on":
                c, s = args[0], args[1]
                if self._is_on(c, s, state): continue
                loc_c = self._get_obj_location(c, state)
                loc_s = self._get_obj_location(s, state)
                if loc_c and loc_s:
                    term1 += 2 if loc_c == loc_s else 5
                else: term1 += 2
                if not self._is_clear(c, state): term1 += 2
                if not self._is_clear(s, state): term1 += 2
            elif pred == "do_clear":
                if not self._is_clear(args[0], state): term1 += 2
            elif pred == "do_get_truck":
                if not self._is_at(args[0], args[1], state): term1 += 1
            elif pred in ["do_lift_crate", "do_load_truck", "do_unload_truck"]:
                term1 += 2
            else: term1 += 1

        term2 = 0
        for crate, target_s in self.goal_on.items():
            if not self._is_on(crate, target_s, state):
                loc_c = self._get_obj_location(crate, state)
                loc_s = self._get_obj_location(target_s, state)
                if loc_c and loc_s:
                    term2 += 2 if loc_c == loc_s else 5
                else: term2 += 2

        return max(term1, term2)

    def _is_on(self, c, s, state): 
        fid = self.on_facts.get((c, s))
        return bool((state >> fid) & 1) if fid is not None else False

    def _is_clear(self, s, state):
        fid = self.clear_facts.get(s)
        return bool((state >> fid) & 1) if fid is not None else False

    def _is_at(self, obj, p, state):
        fid = self.at_facts.get((obj, p))
        return bool((state >> fid) & 1) if fid is not None else False

    def _get_obj_location(self, obj, state):
        for p in self.places:
            if self._is_at(obj, p, state): return p
        for t in self.trucks:
            if any((state >> self.in_facts.get((obj, t), 0)) & 1 for _ in [0] if (obj, t) in self.in_facts):
                for p in self.places:
                    if self._is_at(t, p, state): return p
        for h in self.hoists:
            if any((state >> self.lifting_facts.get((h, obj), 0)) & 1 for _ in [0] if (h, obj) in self.lifting_facts):
                return self.hoist_at.get(h)
        return None

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []
