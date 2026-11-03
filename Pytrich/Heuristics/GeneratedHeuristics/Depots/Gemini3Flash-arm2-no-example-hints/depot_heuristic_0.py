from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.Operators.operator import Operator

class DepotsGemini3FlashHeuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The primary serializing constraints in Depots are the stack structures (blocksworld-like) 
    and the limited transport (logistics-like). Crates must be cleared from stacks before they can 
    be moved, and destinations must be cleared before they can receive crates. Movement across 
    places requires trucks, while movement within a place requires hoists.

    Admissibility Sketch:
    1. Goal-based Term: For each unmet goal on(c, s), a lower bound cost is calculated based on 
       the location of c and s. If they are in the same place, it's at least a Lift + Drop (2). 
       If they are in different places, it's at least Lift + Load + Drive + Unload + Drop (5). 
       Additional clearing costs (2 per blocking crate) are added if the source or target are not clear.
    2. Task-based Term: Each task in the network is assigned a realistic lower-bound decomposition cost 
       based on the current state. do_put_on is costed similarly to unmet goals.
    """

    def __init__(self, depot_gemini3flash_heuristic_0="depot_gemini3flash_heuristic_0"):
        super().__init__(name=depot_gemini3flash_heuristic_0)
        self.at_fact_ids = {} # locatable -> [(place, gid)]
        self.in_fact_ids = {} # crate -> [(truck, gid)]
        self.on_fact_ids = {} # crate -> [(surface, gid)]
        self.clear_fact_ids = {} # surface -> gid
        self.lifting_fact_ids = {} # crate -> [(hoist, gid)]
        self.static_at = {} # locatable -> place
        self.goals = [] # List of (crate, surface)
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self._cache_place = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == "at":
                self.at_fact_ids.setdefault(args[0], []).append((args[1], f.global_id))
                self.places.add(args[1])
            elif pred == "in":
                self.in_fact_ids.setdefault(args[0], []).append((args[1], f.global_id))
                self.trucks.add(args[1])
            elif pred == "on":
                self.on_fact_ids.setdefault(args[0], []).append((args[1], f.global_id))
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "lifting":
                self.lifting_fact_ids.setdefault(args[1], []).append((args[0], f.global_id))

        # Recover static facts from operators
        for op in model.operators:
            pred, args = self._parse_name(op.name)
            if "Lift" in pred or "Drop" in pred:
                self.static_at[args[0]] = args[3] # hoist -> place
                self.static_at[args[2]] = args[3] # surface -> place
                self.hoists.add(args[0])
            elif "Load" in pred or "Unload" in pred:
                self.static_at[args[0]] = args[3] # hoist -> place
                self.static_at[args[2]] = args[3] # truck -> place (initial hint)
                self.hoists.add(args[0])

        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_name(name)
            if pred == "on":
                self.goals.append((args[0], args[1]))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        self._cache_place = {}
        
        # Term 1: Task Network Cost
        task_h = 0
        for task in task_network:
            task_h += self._get_task_cost(task, state)

        # Term 2: Unmet Goals Cost
        goal_h = 0
        for c, s in self.goals:
            on_gid = self.on_fact_ids.get(c, [])
            is_met = False
            for target_s, gid in on_gid:
                if target_s == s and (state >> gid) & 1:
                    is_met = True
                    break
            if not is_met:
                goal_h += self._estimate_put_on_cost(c, s, state)

        return max(task_h, goal_h)

    def _get_task_cost(self, task, state):
        if isinstance(task, Operator):
            return task.cost
        
        pred, args = self._parse_name(task.name)
        if pred == "do_put_on":
            return self._estimate_put_on_cost(args[0], args[1], state)
        elif pred == "do_clear":
            return 0 if self._is_clear(args[0], state) else 2
        elif pred == "do_get_truck":
            p_t = self._find_place(args[0], state)
            return 0 if p_t == args[1] else 1
        elif pred == "do_lift_crate":
            return 1
        elif pred == "do_load_truck" or pred == "do_unload_truck":
            return 2
        elif pred == "nop":
            return 0
        return 1

    def _estimate_put_on_cost(self, crate, surface, state):
        # Check if already done
        on_gids = self.on_fact_ids.get(crate, [])
        for s, gid in on_gids:
            if s == surface and (state >> gid) & 1:
                return 0
        
        p_c = self._find_place(crate, state)
        p_s = self._find_place(surface, state)
        
        base = 2 # Lift + Drop
        if p_c is not None and p_s is not None and p_c != p_s:
            base = 5 # Lift + Load + Drive + Unload + Drop
        
        if not self._is_clear(crate, state): base += 2
        if not self._is_clear(surface, state): base += 2
        return base

    def _find_place(self, obj, state, visited=None):
        if obj in self._cache_place: return self._cache_place[obj]
        if visited is None: visited = set()
        if obj in visited: return None
        visited.add(obj)

        if obj in self.static_at:
            res = self.static_at[obj]
            self._cache_place[obj] = res
            return res

        for p, gid in self.at_fact_ids.get(obj, []):
            if (state >> gid) & 1:
                self._cache_place[obj] = p
                return p

        for t, gid in self.in_fact_ids.get(obj, []):
            if (state >> gid) & 1:
                res = self._find_place(t, state, visited)
                self._cache_place[obj] = res
                return res

        for s, gid in self.on_fact_ids.get(obj, []):
            if (state >> gid) & 1:
                res = self._find_place(s, state, visited)
                self._cache_place[obj] = res
                return res

        for h, gid in self.lifting_fact_ids.get(obj, []):
            if (state >> gid) & 1:
                res = self._find_place(h, state, visited)
                self._cache_place[obj] = res
                return res

        return None

    def _is_clear(self, surface, state):
        gid = self.clear_fact_ids.get(surface)
        if gid is None: return True
        return (state >> gid) & 1

    def _parse_name(self, name):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name:
            parts = name.split("[")
            return parts[0], parts[1].replace("]", "").split(",")
        if "(" in name:
            parts = name.split("(")
            return parts[0], parts[1].replace(")", "").split(",")
        return name, []
