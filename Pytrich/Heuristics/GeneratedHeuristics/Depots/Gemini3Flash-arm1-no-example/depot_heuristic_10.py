from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.Operators.operator import Operator
import collections

class DepotsGemini3FlashHeuristic10(Heuristic):
    """
    Heuristic for the Depots domain (Hybrid of Logistics and Blocksworld).
    
    Bottleneck Analysis:
    The bottleneck in Depots is the serial processing of crates using limited hoists and trucks. 
    Each crate must be unstacked, lifted, loaded, moved, unloaded, and dropped. 
    Surfaces (crates and pallets) must be cleared before use, creating serial dependencies.

    Admissibility Sketches:
    1. Crate Mobility: Each crate not at its target location requires at least 4-5 actions (Lift, Load, Drive, Unload, Drop).
    2. Clear Surface: If a surface is not clear, it requires at least 2 actions per crate on top to be cleared.
    3. Task Decomposition: Every task in the HTN network requires a minimum number of operators to resolve.
    
    We use an additive estimate for tasks in the network because each represents a disjoint sequence of required actions.
    """

    def __init__(self, depot_gemini3flash_heuristic_10="depot_gemini3flash_heuristic_10"):
        super().__init__(depot_gemini3flash_heuristic_10)
        self.fact_map = {} # (pred, args_tuple) -> global_id
        self.at_facts = collections.defaultdict(dict) # obj -> place -> global_id
        self.in_facts = collections.defaultdict(dict) # crate -> truck -> global_id
        self.lifting_facts = collections.defaultdict(dict) # hoist -> crate -> global_id
        self.on_crate_on_surface_facts = collections.defaultdict(dict) # surface -> crate -> global_id
        self.clear_facts = {} # surface -> global_id
        self.on_goal_facts = {} # crate -> surface -> global_id
        
        self.all_places = set()
        self.all_trucks = set()
        self.all_hoists = set()
        self.all_crates = set()
        self.all_surfaces = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, tuple(args)
        return name, ()

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            self.fact_map[(pred, args)] = f.global_id
            
            if pred == "at":
                self.at_facts[args[0]][args[1]] = f.global_id
                self.all_places.add(args[1])
                if "pallet" in args[0] or "crate" in args[0]: self.all_surfaces.add(args[0])
                if "truck" in args[0]: self.all_trucks.add(args[0])
                if "hoist" in args[0]: self.all_hoists.add(args[0])
            elif pred == "in":
                self.in_facts[args[0]][args[1]] = f.global_id
                self.all_crates.add(args[0])
            elif pred == "lifting":
                self.lifting_facts[args[0]][args[1]] = f.global_id
                self.all_crates.add(args[1])
            elif pred == "on":
                self.on_crate_on_surface_facts[args[1]][args[0]] = f.global_id
                self.all_crates.add(args[0])
                self.all_surfaces.add(args[1])
            elif pred == "clear":
                self.clear_facts[args[0]] = f.global_id
                self.all_surfaces.add(args[0])

        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "on":
                self.on_goal_facts[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _is_true(self, state, global_id):
        if global_id is None: return False
        return (state >> global_id) & 1

    def _get_loc_place(self, locatable, state):
        for p, fid in self.at_facts[locatable].items():
            if self._is_true(state, fid): return p
        if locatable in self.all_crates:
            for t, fid in self.in_facts[locatable].items():
                if self._is_true(state, fid): return self._get_loc_place(t, state)
            for h, h_dict in self.lifting_facts.items():
                if self._is_true(state, h_dict.get(locatable)): return self._get_loc_place(h, state)
        return None

    def _estimate_clear_cost(self, surface, state, depth=0):
        if depth > 10: return 0 # Guard against cycles or deep stacks
        if self._is_true(state, self.clear_facts.get(surface)): return 0
        for crate, fid in self.on_crate_on_surface_facts[surface].items():
            if self._is_true(state, fid):
                return 2 + self._estimate_clear_cost(crate, state, depth + 1)
        return 0

    def _estimate_task_cost(self, task, state):
        if isinstance(task, Operator):
            return task.cost
        
        name, args = self._parse_fact_name(task.name)
        if name == "do_put_on":
            c, s = args[0], args[1]
            if self._is_true(state, self.on_crate_on_surface_facts[s].get(c)): return 0
            cost = 2 # Min Lift + Drop
            loc_c = self._get_loc_place(c, state)
            loc_s = self._get_loc_place(s, state)
            if loc_c and loc_s and loc_c != loc_s: cost += 3 # Load + Drive + Unload
            cost += self._estimate_clear_cost(c, state)
            cost += self._estimate_clear_cost(s, state)
            return cost
        elif name == "do_clear":
            return self._estimate_clear_cost(args[0], state)
        elif name == "do_get_truck":
            return 0 if self._is_true(state, self.at_facts[args[0]].get(args[1])) else 1
        elif name == "do_lift_crate":
            return 1
        elif name == "do_load_truck":
            return 2
        elif name == "do_unload_truck":
            return 2
        return 1

    def _compute(self, state, task_network) -> int:
        if not task_network:
            h = 0
            for crate, target_surface in self.on_goal_facts.items():
                if not self._is_true(state, self.on_crate_on_surface_facts[target_surface].get(crate)):
                    h += 2
            return h

        h = 0
        for task in task_network:
            h += self._estimate_task_cost(task, state)
        return h
