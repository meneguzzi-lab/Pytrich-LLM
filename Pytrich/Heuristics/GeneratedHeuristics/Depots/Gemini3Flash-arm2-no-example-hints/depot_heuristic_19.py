from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic19(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots Domain.
    
    Bottleneck Analysis:
    The main bottleneck in Depots is the serial nature of hoists and the logistical coordination of trucks. 
    A crate must be 'cleared' from its stack, 'lifted' by a hoist, 'loaded' into a truck, 'driven' to a 
    new location, 'unloaded', and finally 'dropped' onto its goal surface. Each hoist and truck can only 
    handle one or a limited number of items/tasks at a time, and stack depths create long action sequences.

    Admissible Lower Bounds:
    1. Term 1 (Goal-based): For every goal (on c s), if the condition is not satisfied, it requires at least 
       a Lift and a Drop (2 actions) if c and s are at the same location, or a Lift, Load, Drive, Unload, Drop 
       (5 actions) if they are at different locations. We also add costs for clearing burying crates.
    2. Term 2 (Task Network-based): Sum of state-aware lower bounds for every task in the current network. 
       A 'do_put_on' task costs at least 2 or 5 actions depending on locations, and 'do_clear' costs 2 if the 
       surface is not clear.

    The heuristic returns max(Term1, Term2) for a strong admissible lower bound.
    """

    def __init__(self, depot_gemini3flash_heuristic_19="depot_gemini3flash_heuristic_19"):
        super().__init__(depot_gemini3flash_heuristic_19)
        self.on_fact_map = {} # (crate, surface) -> id
        self.at_fact_map = {} # (obj, place) -> id
        self.in_fact_map = {} # (crate, truck) -> id
        self.lifting_fact_map = {} # (hoist, crate) -> id
        self.clear_fact_map = {} # obj -> id
        self.available_fact_map = {} # hoist -> id
        
        self.on_goal = {} # crate -> surface
        self.task_info = {}
        self.surface_to_place_name = {}
        self.hoist_to_place_name = {}
        self.place_names = []
        self.truck_names = []
        self.hoist_names = []
        self.name_to_id = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Build ID maps for faster lookup
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "on": self.on_fact_map[(args[0], args[1])] = f.global_id
            elif pred == "at": self.at_fact_map[(args[0], args[1])] = f.global_id
            elif pred == "in": self.in_fact_map[(args[0], args[1])] = f.global_id
            elif pred == "lifting": self.lifting_fact_map[(args[0], args[1])] = f.global_id
            elif pred == "clear": self.clear_fact_map[args[0]] = f.global_id
            elif pred == "available": self.available_fact_map[args[0]] = f.global_id

        # Goal parsing
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.on_goal[args[0]] = args[1]

        # Pre-parse all operator and abstract task info
        for group in [model.abstract_tasks, model.operators]:
            for t in group:
                name = t.name
                t_type = name[:name.find('[')] if '[' in name else name
                t_args = name[name.find('[')+1:-1].split(',') if '[' in name else []
                self.task_info[t.global_id] = (t_type, t_args)

        # Location recovery for surfaces/hoists/trucks (handles static fact stripping)
        places = set()
        trucks = set()
        hoists = set()
        for op in model.operators:
            parts = self.task_info[op.global_id][1]
            if op.name.startswith("Drop[") or op.name.startswith("Lift["):
                self.hoist_to_place_name[parts[0]] = parts[3]
                self.surface_to_place_name[parts[2]] = parts[3]
                places.add(parts[3])
                hoists.add(parts[0])
            elif op.name.startswith("Drive["):
                trucks.add(parts[0])
                places.add(parts[1])
                places.add(parts[2])

        self.place_names = list(places)
        self.truck_names = list(trucks)
        self.hoist_names = list(hoists)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 2: TN Task Costs
        h_tn = 0
        crates_in_tn = set()
        for task in task_network:
            t_type, t_args = self.task_info.get(task.global_id, (None, []))
            if t_type == "do_put_on":
                crates_in_tn.add(t_args[0])
                h_tn += self._estimate_put_on(state, t_args[0], t_args[1])
            elif t_type == "do_clear":
                h_tn += self._estimate_clear(state, t_args[0])
            elif t_type == "do_get_truck":
                h_tn += self._estimate_get_truck(state, t_args[0], t_args[1])
            elif t_type in ["Lift", "Drop", "Load", "Unload", "Drive"]:
                h_tn += 1
            # nop and method_preconditions cost 0

        # Term 1: Goal Costs for crates NOT in the current TN branch
        h_goal = 0
        for c_n, s_n in self.on_goal.items():
            if c_n not in crates_in_tn:
                if not self._is_on_satisfied(state, c_n, s_n):
                    h_goal += self._estimate_put_on(state, c_n, s_n)

        return max(h_tn, h_goal)

    def _is_on_satisfied(self, state, c_n, s_n):
        on_f = self.on_fact_map.get((c_n, s_n))
        if on_f is None or not ((state >> on_f) & 1): return False
        # If s_n is a crate, it also needs to be satisfied (stability)
        if s_n in self.on_goal:
            return self._is_on_satisfied(state, s_n, self.on_goal[s_n])
        return True

    def _get_loc(self, state, obj_name):
        # Direct at fact
        for p_name in self.place_names:
            f_id = self.at_fact_map.get((obj_name, p_name))
            if f_id is not None and (state >> f_id) & 1: return p_name
        # Inside truck
        for t_name in self.truck_names:
            f_id = self.in_fact_map.get((obj_name, t_name))
            if f_id is not None and (state >> f_id) & 1: return self._get_loc(state, t_name)
        # Being lifted
        for h_name in self.hoist_names:
            f_id = self.lifting_fact_map.get((h_name, obj_name))
            if f_id is not None and (state >> f_id) & 1: return self._get_loc(state, h_name)
        # Static surface
        return self.surface_to_place_name.get(obj_name)

    def _estimate_put_on(self, state, c_n, s_n):
        on_f = self.on_fact_map.get((c_n, s_n))
        if on_f is not None and (state >> on_f) & 1: return 0
        
        # Check if currently lifting this crate
        for h_name in self.hoist_names:
            lf = self.lifting_fact_map.get((h_name, c_n))
            if lf is not None and (state >> lf) & 1:
                h_loc = self.hoist_to_place_name.get(h_name)
                s_loc = self._get_loc(state, s_n)
                return 1 if h_loc == s_loc else 4

        loc_c = self._get_loc(state, c_n)
        loc_s = self._get_loc(state, s_n)
        if loc_c and loc_s:
            base = 2 if loc_c == loc_s else 5
        else:
            base = 2

        # Nop bonus / clearance penalty
        penalty = 0
        clear_c = self.clear_fact_map.get(c_n)
        if clear_c is not None and not ((state >> clear_c) & 1): penalty += 2
        clear_s = self.clear_fact_map.get(s_n)
        if clear_s is not None and not ((state >> clear_s) & 1): penalty += 2
        
        return base + penalty

    def _estimate_clear(self, state, s_n):
        cf = self.clear_fact_map.get(s_n)
        if cf is not None and (state >> cf) & 1: return 0
        return 2

    def _estimate_get_truck(self, state, t_n, p_n):
        af = self.at_fact_map.get((t_n, p_n))
        if af is not None and (state >> af) & 1: return 0
        return 1

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
