from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic1(Heuristic):
    """
    Heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The main bottleneck is the serialization of surface availability (clearing stacks of crates) 
    and hoist/truck coordination. Moving a crate involves recursive unstacking, loading into 
    trucks for inter-place transport, and clearing the destination surface.

    Admissibility Sketches:
    1. Goal Cost: For each goal on(c, s) not satisfied, we need at least 2 actions (Lift, Drop) 
       if in the same place, and 5 actions (Lift, Load, Drive, Unload, Drop) if in different places.
       Additional clearance costs (+2 per crate blocking) are required lower bounds for any unstacking.
    2. Task Network Cost: The HTN task network represents a commitment to perform work. 
       Summing the minimum cost of each task (2 for do_put_on, 2 for do_clear, etc.) is a lower bound.
    """

    def __init__(self, depot_gemini3flash_heuristic_1="depot_gemini3flash_heuristic_1"):
        super().__init__(depot_gemini3flash_heuristic_1)
        self.on_ids = {}         # (crate, surface) -> fact_id
        self.clear_ids = {}      # obj -> fact_id
        self.at_ids = {}         # obj -> {place -> fact_id}
        self.in_ids = {}         # crate -> {truck -> fact_id}
        self.lifting_ids = {}    # crate -> {hoist -> fact_id}
        self.truck_at_ids = {}   # truck -> {place -> fact_id}
        self.hoist_at_ids = {}   # hoist -> {place -> fact_id}
        self.goal_on = []        # [(crate, surface)]

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to build efficient lookup maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id
            elif pred == "at":
                obj, loc = args[0], args[1]
                # Handle different types of locatables
                if "truck" in obj:
                    if obj not in self.truck_at_ids: self.truck_at_ids[obj] = {}
                    self.truck_at_ids[obj][loc] = fact.global_id
                elif "hoist" in obj:
                    if obj not in self.hoist_at_ids: self.hoist_at_ids[obj] = {}
                    self.hoist_at_ids[obj][loc] = fact.global_id
                else:
                    if obj not in self.at_ids: self.at_ids[obj] = {}
                    self.at_ids[obj][loc] = fact.global_id
            elif pred == "in":
                if args[0] not in self.in_ids: self.in_ids[args[0]] = {}
                self.in_ids[args[0]][args[1]] = fact.global_id
            elif pred == "lifting":
                if args[1] not in self.lifting_ids: self.lifting_ids[args[1]] = {}
                self.lifting_ids[args[1]][args[0]] = fact.global_id

        # 2. Extract Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_on.append((args[0], args[1]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 1: Goal State Progress
        h_goal = 0
        for crate, surface in self.goal_on:
            h_goal += self._estimate_on_cost(crate, surface, state)

        # Term 2: Task Network Work Remaining
        h_tn = 0
        for task in task_network:
            if hasattr(task, 'cost'): # Operator
                if task.name == "nop": continue
                h_tn += task.cost if task.cost > 0 else 1
            else: # AbstractTask
                t_name, t_args = self._parse_task_name(task.name)
                if t_name == "do_put_on":
                    h_tn += self._estimate_on_cost(t_args[0], t_args[1], state)
                elif t_name == "do_clear":
                    if not self._is_clear(state, t_args[0]): h_tn += 2
                elif t_name == "do_get_truck":
                    if not self._is_at(state, t_args[0], t_args[1]): h_tn += 1
                elif t_name == "do_lift_crate": h_tn += 1
                elif t_name == "do_load_truck": h_tn += 2
                elif t_name == "do_unload_truck": h_tn += 2
                else: h_tn += 1

        return max(h_goal, h_tn)

    def _estimate_on_cost(self, crate, surface, state):
        # 1. Already satisfied
        f_id = self.on_ids.get((crate, surface))
        if f_id is not None and (state >> f_id) & 1: return 0

        # 2. Base move cost
        cost = 2 # Lift + Drop

        # 3. Clearance costs
        if not self._is_clear(state, crate): cost += 2
        if not self._is_clear(state, surface): cost += 2

        # 4. Location costs
        loc_c = self._get_loc(state, crate)
        loc_s = self._get_loc(state, surface)
        if loc_c and loc_s and loc_c != loc_s: cost += 3 # Load + Drive + Unload

        return cost

    def _is_clear(self, state, obj):
        f_id = self.clear_ids.get(obj)
        return f_id is not None and (state >> f_id) & 1

    def _is_at(self, state, obj, loc):
        # Generic 'at' check for trucks, hoists, pallets, or crates
        f_id = self.at_ids.get(obj, {}).get(loc)
        if f_id is not None and (state >> f_id) & 1: return True
        f_id = self.truck_at_ids.get(obj, {}).get(loc)
        if f_id is not None and (state >> f_id) & 1: return True
        f_id = self.hoist_at_ids.get(obj, {}).get(loc)
        if f_id is not None and (state >> f_id) & 1: return True
        return False

    def _get_loc(self, state, obj):
        # Direct check
        for loc, f_id in self.at_ids.get(obj, {}).items():
            if (state >> f_id) & 1: return loc
        # Check trucks
        for truck, in_f_id in self.in_ids.get(obj, {}).items():
            if (state >> in_f_id) & 1:
                for loc, t_at_id in self.truck_at_ids.get(truck, {}).items():
                    if (state >> t_at_id) & 1: return loc
        # Check hoists
        for hoist, l_f_id in self.lifting_ids.get(obj, {}).items():
            if (state >> l_f_id) & 1:
                for loc, h_at_id in self.hoist_at_ids.get(hoist, {}).items():
                    if (state >> h_at_id) & 1: return loc
        return None

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

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return task, args
        return name, []
