from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from typing import List, Union

class DepotsGemini3FlashHeuristic12(Heuristic):
    """
    Heuristic for the Depots domain.
    Bottleneck: Hoist availability and Crate accessibility. Hoists are required for all stacking/unstacking,
    and crates must be cleared before being moved.

    Independent Admissible Lower Bounds:
    1. Task Network Cost: Each task in the network has a minimum number of actions to achieve.
       A 'do_put_on' task costs at least 2 (Lift, Drop) if same-place, or 5 (Lift, Load, Drive, Unload, Drop) if inter-place.
    2. Stacking Constraints: If a surface is not clear, at least 2 actions (Lift/Drop) are needed to move the blocking crate.
    3. Truck Logistics: Every crate requiring a change of location requires at least one truck movement sequence.
    """

    def __init__(self, depot_gemini3flash_heuristic_12="DepotsGemini3FlashHeuristic12"):
        super().__init__(depot_gemini3flash_heuristic_12)
        self.at_ids = {}      # (obj, place) -> global_id
        self.in_ids = {}      # (crate, truck) -> global_id
        self.on_ids = {}      # (crate, surface) -> global_id
        self.clear_ids = {}   # surface -> global_id
        self.lifting_ids = {} # crate -> {hoist -> global_id}

        self.place_names = []
        self.truck_names = []
        self.pallets = set()
        self.hoist_to_place = {}
        self.surface_to_place = {} # pallet -> place
        self.task_data = {}   # global_id -> (name, args)
        self.goals_on = []

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            bracket_idx = name.find("[")
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")]
            return predicate, args
        parts = name.split()
        if len(parts) > 1:
            return parts[0], parts[1:]
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_name(g_name)
            if pred == 'on' and len(args) >= 2:
                self.goals_on.append((args[0], args[1]))

        # 2. Parse facts for indexing
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == 'at' and len(args) >= 2:
                self.at_ids[(args[0], args[1])] = f.global_id
                # Identify places, trucks, hoists from fact signatures
                if "distributor" in args[1] or "depot" in args[1]:
                    if args[1] not in self.place_names: self.place_names.append(args[1])
                if "truck" in args[0]:
                    if args[0] not in self.truck_names: self.truck_names.append(args[0])
                if "hoist" in args[0]:
                    self.hoist_to_place[args[0]] = args[1]
                if "pallet" in args[0]:
                    self.pallets.add(args[0])
                    self.surface_to_place[args[0]] = args[1]
            elif pred == 'in' and len(args) >= 2:
                self.in_ids[(args[0], args[1])] = f.global_id
            elif pred == 'on' and len(args) >= 2:
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == 'clear' and len(args) >= 1:
                self.clear_ids[args[0]] = f.global_id
            elif pred == 'lifting' and len(args) >= 2:
                if args[1] not in self.lifting_ids: self.lifting_ids[args[1]] = {}
                self.lifting_ids[args[1]][args[0]] = f.global_id

        # 3. Pre-map all tasks and operators for fast lookup
        for abt in model.abstract_tasks:
            self.task_data[abt.global_id] = self._parse_name(abt.name)
        for op in model.operators:
            self.task_data[op.global_id] = (op.name, [])

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _get_crate_place(self, c_name, state):
        for p_name in self.place_names:
            f_id = self.at_ids.get((c_name, p_name))
            if f_id is not None and (state >> f_id) & 1:
                return p_name
        # Check if lifted
        if c_name in self.lifting_ids:
            for h_name, f_id in self.lifting_ids[c_name].items():
                if (state >> f_id) & 1:
                    return self.hoist_to_place.get(h_name)
        return None

    def _get_crate_truck(self, c_name, state):
        for t_name in self.truck_names:
            f_id = self.in_ids.get((c_name, t_name))
            if f_id is not None and (state >> f_id) & 1:
                return t_name
        return None

    def _get_surface_place(self, s_name, state):
        if s_name in self.surface_to_place: return self.surface_to_place[s_name]
        return self._get_crate_place(s_name, state)

    def _compute_put_on_cost(self, c, s, state):
        # Goal already satisfied?
        on_f_id = self.on_ids.get((c, s))
        if on_f_id is not None and (state >> on_f_id) & 1: return 0

        c_place = self._get_crate_place(c, state)
        s_place = self._get_surface_place(s, state)

        cost = 2 # Default Lift/Drop
        if c_place:
            if s_place and s_place != c_place:
                cost = 5 # Lift, Load, Drive, Unload, Drop
        else:
            truck = self._get_crate_truck(c, state)
            if truck:
                # Assume needs Drive if truck place unknown or wrong
                cost = 3 # Drive, Unload, Drop

        # Surface Clear check (min actions to clear)
        if c in self.clear_ids and not (state >> self.clear_ids[c]) & 1: cost += 2
        if s in self.clear_ids and not (state >> self.clear_ids[s]) & 1: cost += 2
        return cost

    def _compute(self, state, task_network):
        h = 0
        if not task_network:
            for c, s in self.goals_on:
                h += self._compute_put_on_cost(c, s, state)
            return h

        for task in task_network:
            data = self.task_data.get(task.global_id)
            if data:
                name, args = data
                if name == 'do_put_on' and len(args) >= 2:
                    h += self._compute_put_on_cost(args[0], args[1], state)
                elif name == 'do_clear' and len(args) >= 1:
                    f_id = self.clear_ids.get(args[0])
                    if f_id is not None and not (state >> f_id) & 1: h += 2
                elif name == 'do_get_truck' and len(args) >= 2:
                    # check at(t, p)
                    f_id = self.at_ids.get((args[0], args[1]))
                    if f_id is not None and not (state >> f_id) & 1: h += 1
                elif name == 'do_lift_crate': h += 1
                elif name == 'do_load_truck' or name == 'do_unload_truck': h += 2
                else:
                    h += getattr(task, 'cost', 1)
            else:
                h += getattr(task, 'cost', 1)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
