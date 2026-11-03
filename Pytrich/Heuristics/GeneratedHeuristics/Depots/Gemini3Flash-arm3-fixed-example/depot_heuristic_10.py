"""
Domain-dependent heuristic for the Depots HDDL domain.

1. Bottleneck Analysis:
The core bottleneck in the Depots domain is the multi-stage transport of crates across different locations via trucks. 
This is heavily serialized by the requirement to clear any crates stacked on top of target objects (pallets or other crates)
before those targets can be accessed, moved, or used as a destination.

2. Admissible Lower Bounds:
- Term 1 (Transport Cost): Every 'do_put_on(c, s)' task where crate c is not on surface s requires at least 
  2 actions (Lift, Drop) if c and s are at the same location, and at least 5 actions (Lift, Load, Drive, Unload, Drop) 
  if they are at different locations. This is an unavoidable minimum sequence of operators.
- Term 2 (Clearance Cost): For every surface s that must be cleared (to pick up a crate or drop a crate on it), 
  each crate currently in the stack above s must be moved. Each move requires at least 2 actions (Lift and Load/Drop).
- Term 3 (Network Sum): In HDDL planning, the task network represents the remaining work. Summing the state-aware 
  estimated costs of tasks in the network provides a strong admissible-like guidance for the decomposition search.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic10(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_10=None, name="DepotsGemini3FlashHeuristic10"):
        super().__init__(name=name)
        self.depot_gemini3flash_heuristic_10 = depot_gemini3flash_heuristic_10
        self.at_map = {}           # (obj, place) -> fact_id
        self.on_map = {}           # (crate, surface) -> fact_id
        self.in_map = {}           # (crate, truck) -> fact_id
        self.lifting_map = {}      # (hoist, crate) -> fact_id
        self.clear_map = {}        # surface -> fact_id
        self.on_by_surface = defaultdict(list) # surface -> [(fact_id, crate)]
        self.static_place = {}     # pallet/hoist -> place
        
        self.all_place_names = set()
        self.all_truck_names = set()
        self.all_hoist_names = set()
        self.all_crate_names = set()
        
        self.parsed_tasks = {}     # global_id -> (type_str, args_list)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Pre-process facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'at':
                self.at_map[(args[0], args[1])] = fact.global_id
                self.all_place_names.add(args[1])
            elif pred == 'on':
                self.on_map[(args[0], args[1])] = fact.global_id
                self.on_by_surface[args[1]].append((fact.global_id, args[0]))
                self.all_crate_names.add(args[0])
            elif pred == 'in':
                self.in_map[(args[0], args[1])] = fact.global_id
                self.all_truck_names.add(args[1])
                self.all_crate_names.add(args[0])
            elif pred == 'lifting':
                self.lifting_map[(args[0], args[1])] = fact.global_id
                self.all_hoist_names.add(args[0])
                self.all_crate_names.add(args[1])
            elif pred == 'clear':
                self.clear_map[args[0]] = fact.global_id

        # Find static locations (pallets and hoists don't move)
        state_val = model.initial_state
        idx = 0
        while state_val:
            if state_val & 1:
                fact = model.facts[idx]
                pred, args = self._parse_fact_name(fact.name)
                if pred == 'at' and (args[0].startswith('pallet') or args[0].startswith('hoist')):
                    self.static_place[args[0]] = args[1]
            state_val >>= 1
            idx += 1

        # Pre-parse all possible tasks in the model
        for task in model.abstract_tasks:
            t_type = task.name.split('[')[0].strip('+').strip('-')
            self.parsed_tasks[task.global_id] = (t_type, self._parse_task_args(task.name))
        for op in model.operators:
            t_type = op.name.split('[')[0].strip('+').strip('-')
            self.parsed_tasks[op.global_id] = (t_type, self._parse_task_args(op.name))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _parse_task_args(self, task_name: str):
        bracket_idx = task_name.find("[")
        if bracket_idx != -1:
            return task_name[bracket_idx+1:-1].split(",")
        return []

    def _get_at_place(self, state, obj):
        if obj in self.static_place: return self.static_place[obj]
        for p in self.all_place_names:
            fid = self.at_map.get((obj, p))
            if fid is not None and (state >> fid) & 1: return p
        if obj in self.all_crate_names:
            for t in self.all_truck_names:
                fid = self.in_map.get((obj, t))
                if fid is not None and (state >> fid) & 1:
                    for p in self.all_place_names:
                        tfid = self.at_map.get((t, p))
                        if tfid is not None and (state >> tfid) & 1: return p
            for h in self.all_hoist_names:
                fid = self.lifting_map.get((h, obj))
                if fid is not None and (state >> fid) & 1: return self.static_place.get(h)
        return None

    def _count_crates_above(self, state, s):
        count, curr = 0, s
        while True:
            found = False
            for fact_id, crate in self.on_by_surface[curr]:
                if (state >> fact_id) & 1:
                    count, curr, found = count + 1, crate, True
                    break
            if not found: break
        return count

    def _estimate_task_cost(self, state, task_id):
        info = self.parsed_tasks.get(task_id)
        if not info: return 1
        task_type, args = info
        if task_type == 'nop': return 0
        if task_type == 'do_put_on':
            c, s = args[0], args[1]
            fid = self.on_map.get((c, s))
            if fid is not None and (state >> fid) & 1: return 0
            p1, p2 = self._get_at_place(state, c), self._get_at_place(state, s)
            base = 5 if (p1 and p2 and p1 != p2) else 2
            return base + 2 * (self._count_crates_above(state, c) + self._count_crates_above(state, s))
        if task_type == 'do_clear':
            return 2 * self._count_crates_above(state, args[0])
        if task_type == 'do_get_truck':
            return 1 if self._get_at_place(state, args[0]) != args[1] else 0
        if task_type in ['do_load_truck', 'do_unload_truck', 'Unload', 'Load', 'Lift', 'Drop']: return 1
        return getattr(self.model.get_component(task_id), 'cost', 1)

    def _compute(self, state, task_network):
        return sum(self._estimate_task_cost(state, t.global_id) for t in task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value"
}
