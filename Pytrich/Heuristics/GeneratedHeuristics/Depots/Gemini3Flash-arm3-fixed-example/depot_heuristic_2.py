import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    1. Bottleneck Analysis:
       The primary bottleneck is stack manipulation (clearing crates to reach a target) 
       combined with logistical transport (moving crates via trucks between locations).
       A crate at the bottom of a stack at location A that needs to be at the bottom 
       of a stack at location B requires a long sequence of clear, lift, load, 
       drive, unload, clear, and drop operations.

    2. Admissible Lower Bounds:
       - Task Network Minimal Cost: Each grounded task in the network has a minimum 
         number of actions to completion. A 'do_put_on' task for a crate at the wrong 
         location requires at least 5 actions (Lift, Load, Drive, Unload, Drop). 
         If at the right location, it requires at least 2 (Lift, Drop).
       - Clearing Costs: If a crate or target surface is not clear, at least one 
         crate must be moved per level of depth, adding significant cost.

    3. Preprocessing:
       We pre-parse all grounded tasks and facts in the model to enable O(1) 
       lookups during search. We map crates and surfaces to their possible 'at', 
       'on', and 'in' fact IDs.
    """

    def __init__(self, depot_gemini3flash_heuristic_2=None, name="DepotsGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.task_data = {}  # global_id -> (predicate, args)
        self.at_fact_ids = {}  # obj -> [(place, global_id)]
        self.in_fact_ids = {}  # crate -> [(truck, global_id)]
        self.lifting_fact_ids = {}  # hoist -> [(crate, global_id)]
        self.on_fact_ids = {}  # crate -> {surface_name: global_id}
        self.clear_fact_ids = {}  # surface -> global_id
        self.goal_on = {}  # crate -> surface_name

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Pre-parse all grounded facts
        for fact in model.facts:
            pred, args = self._parse_name(fact.name)
            if not pred: continue

            if pred == "at":
                obj, place = args[0], args[1]
                if obj not in self.at_fact_ids: self.at_fact_ids[obj] = []
                self.at_fact_ids[obj].append((place, fact.global_id))
            elif pred == "in":
                crate, truck = args[0], args[1]
                if crate not in self.in_fact_ids: self.in_fact_ids[crate] = []
                self.in_fact_ids[crate].append((truck, fact.global_id))
            elif pred == "lifting":
                hoist, crate = args[0], args[1]
                if hoist not in self.lifting_fact_ids: self.lifting_fact_ids[hoist] = []
                self.lifting_fact_ids[hoist].append((crate, fact.global_id))
            elif pred == "on":
                crate, surface = args[0], args[1]
                if crate not in self.on_fact_ids: self.on_fact_ids[crate] = {}
                self.on_fact_ids[crate][surface] = fact.global_id
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = fact.global_id

        # Pre-parse all grounded tasks (Abstract and Operators)
        for t in model.abstract_tasks:
            self.task_data[t.global_id] = self._parse_name(t.name)
        for o in model.operators:
            self.task_data[o.global_id] = self._parse_name(o.name)

        # Parse goals
        goal_facts = model.state_explicit_repr(model.goals)
        for g_name in goal_facts:
            pred, args = self._parse_name(g_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return None, []

    def _get_loc(self, state, obj):
        # Check direct 'at' facts
        for place, fid in self.at_fact_ids.get(obj, []):
            if (state >> fid) & 1: return place
        # If crate, check if in a truck or being lifted
        for truck, fid in self.in_fact_ids.get(obj, []):
            if (state >> fid) & 1: return self._get_loc(state, truck)
        for hoist, fid in self.lifting_fact_ids.get(obj, []):
            if (state >> fid) & 1: return self._get_loc(state, hoist)
        return None

    def _is_on(self, state, crate, surface):
        fid = self.on_fact_ids.get(crate, {}).get(surface)
        return (state >> fid) & 1 if fid is not None else False

    def _is_clear(self, state, surface):
        fid = self.clear_fact_ids.get(surface)
        return (state >> fid) & 1 if fid is not None else False

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            data = self.task_data.get(task.global_id)
            if not data: continue
            pred, args = data

            if pred == "do_put_on":
                c, s = args[0], args[1]
                if not self._is_on(state, c, s):
                    loc_c = self._get_loc(state, c)
                    loc_s = self._get_loc(state, s)
                    if loc_c and loc_s and loc_c != loc_s:
                        h += 5
                    else:
                        h += 2
                    # Penalty for blocked surfaces
                    if not self._is_clear(state, c): h += 2
                    if not self._is_clear(state, s): h += 2
            elif pred == "do_clear":
                if not self._is_clear(state, args[0]): h += 2
            elif pred == "do_load_truck":
                h += 2
            elif pred == "do_unload_truck":
                h += 2
            elif pred in ["Lift", "Drop", "Load", "Unload", "Drive"]:
                h += 1
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
