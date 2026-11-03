from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from typing import List, Union

class DepotsGemini3FlashHeuristic4(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots Domain.

    Bottleneck Analysis:
    The bottleneck in the Depots domain is the interaction between logistics (moving crates via trucks)
    and stacking (ordering crates on surfaces using hoists). Work is serialized because a crate 
    can only be in one place at a time, trucks have a limited capacity for movement, and 
    stacking constraints require clearing surfaces before usage.

    Admissible Lower Bounds:
    1. Goal Stacking/Logistics Cost: For each crate not correctly placed in its goal tower, we calculate 
       the minimum required actions (Lift, Load, Drive, Unload, Drop) to reach its target destination. 
       This is a lower bound because these physical steps are unavoidable.
    2. Obstruction Clearance Cost: If a crate or its target surface is buried, extra Lift actions are 
       required to clear the path. 
    3. Task Network Cost: Every non-trivial task in the network eventually expands to at least one action.
    """

    def __init__(self, depot_gemini3flash_heuristic_4="depots_heuristic_4"):
        super().__init__(name=depot_gemini3flash_heuristic_4)
        self.at_ids = {}        # (locatable, place) -> id
        self.on_ids = {}        # (crate, surface) -> id
        self.in_ids = {}        # (crate, truck) -> id
        self.lifting_ids = {}   # (hoist, crate) -> id
        self.clear_ids = {}     # surface -> id
        
        self.goal_on = {}       # crate -> target_surface
        self.surface_to_place = {} # surface -> place
        
        self.place_names = []
        self.truck_names = []
        self.crate_names = []
        self.pallet_names = []
        self.hoist_names = []

        self.crate_at_data = {} # crate -> [(place_id, place_name)]
        self.crate_in_data = {} # crate -> [(truck_id, truck_name)]
        self.crate_lift_data = {} # crate -> [(hoist_id, hoist_name)]
        self.truck_at_data = {} # truck -> [(place_id, place_name)]

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse Facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                self.at_ids[(args[0], args[1])] = fact.global_id
                if args[0] not in self.surface_to_place:
                    self.surface_to_place[args[0]] = args[1]
            elif pred == "on":
                self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "in":
                self.in_ids[(args[0], args[1])] = fact.global_id
            elif pred == "lifting":
                self.lifting_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = fact.global_id

        # Collect Names
        for f_name in [f.name for f in model.facts]:
            pred, args = self._parse_fact_name(f_name)
            if not pred: continue
            for arg in args:
                if "depot" in arg or "distributor" in arg:
                    if arg not in self.place_names: self.place_names.append(arg)
                elif "truck" in arg:
                    if arg not in self.truck_names: self.truck_names.append(arg)
                elif "hoist" in arg:
                    if arg not in self.hoist_names: self.hoist_names.append(arg)
                elif "pallet" in arg:
                    if arg not in self.pallet_names: self.pallet_names.append(arg)
                elif "crate" in arg:
                    if arg not in self.crate_names: self.crate_names.append(arg)

        # Precompute Data Structures for faster bit-shifting in __call__
        for c in self.crate_names:
            self.crate_at_data[c] = [(self.at_ids[(c, p)], p) for p in self.place_names if (c, p) in self.at_ids]
            self.crate_in_data[c] = [(self.in_ids[(c, t)], t) for t in self.truck_names if (c, t) in self.in_ids]
            self.crate_lift_data[c] = [(self.lifting_ids[(h, c)], h) for h in self.hoist_names if (h, c) in self.lifting_ids]
        
        for t in self.truck_names:
            self.truck_at_data[t] = [(self.at_ids[(t, p)], p) for p in self.place_names if (t, p) in self.at_ids]

        # Parse Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network: List) -> int:
        goal_cost = 0
        memo_correct = {}

        def is_correctly_placed(c):
            if c in memo_correct: return memo_correct[c]
            target_s = self.goal_on.get(c)
            if not target_s: return True
            
            fid = self.on_ids.get((c, target_s), -1)
            if fid == -1 or not ((state >> fid) & 1):
                memo_correct[c] = False
                return False
            
            if target_s in self.pallet_names:
                memo_correct[c] = True
                return True
            
            res = is_correctly_placed(target_s)
            memo_correct[c] = res
            return res

        # Term 1: Stacking and Logistics
        for c in self.crate_names:
            if c in self.goal_on and not is_correctly_placed(c):
                target_s = self.goal_on[c]
                target_p = self.surface_to_place.get(target_s)
                
                # Base Cost: at least Lift + Drop = 2
                c_cost = 2
                
                # Current location status
                curr_p = None
                in_truck = None
                lifting = False
                
                for fid, p_name in self.crate_at_data.get(c, []):
                    if (state >> fid) & 1: curr_p = p_name; break
                if not curr_p:
                    for fid, t_name in self.crate_in_data.get(c, []):
                        if (state >> fid) & 1: in_truck = t_name; break
                if not curr_p and not in_truck:
                    for fid, h_name in self.crate_lift_data.get(c, []):
                        if (state >> fid) & 1: lifting = True; break

                if lifting:
                    c_cost = 1 # Just need Drop
                elif in_truck:
                    # Need Unload + Drop
                    c_cost = 2
                    # Check if truck needs to move
                    truck_at_target = False
                    for fid, p_name in self.truck_at_data.get(in_truck, []):
                        if (state >> fid) & 1 and p_name == target_p:
                            truck_at_target = True; break
                    if not truck_at_target: c_cost += 1 # Drive
                elif curr_p:
                    if curr_p != target_p:
                        # Lift + Load + Drive + Unload + Drop = 5
                        c_cost = 5
                
                # Clearance penalties (Buried)
                c_clear_id = self.clear_ids.get(c, -1)
                s_clear_id = self.clear_ids.get(target_s, -1)
                if c_clear_id != -1 and not ((state >> c_clear_id) & 1):
                    c_cost += 1
                if s_clear_id != -1 and not ((state >> s_clear_id) & 1):
                    c_cost += 1
                
                goal_cost += c_cost

        # Term 2: Task Network Progress
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                tn_cost += task.cost
            elif "do_put_on" in task.name:
                tn_cost += 1 # Abstract tasks take at least 1 op
            elif "do_clear" in task.name or "do_lift" in task.name:
                tn_cost += 1

        return max(goal_cost, tn_cost)

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
