"""
Domain-dependent heuristic for the Depots planning domain.

Bottleneck Analysis:
The primary bottleneck in Depots is the synchronization of hoists, trucks, and crates across locations. 
Each crate manipulation requires a clear surface and an available hoist, and movement between 
locations requires truck positioning. The serialization of these resources (one hoist per crate, 
one truck per transport) is the main constraint.

Admissible Lower Bounds:
1. Manipulation Actions: Every crate not at its goal surface must be lifted and dropped (at least 2 actions). 
   If the target surface is at a different location, it also requires being loaded and unloaded (at least 4 actions total).
2. Logistics (Travel): If any crate needs to move from Place A to Place B, at least one Drive action is required.
3. Task Network Decomposition: Each abstract task in the HTN network requires a minimum number of actions 
   to reach a primitive state, based on the current state (e.g., do_put_on remote costs more than do_put_on local).

"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic8(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_8=None, name="DepotsGemini3FlashHeuristic8"):
        super().__init__(name=name)
        self.on_goals = {}  # crate -> target surface
        self.at_fact_ids = []
        self.at_lookup = {} # global_id -> (obj, place)
        self.in_fact_ids = []
        self.in_lookup = {} # global_id -> (crate, truck)
        self.on_fact_ids = []
        self.on_lookup = {} # global_id -> (crate, surface)
        self.clear_fact_ids = []
        self.clear_lookup = {} # global_id -> (surface)
        
    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Pre-process Goal State
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_names:
                pred, args = self._parse_fact_name(name)
                if pred == "on" and len(args) == 2:
                    self.on_goals[args[0]] = args[1]
                    
        # Pre-process Fact Mappings
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_fact_ids.append(f.global_id)
                self.at_lookup[f.global_id] = (args[0], args[1])
            elif pred == "in":
                self.in_fact_ids.append(f.global_id)
                self.in_lookup[f.global_id] = (args[0], args[1])
            elif pred == "on":
                self.on_fact_ids.append(f.global_id)
                self.on_lookup[f.global_id] = (args[0], args[1])
            elif pred == "clear":
                self.clear_fact_ids.append(f.global_id)
                self.clear_lookup[f.global_id] = args[0]

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].split()
            if inner:
                return inner[0], inner[1:]
        elif "[" in name and name.endswith("]"):
            # Handle grounded task names like do_put_on[c1,s1]
            bracket_idx = name.find("[")
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _compute(self, state, task_network):
        # 1. State extraction
        curr_at = {} # obj -> place
        for fid in self.at_fact_ids:
            if (state >> fid) & 1:
                obj, p = self.at_lookup[fid]
                curr_at[obj] = p
        
        curr_in = {} # crate -> truck
        for fid in self.in_fact_ids:
            if (state >> fid) & 1:
                c, t = self.in_lookup[fid]
                curr_in[c] = t

        curr_on = {} # crate -> surface
        for fid in self.on_fact_ids:
            if (state >> fid) & 1:
                c, s = self.on_lookup[fid]
                curr_on[c] = s

        is_clear = set()
        for fid in self.clear_fact_ids:
            if (state >> fid) & 1:
                is_clear.add(self.clear_lookup[fid])

        # 2. Task-based Cost Estimate
        h_tasks = 0
        for task in task_network:
            if hasattr(task, 'cost'): # Primitive operator
                h_tasks += task.cost
                continue
            
            pred, args = self._parse_fact_name(task.name)
            if not pred: continue
            
            if pred == "do_put_on":
                c, s = args[0], args[1]
                if curr_on.get(c) == s: continue
                p_c = curr_at.get(c)
                p_s = curr_at.get(s)
                if p_c and p_s and p_c == p_s: h_tasks += 2 # Lift, Drop
                else: h_tasks += 5 # Lift, Load, Drive, Unload, Drop
            elif pred == "do_clear":
                s = args[0]
                if s not in is_clear: h_tasks += 2
            elif pred == "do_get_truck":
                t, p = args[0], args[1]
                if curr_at.get(t) != p: h_tasks += 1
            elif pred == "do_lift_crate":
                h_tasks += 1
            elif pred == "do_load_truck":
                h_tasks += 2 # Lift + Load
            elif pred == "do_unload_truck":
                h_tasks += 2 # Unload + Drop

        # 3. Goal-based Cost Estimate (Admissible Backup)
        h_goals = 0
        drive_required = 0
        for crate, target_s in self.on_goals.items():
            if curr_on.get(crate) != target_s:
                p_c = curr_at.get(crate)
                p_s = curr_at.get(target_s)
                if p_c and p_s and p_c == p_s:
                    h_goals += 2
                else:
                    h_goals += 4
                    drive_required = 1
        h_goals += drive_required

        # Return max to remain admissible, plus sub-unit tie-breaker
        # Task network length is a good signal of remaining complexity in HTN
        return max(h_tasks, h_goals) + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
