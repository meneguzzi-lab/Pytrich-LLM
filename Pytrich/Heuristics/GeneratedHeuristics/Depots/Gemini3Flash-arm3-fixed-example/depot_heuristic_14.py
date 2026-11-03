"""
Domain-dependent heuristic for the Depots HDDL domain.

1. Bottleneck Analysis:
The primary bottleneck in the Depots domain is the sequential nature of crate movement
and stack management. A crate cannot be moved until it is clear, and it cannot be placed
on a surface until that surface is clear. Furthermore, movement between different places
requires a truck, which can only be at one place at a time and must follow a 
Lift -> Load -> Drive -> Unload -> Drop sequence.

2. Admissible Lower Bounds:
- Term 1 (Task Displacement): Each 'do_put_on(c, s)' task in the network requires at 
  minimum a Lift and a Drop (2 actions). If the crate and target are at different 
  locations, it also requires Load, Drive, and Unload (+3 actions).
- Term 2 (Stack Clearing): If a crate or a target surface is not clear, at least 
  two additional actions (Lift + Load/Drop) are required for every crate blocking it.
- Term 3 (Logistics): Any 'do_get_truck' or 'do_lift_crate' task represents at least 
  one unavoidable action in the decomposition chain.

Admissibility Sketch: These terms represent the absolute minimum number of operators 
required to decompose the current task network into a sequence of actions. Clearing 
a stack of height k requires at least 2k actions. Inter-location transport 
requires a fixed sequence of 5 actions.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic14(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_14=None, name="DepotsGemini3FlashHeuristic14"):
        super().__init__(name=name)
        self.on_fact_ids = {}      # crate -> surface -> global_id
        self.at_fact_ids = {}      # locatable -> place -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.in_fact_ids = {}      # crate -> truck -> global_id
        self.lifting_fact_ids = {} # hoist -> crate -> global_id
        self.task_metadata = {}    # global_id -> (task_type, args)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Preprocess facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                c, s = args[0], args[1]
                if c not in self.on_fact_ids: self.on_fact_ids[c] = {}
                self.on_fact_ids[c][s] = f.global_id
            elif pred == "at":
                loc, p = args[0], args[1]
                if loc not in self.at_fact_ids: self.at_fact_ids[loc] = {}
                self.at_fact_ids[loc][p] = f.global_id
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "in":
                c, t = args[0], args[1]
                if c not in self.in_fact_ids: self.in_fact_ids[c] = {}
                self.in_fact_ids[c][t] = f.global_id
            elif pred == "lifting":
                h, c = args[0], args[1]
                if c not in self.lifting_fact_ids: self.lifting_fact_ids[c] = {}
                self.lifting_fact_ids[c][h] = f.global_id

        # 2. Preprocess tasks
        for task in self.model.abstract_tasks:
            self.task_metadata[task.global_id] = self._parse_task_string(task.name)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def _parse_task_string(self, name: str):
        if "[" in name:
            base = name[:name.find("[")]
            args = name[name.find("[")+1:-1].split(",")
            return base, args
        return name, []

    def _get_loc(self, state, item):
        if item in self.at_fact_ids:
            for p, fid in self.at_fact_ids[item].items():
                if (state >> fid) & 1: return p
        if item in self.in_fact_ids:
            for t, fid in self.in_fact_ids[item].items():
                if (state >> fid) & 1: return self._get_loc(state, t)
        if item in self.lifting_fact_ids:
            for h, fid in self.lifting_fact_ids[item].items():
                if (state >> fid) & 1: return self._get_loc(state, h)
        return None

    def _compute(self, state, task_network):
        h_total = 0
        for task in task_network:
            if hasattr(task, 'global_id') and task.global_id in self.task_metadata:
                task_type, args = self.task_metadata[task.global_id]
                
                if task_type == "do_put_on":
                    c, s = args[0], args[1]
                    # Is it already there?
                    already_on = False
                    if c in self.on_fact_ids and s in self.on_fact_ids[c]:
                        if (state >> self.on_fact_ids[c][s]) & 1: already_on = True
                    
                    if not already_on:
                        cost = 2 # Lift + Drop
                        loc_c = self._get_loc(state, c)
                        loc_s = self._get_loc(state, s)
                        if loc_c and loc_s and loc_c != loc_s:
                            cost += 3 # Load + Drive + Unload
                        
                        # Penalty if not clear
                        if c in self.clear_fact_ids and not ((state >> self.clear_fact_ids[c]) & 1):
                            cost += 2
                        if s in self.clear_fact_ids and not ((state >> self.clear_fact_ids[s]) & 1):
                            cost += 2
                        h_total += cost

                elif task_type == "do_clear":
                    s = args[0]
                    if s in self.clear_fact_ids and not ((state >> self.clear_fact_ids[s]) & 1):
                        h_total += 2
                
                elif task_type in ["do_load_truck", "do_unload_truck"]:
                    h_total += 4
                elif task_type in ["do_get_truck", "do_lift_crate"]:
                    h_total += 1
                else:
                    h_total += 1
            else:
                # Primitive action
                h_total += 1
        
        return h_total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
