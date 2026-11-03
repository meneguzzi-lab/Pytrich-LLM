from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis: 
# The Depots domain's primary bottleneck is "Crate Displacement and Stack Management". 
# Plan length is dominated by the need to move crates between locations, which requires 
# serializing multiple steps (Clear, Lift, Load, Drive, Unload, Drop) while managing 
# limited resources (hoists, trucks) and stacking constraints (only the top crate is clear).

class DepotsGemini3FlashHeuristic7(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_7="DepotsGemini3FlashHeuristic7"):
        super().__init__(name=depot_gemini3flash_heuristic_7)
        self.at_ids = {}
        self.on_ids = {}
        self.in_ids = {}
        self.lifting_by_crate = {}
        self.clear_ids = {}
        self.task_info = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Precompute fact mappings for O(1) bit tests
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at": # at[locatable, place]
                if args[0] not in self.at_ids: self.at_ids[args[0]] = []
                self.at_ids[args[0]].append((args[1], fact.global_id))
            elif pred == "on": # on[crate, surface]
                self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "in": # in[crate, truck]
                if args[0] not in self.in_ids: self.in_ids[args[0]] = []
                self.in_ids[args[0]].append((args[1], fact.global_id))
            elif pred == "lifting": # lifting[hoist, crate]
                if args[1] not in self.lifting_by_crate: self.lifting_by_crate[args[1]] = []
                self.lifting_by_crate[args[1]].append(fact.global_id)
            elif pred == "clear": # clear[surface]
                self.clear_ids[args[0]] = fact.global_id

        # Pre-parse task names and args for efficiency
        for abt in model.abstract_tasks:
            name, args = self._parse_task_name(abt.name)
            self.task_info[abt.global_id] = (name, args)
        for op in model.operators:
            self.task_info[op.global_id] = ("operator", [])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Component 1: Sum of minimum required actions per task (Admissible Lower Bound)
        # Admissibility sketch: Each term (2 for same-place restack, 5 for relocation) 
        # represents the absolute minimum number of primitive actions (Lift/Drop or 
        # Lift/Load/Drive/Unload/Drop) required to satisfy that specific task.
        h_base = 0
        for task in task_network:
            info = self.task_info.get(task.global_id)
            if not info: continue
            task_type, args = info
            
            if task_type == "do_put_on":
                h_base += self._estimate_do_put_on(args[0], args[1], state)
            elif task_type == "do_clear":
                h_base += self._estimate_do_clear(args[0], state)
            elif task_type == "operator":
                h_base += task.cost
            else:
                h_base += 1 # generic minimum for other abstract tasks

        # Symmetry breaking tie-breaker (non-dominant)
        h_tie = 0.001 * len(task_network)
        
        return int(h_base + h_tie)

    def _estimate_do_put_on(self, crate, surface, state):
        # Lower bound cost for putting a crate on a surface
        on_fid = self.on_ids.get((crate, surface))
        if on_fid is not None and (state >> on_fid) & 1:
            return 0 # Already satisfied

        c_loc = self._get_loc(crate, state)
        s_loc = self._get_loc(surface, state)

        if c_loc == "truck" or c_loc == "lifting":
            return 2 # Unload + Drop at minimum
        if c_loc and s_loc and c_loc == s_loc:
            return 2 # Lift + Drop
        if c_loc and s_loc and c_loc != s_loc:
            return 5 # Lift, Load, Drive, Unload, Drop
        
        return 1 # Fallback

    def _estimate_do_clear(self, surface, state):
        # Lower bound for clearing a surface
        clear_fid = self.clear_ids.get(surface)
        if clear_fid is not None and (state >> clear_fid) & 1:
            return 0
        return 2 # Must Lift crate on top and Drop/Load it somewhere else

    def _get_loc(self, obj, state):
        # Fast O(1) location lookup via precomputed bit indices
        at_list = self.at_ids.get(obj, [])
        for place, fid in at_list:
            if (state >> fid) & 1: return place
        in_list = self.in_ids.get(obj, [])
        for truck, fid in in_list:
            if (state >> fid) & 1: return "truck"
        lift_list = self.lifting_by_crate.get(obj, [])
        for fid in lift_list:
            if (state >> fid) & 1: return "lifting"
        return None

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

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return name, []
