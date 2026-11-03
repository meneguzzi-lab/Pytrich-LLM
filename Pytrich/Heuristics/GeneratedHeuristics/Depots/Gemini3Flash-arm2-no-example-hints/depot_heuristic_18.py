from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic18(Heuristic):
    """
    Heuristic for the Depots domain.
    Bottleneck: The serialization of hoist and truck resources, specifically clearing crates off 
    surfaces and physical transport (Lift-Load-Drive-Unload-Drop).

    Admissibility Sketches:
    1. Goal-based Term: For every 'on(c, s)' goal not satisfied, we need at least a Lift and a Drop (cost 2). 
       If the current location of 'c' differs from 's', we need Lift-Load-Drive-Unload-Drop (cost 5). 
       This is a lower bound because these operations are physically necessary.
    2. Task-based Term: Each task in the network has a minimum expansion cost. 'do_put_on' is 
       either a nop, or requires 2 ops (same place) or 5 ops (different place). 'do_clear' requires 
       2 ops for every crate currently blocking the surface.
    """

    def __init__(self, depot_gemini3flash_heuristic_18="depot_gemini3flash_heuristic_18"):
        super().__init__(name=depot_gemini3flash_heuristic_18)
        self.on_id = {} # (crate, surface) -> global_id
        self.clear_id = {} # surface -> global_id
        self.at_id = {} # (locatable, place) -> global_id
        self.in_id = {} # (crate, truck) -> global_id
        self.lifting_id = {} # (hoist, crate) -> global_id
        
        self.on_lookup = {} # surface -> [(crate, fact_id)]
        self.on_crate_to_surface_lookup = {} # crate -> [(surface, fact_id)]
        self.at_lookup = {} # locatable -> [(place, fact_id)]
        self.in_lookup = {} # crate -> [(truck, fact_id)]
        self.lifting_lookup = {} # hoist -> [(crate, fact_id)]
        
        self.obj_to_place = {} # pallet/hoist -> static place
        self.goals_on = [] # [(crate, surface)]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args, _ = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_id[(args[0], args[1])] = f.global_id
                self.on_lookup.setdefault(args[1], []).append((args[0], f.global_id))
                self.on_crate_to_surface_lookup.setdefault(args[0], []).append((args[1], f.global_id))
            elif pred == "clear":
                self.clear_id[args[0]] = f.global_id
            elif pred == "at":
                self.at_id[(args[0], args[1])] = f.global_id
                self.at_lookup.setdefault(args[0], []).append((args[1], f.global_id))
            elif pred == "in":
                self.in_id[(args[0], args[1])] = f.global_id
                self.in_lookup.setdefault(args[0], []).append((args[1], f.global_id))
            elif pred == "lifting":
                self.lifting_id[(args[0], args[1])] = f.global_id
                self.lifting_lookup.setdefault(args[1], []).append((args[0], f.global_id))

        for op in model.operators:
            if op.name.startswith("Drop") or op.name.startswith("Lift"):
                parts = self._parse_op_name(op.name)
                if len(parts) >= 4:
                    hoist, crate, surface, place = parts[0], parts[1], parts[2], parts[3]
                    self.obj_to_place[surface] = place
                    self.obj_to_place[hoist] = place

        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args, _ = self._parse_fact_name(gname)
            if pred == "on":
                self.goals_on.append((args[0], args[1]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h_goal = 0
        for crate, surface in self.goals_on:
            fid = self.on_id.get((crate, surface))
            if fid is not None and (state >> fid) & 1:
                continue
            
            c_place = self._get_crate_place(crate, state)
            s_place = self.obj_to_place.get(surface)
            
            if c_place and s_place:
                if c_place == s_place:
                    h_goal += 2
                else:
                    h_goal += 5
            else:
                h_goal += 2

        h_tn = 0
        for task in task_network:
            if hasattr(task, 'name'):
                if "do_put_on" in task.name:
                    parts = self._parse_task_name(task.name)
                    if len(parts) >= 2:
                        c, s = parts[0], parts[1]
                        on_fid = self.on_id.get((c, s))
                        if on_fid is not None and (state >> on_fid) & 1: continue
                        c_place = self._get_crate_place(c, state)
                        s_place = self.obj_to_place.get(s)
                        cost = 5 if (c_place and s_place and c_place != s_place) else 2
                        h_tn += cost + self._estimate_clear_cost(c, state) + self._estimate_clear_cost(s, state)
                elif "do_clear" in task.name:
                    parts = self._parse_task_name(task.name)
                    if len(parts) >= 1:
                        h_tn += self._estimate_clear_cost(parts[0], state)
                elif not "nop" in task.name:
                    h_tn += getattr(task, 'cost', 1)

        return max(h_goal, h_tn)

    def _get_crate_place(self, crate, state):
        if crate in self.at_lookup:
            for place, fid in self.at_lookup[crate]:
                if (state >> fid) & 1: return place
        if crate in self.in_lookup:
            for truck, fid in self.in_lookup[crate]:
                if (state >> fid) & 1:
                    for p, tfid in self.at_lookup.get(truck, []):
                        if (state >> tfid) & 1: return p
        if crate in self.lifting_lookup:
            for hoist, fid in self.lifting_lookup[crate]:
                if (state >> fid) & 1: return self.obj_to_place.get(hoist)
        if crate in self.on_crate_to_surface_lookup:
            for surface, fid in self.on_crate_to_surface_lookup[crate]:
                if (state >> fid) & 1: return self.obj_to_place.get(surface)
        return None

    def _estimate_clear_cost(self, surface, state):
        cost = 0
        curr = surface
        while True:
            found = False
            if curr in self.on_lookup:
                for crate, fid in self.on_lookup[curr]:
                    if (state >> fid) & 1:
                        cost += 2
                        curr = crate
                        found = True
                        break
            if not found: break
        return cost

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args, prefix
        return None, [], None

    def _parse_op_name(self, name):
        # Format: Name[arg1,arg2,...]
        match = re.search(r'\[(.*?)\]', name)
        if match:
            return match.group(1).split(',')
        return []

    def _parse_task_name(self, name):
        # Format: Name arg1 arg2 ... (or Name(arg1, arg2))
        name = name.replace("(", " ").replace(")", " ").replace(",", " ")
        return name.split()[1:]
