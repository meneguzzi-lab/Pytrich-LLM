import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    1. Bottleneck Analysis:
       The primary bottleneck in the Depots domain is the serialization of crate manipulation 
       due to stacking constraints (Blocksworld) and the limited transport capacity and 
       location-dependency of trucks and hoists (Logistics).

    2. Admissible Lower Bounds:
       - do_put_on(c, s): Requires at least 2 actions (Lift, Drop) if the crate and surface are at 
         the same place, and at least 5 actions (Lift, Load, Drive, Unload, Drop) if they are at 
         different places.
       - do_clear(s): Requires at least 2 actions to move the crate currently on top of surface s.
       - do_get_truck(t, p): Requires at least 1 action (Drive) if the truck is not already at place p.
       - Operator costs: The grounded cost of individual actions (Lift, Drop, Load, Unload, Drive).
       Summing these over the current task network is admissible because every task in the network 
       must be eventually decomposed and executed.
    """

    def __init__(self, depot_gemini3flash_heuristic_5=None, name="DepotsGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.at_map = {}          # (item, place) -> fact_id
        self.on_map = {}          # (crate, surface) -> fact_id
        self.in_map = {}          # (crate, truck) -> fact_id
        self.lifting_map = {}     # (hoist, crate) -> fact_id
        self.clear_map = {}       # surface -> fact_id
        
        self.crates = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()
        self.surfaces = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                self.at_map[(args[0], args[1])] = fact.global_id
                self.places.add(args[1])
            elif pred == "on":
                self.on_map[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[1])
            elif pred == "in":
                self.in_map[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
                self.trucks.add(args[1])
            elif pred == "lifting":
                self.lifting_map[(args[0], args[1])] = fact.global_id
                self.hoists.add(args[0])
                self.crates.add(args[1])
            elif pred == "clear":
                self.clear_map[args[0]] = fact.global_id
                self.surfaces.add(args[0])

        h_value = self._calculate_heuristic(initial_node.state, initial_node.task_network)
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

    def _parse_task_args(self, name):
        name = name.strip().replace('(', '').replace(')', '')
        parts = name.split()
        if not parts:
            return None, []
        return parts[0], parts[1:]

    def _get_item_place(self, state, item):
        for p in self.places:
            fid = self.at_map.get((item, p))
            if fid is not None and (state >> fid) & 1:
                return p
        if item in self.crates:
            for t in self.trucks:
                fid = self.in_map.get((item, t))
                if fid is not None and (state >> fid) & 1:
                    return self._get_item_place(state, t)
            for h in self.hoists:
                fid = self.lifting_map.get((h, item))
                if fid is not None and (state >> fid) & 1:
                    return self._get_item_place(state, h)
        return None

    def _calculate_heuristic(self, state, task_network):
        h = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                h += task.cost
            else:
                t_name, t_args = self._parse_task_args(task.name)
                if t_name == "do_put_on":
                    c, s = t_args[0], t_args[1]
                    on_fid = self.on_map.get((c, s))
                    if on_fid is not None and (state >> on_fid) & 1:
                        continue
                    
                    # Check crate state
                    is_lifted = any((state >> self.lifting_map[(h, c)]) & 1 for h in self.hoists if (h, c) in self.lifting_map)
                    is_in_truck = any((state >> self.in_map[(c, t)]) & 1 for t in self.trucks if (c, t) in self.in_map)
                    
                    p1 = self._get_item_place(state, c)
                    p2 = self._get_item_place(state, s)
                    
                    if is_lifted:
                        h += 1 if p1 == p2 else 2
                    elif is_in_truck:
                        h += 2 if p1 == p2 else 3
                    else:
                        h += 2 if p1 == p2 else 5
                
                elif t_name == "do_clear":
                    s = t_args[0]
                    clear_fid = self.clear_map.get(s)
                    if clear_fid is not None and not ((state >> clear_fid) & 1):
                        h += 2
                
                elif t_name == "do_get_truck":
                    t, p = t_args[0], t_args[1]
                    at_fid = self.at_map.get((t, p))
                    if at_fid is not None and not ((state >> at_fid) & 1):
                        h += 1
                
                elif t_name == "do_lift_crate":
                    h += 1
                elif t_name == "do_load_truck":
                    h += 2
                elif t_name == "do_unload_truck":
                    h += 2

        return h + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
