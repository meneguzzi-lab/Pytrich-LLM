from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic14(Heuristic):
    """
    Heuristic for Depots domain.
    Bottleneck: The serial clearance of crate stacks on surfaces and the multi-step transport via trucks.
    Admissible Bounds:
    - do_put_on(c, s) requires at least 2 actions (Lift, Drop) if not satisfied.
    - If c and s are at different places, it requires at least 4 actions (Lift, Load, Unload, Drop) and potentially a Drive.
    - Clearing a surface requires 2 actions (Lift, Load/Drop) for every crate in the stack above it.
    """
    def __init__(self, depot_gemini3flash_heuristic_14="depot_gemini3flash_heuristic_14"):
        super().__init__(depot_gemini3flash_heuristic_14)
        self.at_id = {} # (locatable, place) -> id
        self.on_id = {} # (crate, surface) -> id
        self.on_facts_by_surface = {} # surface -> [(id, crate)]
        self.clear_id = {} # surface -> id
        self.in_id = {} # (crate, truck) -> id
        self.available_id = {} # hoist -> id
        self.places = []
        self.trucks = []
        self.hoists = []
        self.crates = []
        self.surfaces = []
        self.pallet_to_place = {}

    def _parse_fact_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return prefix, predicate, args
        return None, None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            prefix, pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_id[(args[0], args[1])] = f.global_id
                if args[1] not in self.places: self.places.append(args[1])
            elif pred == "on":
                self.on_id[(args[0], args[1])] = f.global_id
                if args[1] not in self.on_facts_by_surface: self.on_facts_by_surface[args[1]] = []
                self.on_facts_by_surface[args[1]].append((f.global_id, args[0]))
            elif pred == "clear":
                self.clear_id[args[0]] = f.global_id
            elif pred == "in":
                self.in_id[(args[0], args[1])] = f.global_id
            elif pred == "available":
                self.available_id[args[0]] = f.global_id

        self.trucks = [o for o in model.desc.get('truck', [])] # Assuming desc has types or just using manual lists
        # Standard discovery of objects from facts
        hoist_objs = set()
        truck_objs = set()
        crate_objs = set()
        surface_objs = set()
        for (loc, p) in self.at_id.keys():
            if "hoist" in loc: hoist_objs.add(loc)
            if "truck" in loc: truck_objs.add(loc)
            if "crate" in loc: crate_objs.add(loc)
            if "pallet" in loc: surface_objs.add(loc)
        for (c, s) in self.on_id.keys():
            crate_objs.add(c)
            surface_objs.add(s)
        
        self.hoists = list(hoist_objs)
        self.trucks = list(truck_objs)
        self.crates = list(crate_objs)
        self.surfaces = list(surface_objs)

        # Pallets are static
        for s in self.surfaces:
            if "pallet" in s:
                for p in self.places:
                    if (s, p) in self.at_id and (model.initial_state >> self.at_id[(s, p)]) & 1:
                        self.pallet_to_place[s] = p

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _get_at_place(self, obj, state):
        for p in self.places:
            if (obj, p) in self.at_id and (state >> self.at_id[(obj, p)]) & 1:
                return p
        return None

    def _get_crate_on(self, surface, state):
        if surface not in self.on_facts_by_surface: return None
        for f_id, crate in self.on_facts_by_surface[surface]:
            if (state >> f_id) & 1: return crate
        return None

    def _get_clearance_cost(self, surface, state):
        cost = 0
        curr = surface
        while curr in self.clear_id and not ((state >> self.clear_id[curr]) & 1):
            nxt = self._get_crate_on(curr, state)
            if nxt:
                cost += 2
                curr = nxt
            else: break
        return cost

    def _compute(self, state, task_network):
        total_h = 0
        for task in task_network:
            parts = task.name.split()
            if not parts: continue
            cmd = parts[0]
            args = parts[1:]

            if cmd == "do_put_on":
                c, s2 = args[0], args[1]
                if not ((c, s2) in self.on_id and (state >> self.on_id[(c, s2)]) & 1):
                    p2 = self.pallet_to_place.get(s2) or self._get_at_place(s2, state)
                    p1 = self._get_at_place(c, state)
                    if p1 == p2 and p1 is not None:
                        total_h += 2 + self._get_clearance_cost(c, state) + self._get_clearance_cost(s2, state)
                    else:
                        total_h += 5 + self._get_clearance_cost(c, state) + self._get_clearance_cost(s2, state)
            elif cmd == "do_clear":
                total_h += self._get_clearance_cost(args[0], state)
            elif cmd == "do_get_truck":
                t, p = args[0], args[1]
                if not ((t, p) in self.at_id and (state >> self.at_id[(t, p)]) & 1):
                    total_h += 1
            elif cmd == "do_lift_crate":
                total_h += 1
            elif cmd == "do_load_truck":
                total_h += 2 + self._get_clearance_cost(args[0], state)
            elif cmd == "do_unload_truck":
                total_h += 2 + self._get_clearance_cost(args[1], state)
            elif cmd == "nop":
                pass
            else:
                total_h += 1
        
        # Basic symmetry breaking
        total_h += 0.001 * len(task_network)
        return int(total_h)
