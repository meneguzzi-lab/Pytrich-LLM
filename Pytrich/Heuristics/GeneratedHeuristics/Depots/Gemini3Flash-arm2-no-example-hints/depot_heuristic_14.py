from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic14(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots Domain.

    Bottleneck Analysis:
    The primary bottleneck is the sequential availability of hoists at specific locations and the 
    movement of trucks between places. Crates are often stacked deep on pallets, requiring multiple 
    hoist actions to clear surfaces before a target crate can be moved or placed.

    Admissible Lower Bounds (max-composite):
    1. Goal Displacement: Every crate not well-placed on its goal surface requires at least a Lift and 
       a Drop/Load (2 actions). If the goal location is different, it requires at least 5 actions 
       (Lift, Load, Drive, Unload, Drop).
    2. Obstruction Clearing: Every crate obstructing a misplaced crate or a target goal surface 
       must be moved, requiring at least 2 actions (Lift/Drop).
    3. Task Network Progress: The remaining HTN tasks (do_put_on, do_clear) represent remaining work
       where each non-nop task contributes to a minimum of 2 actions.
    """

    def __init__(self, depot_gemini3flash_heuristic_14="DepotsGemini3FlashHeuristic14"):
        super().__init__(depot_gemini3flash_heuristic_14)
        self.goal_on = {} # crate -> target_surface
        self.goal_loc = {}
        self.hoist_at = {} # static
        self.pallet_at = {} # static
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.crates = set()
        self.surfaces = set()

        # Fast lookup indices
        self.on_ids = []
        self.at_ids = []
        self.in_ids = []
        self.lifting_ids = []
        self.available_ids = []
        self.clear_ids = []
        self.id_to_args = {}

    def _parse_comp_name(self, name):
        if "[" in name and name.endswith("]"):
            idx = name.find("[")
            return name[:idx], name[idx+1:-1].split(",")
        return name, []

    def _parse_fact_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if "[" in name else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # 2. Extract types and static locations from operator groundings (PANDA warning)
        for op in model.operators:
            pred, args = self._parse_comp_name(op.name)
            if pred in ["Lift", "Drop", "Load", "Unload"]:
                # args: hoist, crate, surface/truck, place
                h, c, st, p = args[0], args[1], args[2], args[3]
                self.hoists.add(h)
                self.hoist_at[h] = p
                self.places.add(p)
                self.crates.add(c)
                if pred in ["Lift", "Drop"]:
                    # Surface could be a pallet or crate. If it's never a crate, it's a pallet.
                    # In Depots, we'll collect all surfaces and then identify pallets.
                    self.surfaces.add(st)

        # Identifying pallets: Surfaces that aren't crates
        self.pallets = self.surfaces - self.crates

        # 3. Static Pallet Locations from Init Facts
        init_facts = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "at":
                if args[0] in self.pallets:
                    self.pallet_at[args[0]] = args[1]
                if args[1] in self.places:
                    self.places.add(args[1])
            elif pred == "in":
                self.trucks.add(args[1])

        # 4. Precompute Goal Locations
        for c in self.crates:
            curr = c
            visited = set()
            while curr in self.goal_on and curr not in visited:
                visited.add(curr)
                curr = self.goal_on[curr]
            if curr in self.pallet_at:
                self.goal_loc[c] = self.pallet_at[curr]

        # 5. Fast Fact ID Categorization
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            self.id_to_args[fact.global_id] = (pred, args)
            if pred == "on": self.on_ids.append(fact.global_id)
            elif pred == "at": self.at_ids.append(fact.global_id)
            elif pred == "in": self.in_ids.append(fact.global_id)
            elif pred == "lifting": self.lifting_ids.append(fact.global_id)
            elif pred == "clear": self.clear_ids.append(fact.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        # Build current world state maps
        truck_loc = {}
        is_on = {} # crate -> surface
        on_top_of = {} # surface -> crate
        loc_map = {} # locatable -> place
        in_truck = {} # crate -> truck
        lifting_by = {} # crate -> hoist

        for fid in self.at_ids:
            if (state >> fid) & 1:
                args = self.id_to_args[fid][1]
                loc_map[args[0]] = args[1]
                if args[0] in self.trucks: truck_loc[args[0]] = args[1]

        for fid in self.on_ids:
            if (state >> fid) & 1:
                args = self.id_to_args[fid][1]
                is_on[args[0]] = args[1]
                on_top_of[args[1]] = args[0]

        for fid in self.in_ids:
            if (state >> fid) & 1:
                args = self.id_to_args[fid][1]
                in_truck[args[0]] = args[1]
                if args[1] in truck_loc:
                    loc_map[args[0]] = truck_loc[args[1]]

        for fid in self.lifting_ids:
            if (state >> fid) & 1:
                args = self.id_to_args[fid][1]
                lifting_by[args[1]] = args[0]
                if args[0] in self.hoist_at:
                    loc_map[args[1]] = self.hoist_at[args[0]]

        # Recursive Well-Placed check
        memo_wp = {}
        def is_well_placed(c):
            if c in self.pallets: return True
            if c not in is_on: return False
            if c in memo_wp: return memo_wp[c]
            target_s = self.goal_on.get(c)
            if target_s and is_on[c] == target_s:
                res = is_well_placed(target_s)
            else:
                res = False
            memo_wp[c] = res
            return res

        # Determine crates that MUST move
        must_move = set()
        for c in self.crates:
            if not is_well_placed(c):
                must_move.add(c)

        # Add obstructions: crates on top of goal surfaces
        for c, target_s in self.goal_on.items():
            if target_s in on_top_of:
                curr = on_top_of[target_s]
                while curr:
                    if curr == c: break # Crate itself is okay if it's the goal
                    must_move.add(curr)
                    curr = on_top_of.get(curr)

        # Compute Goal H
        h_goals = 0
        for c in must_move:
            g_loc = self.goal_loc.get(c)
            c_loc = loc_map.get(c)

            if c in lifting_by:
                h_goals += 1
                if c_loc != g_loc: h_goals += 3
            elif c in in_truck:
                h_goals += 2
                if c_loc != g_loc: h_goals += 1
            else:
                h_goals += 2
                if c_loc != g_loc: h_goals += 3

        # Compute Task Network H
        h_tn = 0
        for task in task_network:
            t_name, t_args = self._parse_comp_name(task.name)
            if t_name == "do_put_on":
                c, s = t_args[0], t_args[1]
                # If not satisfied, this task requires work similar to h_goals
                if is_on.get(c) != s:
                    base = 2
                    if loc_map.get(c) != self.goal_loc.get(c): base += 3
                    h_tn += base
            elif t_name == "do_clear":
                s = t_args[0]
                if s in on_top_of:
                    curr = on_top_of[s]
                    while curr:
                        h_tn += 2
                        curr = on_top_of.get(curr)
            elif t_name == "do_get_truck":
                t, p = t_args[0], t_args[1]
                if truck_loc.get(t) != p: h_tn += 2
            elif t_name != "nop":
                h_tn += 1

        return max(h_goals, h_tn)
