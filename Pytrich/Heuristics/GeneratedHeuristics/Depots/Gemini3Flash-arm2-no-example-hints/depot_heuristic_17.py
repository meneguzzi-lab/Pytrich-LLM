from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# ANALYSIS:
# 1. Bottleneck: The primary bottleneck in the Depots domain is the clearance requirement (a crate/surface must be clear to be moved/placed upon) 
#    and the capacity constraints of hoists (one crate at a time) and trucks (inter-location transport).
# 2. Admissible Lower Bounds:
#    - Goal Dissatisfaction: Each crate not at its goal position 'on(c, g)' requires at least one Lift and one Drop (2 actions).
#    - Geographic Transport: If a crate's current location and its goal surface's location differ, it requires at least a Load and an Unload (additional 2 actions).
#    - Task Network Work: Each active do_put_on(c, s) task in the network, if not already satisfied, necessitates the same minimal 2 or 4 actions.
# 3. Strategy: We use the max of a state-based goal cost and a task-network work estimate to maintain admissibility while providing guidance.

class DepotsGemini3FlashHeuristic17(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_17="DepotsHeuristic"):
        super().__init__(depot_gemini3flash_heuristic_17)
        self.obj_to_id = {}
        self.id_to_obj = {}
        self.crate_ids = set()
        self.pallet_ids = set()
        self.truck_ids = set()
        self.hoist_ids = set()
        self.place_ids = set()
        
        self.fact_on_ids = {}
        self.fact_at_ids = {}
        self.fact_in_ids = {}
        self.fact_lifting_ids = {}
        self.fact_clear_ids = {}
        
        self.static_locs = {} # Pallets and Hoists don't move locations
        self.goal_on = {} # crate_id -> surface_id

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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map Objects
        # We don't have types directly, so we infer from predicate usage or initial state
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            for arg in args:
                if arg not in self.obj_to_id:
                    idx = len(self.obj_to_id)
                    self.obj_to_id[arg] = idx
                    self.id_to_obj[idx] = arg
            
            if pred == "on":
                self.fact_on_ids[(self.obj_to_id[args[0]], self.obj_to_id[args[1]])] = fact.global_id
                self.crate_ids.add(self.obj_to_id[args[0]])
            elif pred == "at":
                self.fact_at_ids[(self.obj_to_id[args[0]], self.obj_to_id[args[1]])] = fact.global_id
                self.place_ids.add(self.obj_to_id[args[1]])
            elif pred == "in":
                self.fact_in_ids[(self.obj_to_id[args[0]], self.obj_to_id[args[1]])] = fact.global_id
                self.truck_ids.add(self.obj_to_id[args[1]])
            elif pred == "lifting":
                self.fact_lifting_ids[(self.obj_to_id[args[0]], self.obj_to_id[args[1]])] = fact.global_id
                self.hoist_ids.add(self.obj_to_id[args[0]])
            elif pred == "clear":
                self.fact_clear_ids[self.obj_to_id[args[0]]] = fact.global_id

        # 2. Identify Static Locations (Pallets and Hoists stay put)
        init_state = initial_node.state
        for f_tuple, g_id in self.fact_at_ids.items():
            obj_id, p_id = f_tuple
            if (init_state >> g_id) & 1:
                obj_name = self.id_to_obj[obj_id]
                # Heuristic check for pallet/hoist based on naming convention or fact persistence
                # In Depots, hoists and pallets are locatable but don't Drive.
                if "pallet" in obj_name or "hoist" in obj_name or "depot" in obj_name or "distributor" in obj_name:
                    self.static_locs[obj_id] = p_id
                    if "pallet" in obj_name: self.pallet_ids.add(obj_id)

        # 3. Parse Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on[self.obj_to_id[args[0]]] = self.obj_to_id[args[1]]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_loc(self, obj_id, state):
        if obj_id in self.static_locs:
            return self.static_locs[obj_id]
        for p_id in self.place_ids:
            if (state >> self.fact_at_ids.get((obj_id, p_id), 0)) & 1:
                return p_id
        for t_id in self.truck_ids:
            if (state >> self.fact_in_ids.get((obj_id, t_id), 0)) & 1:
                for p_id in self.place_ids:
                    if (state >> self.fact_at_ids.get((t_id, p_id), 0)) & 1:
                        return p_id
        for h_id in self.hoist_ids:
            if (state >> self.fact_lifting_ids.get((h_id, obj_id), 0)) & 1:
                return self.static_locs.get(h_id)
        return None

    def _compute_crate_cost(self, c_id, s_id, state):
        # Admissible cost to put crate c on surface s
        if (state >> self.fact_on_ids.get((c_id, s_id), 0)) & 1:
            return 0
        cost = 2 # Lift + Drop
        loc_c = self._find_loc(c_id, state)
        loc_s = self._find_loc(s_id, state)
        if loc_c is not None and loc_s is not None and loc_c != loc_s:
            cost += 2 # Load + Unload
        return cost

    def _compute(self, state, task_network):
        # 1. Goal-based lower bound
        h_goal = 0
        for c_id, s_id in self.goal_on.items():
            h_goal += self._compute_crate_cost(c_id, s_id, state)

        # 2. Task-network lower bound
        h_tn = 0
        for task in task_network:
            if hasattr(task, 'name'):
                parts = task.name.split()
                t_name = parts[0]
                args = parts[1:]
                if t_name == "do_put_on" and len(args) == 2:
                    c_id = self.obj_to_id.get(args[0])
                    s_id = self.obj_to_id.get(args[1])
                    if c_id is not None and s_id is not None:
                        h_tn += self._compute_crate_cost(c_id, s_id, state)
                elif t_name == "do_clear":
                    s_id = self.obj_to_id.get(args[0])
                    if s_id is not None and not ((state >> self.fact_clear_ids.get(s_id, 0)) & 1):
                        h_tn += 1
                elif t_name == "do_get_truck":
                    t_id = self.obj_to_id.get(args[0])
                    p_id = self.obj_to_id.get(args[1])
                    if t_id is not None and p_id is not None:
                        if not ((state >> self.fact_at_ids.get((t_id, p_id), 0)) & 1):
                            h_tn += 1

        return max(h_goal, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
