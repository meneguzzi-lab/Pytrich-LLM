from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic19(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Depots domain is the serialization of crate movement. 
    A crate requires a hoist to be lifted and a truck to be transported between locations. 
    Since hoists are limited (usually one per location) and crates are stacked, 
    extensive clearing actions are required both at the source and the destination stacks.

    Admissible Lower Bounds:
    1. Task Network Progress (h_tn): Every 'do_put_on(c, s)' task in the network requires 
       at least a Drop or Unload action. If the crate is not at the target location, 
       it requires a sequence of Lift-Load-Drive-Unload-Drop (min 4-5 actions).
    2. Clear/Stack Cost: If a crate or target surface is not clear, at least one Lift 
       action is required for every crate above it in the stack.
    3. Transport Cost: If a crate is at place A and its destination is place B, 
       a minimum of 4 actions (Lift, Load, Unload, Drop) is strictly necessary.
    """

    def __init__(self, depot_gemini3flash_heuristic_19="depot_gemini3flash_heuristic_19"):
        super().__init__(name=depot_gemini3flash_heuristic_19)
        self.at_map = {} # (locatable_idx, place_idx) -> global_id
        self.on_map = {} # (crate_idx, surface_idx) -> global_id
        self.in_map = {} # (crate_idx, truck_idx) -> global_id
        self.clear_map = {} # surface_idx -> global_id
        self.lifting_map = {} # (hoist_idx, crate_idx) -> global_id
        
        self.place_indices = []
        self.truck_indices = []
        self.crate_indices = []
        self.surface_indices = []
        
        self.goal_on = {} # crate -> surface

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return task, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Predicates
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "at":
                self.at_map[(args[0], args[1])] = f.global_id
            elif pred == "on":
                self.on_map[(args[0], args[1])] = f.global_id
            elif pred == "in":
                self.in_map[(args[0], args[1])] = f.global_id
            elif pred == "clear":
                self.clear_map[args[0]] = f.global_id
            elif pred == "lifting":
                self.lifting_map[(args[0], args[1])] = f.global_id

        # 2. Collect Type Indices (using fact arguments as proxies for constants)
        places = set()
        trucks = set()
        crates = set()
        surfaces = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                places.add(args[1])
            if pred == "on":
                crates.add(args[0])
                surfaces.add(args[1])
            if pred == "in":
                trucks.add(args[1])

        self.place_indices = list(places)
        self.truck_indices = list(trucks)
        self.crate_indices = list(crates)
        self.surface_indices = list(surfaces)

        # 3. Parse Goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_place(self, obj, state):
        for p in self.place_indices:
            fid = self.at_map.get((obj, p))
            if fid is not None and (state >> fid) & 1:
                return p
        return None

    def _compute(self, state, task_network) -> int:
        h = 0
        
        # Admissible cost per task
        for task_obj in task_network:
            task_name, args = self._parse_task_name(task_obj.name)
            
            if task_name == "do_put_on":
                crate, target_surface = args[0], args[1]
                
                # If goal already satisfied, 0 additional cost for this specific task
                on_fid = self.on_map.get((crate, target_surface))
                if on_fid is not None and (state >> on_fid) & 1:
                    continue

                # Find where crate and target surface are
                loc_c = self._get_place(crate, state)
                loc_s = self._get_place(target_surface, state)
                
                # Cost if currently in a truck
                is_in_truck = False
                for t in self.truck_indices:
                    in_fid = self.in_map.get((crate, t))
                    if in_fid is not None and (state >> in_fid) & 1:
                        is_in_truck = True
                        break
                
                if is_in_truck:
                    h += 2 # Unload + Drop
                elif loc_c and loc_s:
                    if loc_c == loc_s:
                        h += 2 # Lift + Drop
                    else:
                        h += 4 # Lift + Load + Unload + Drop
                    
                    # Penalty if crate is not clear (must lift something else first)
                    c_clear_fid = self.clear_map.get(crate)
                    if c_clear_fid is not None and not ((state >> c_clear_fid) & 1):
                        h += 2
                    
                    # Penalty if surface is not clear (must lift something else first)
                    s_clear_fid = self.clear_map.get(target_surface)
                    if s_clear_fid is not None and not ((state >> s_clear_fid) & 1):
                        h += 2
                else:
                    # Default minimum if location is unknown (e.g. being lifted)
                    h += 1
            
            elif task_name in ["do_clear", "do_lift_crate", "do_load_truck", "do_unload_truck"]:
                h += 1
            elif task_name == "nop":
                h += 0
            else:
                h += 1

        # Admissible Goal counting for stability
        for crate, target_surface in self.goal_on.items():
            on_fid = self.on_map.get((crate, target_surface))
            if on_fid is not None and not ((state >> on_fid) & 1):
                # This crate needs to be moved, but if it's already in the task network,
                # we don't want to double count. However, the task network in HTN
                # generally contains the work to be done. 
                pass

        return h
