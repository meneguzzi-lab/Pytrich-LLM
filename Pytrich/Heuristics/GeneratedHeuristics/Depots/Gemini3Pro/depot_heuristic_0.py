from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class DepotGemini3ProHeuristic0(TaskDecompositionHeuristic):
    """
    Initial state-aware heuristic for the Depots domain (v0).
    Penalizes optimistic assumptions about:
    1. Truck Location (must be at loading site).
    2. Crate Clearance (must be clear to lift/stack).
    3. Hoist Availability (must be available to lift).
    """
    def __init__(self, name="DepotGemini3ProHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
            predicate = parts[0]
            args = tuple(parts[1:])
            key = (predicate, args)
            self.fact_map[key] = fact.global_id

        def get_fact_id(pred, args_tuple):
            key_std = (pred, args_tuple)
            key_plus = ("+" + pred, args_tuple)
            # Depots often has "+at", "-at".
            if key_plus in self.fact_map:
                return self.fact_map[key_plus]
            if key_std in self.fact_map:
                return self.fact_map[key_std]
            return None

        # 2. Identify Optimistic Tasks
        for node in self.and_or_graph.nodes:
            if node and node.content_type == ContentType.ABSTRACT_TASK:
                task_name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                parts = task_name_clean.split()
                if not parts:
                    continue
                
                task_type = parts[0]
                
                base_penalty = 0
                cheapest = node.value
                alternative = float('inf')
                for child in node.successors:
                    if child.value > cheapest:
                        alternative = min(alternative, child.value)
                
                if alternative < float('inf'):
                    base_penalty = alternative - cheapest

                # --- 1. DO_PUT_ON(c, s) ---
                if "do_put_on" in task_type:
                    # do_put_on crate0 pallet2
                    if len(parts) > 2:
                        crate = parts[1]
                        surface = parts[2]
                        
                        # Only check if surface is a crate (pallets usually always "clear" to stack if empty, or handled by domain logic)
                        # Actually pallets can be non-clear.
                        # But typically 'clear' predicate applies to both.
                        
                        fid_clear = get_fact_id("clear", (surface,))
                        
                        if fid_clear is not None:
                            self.optimistic_tasks[node.ID] = {
                                "type": "put_on_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_clear": fid_clear,
                                "extra_penalty": 1.0
                            }

                # --- 2. DO_LOAD_TRUCK(c, s, l, t) ---
                elif "do_load_truck" in task_type:
                    # do_load_truck crate0 pallet2 distributor1 truck1
                    if len(parts) > 4:
                        truck = parts[4]
                        loc = parts[3]
                        
                        fid_at = get_fact_id("at", (truck, loc))
                        
                        if fid_at is not None:
                             self.optimistic_tasks[node.ID] = {
                                "type": "load_truck_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_at": fid_at,
                                "extra_penalty": 1.0 # Drive cost
                            }

                # --- 3. DO_LIFT_CRATE(c, l, h) ---
                elif "do_lift_crate" in task_type:
                    if len(parts) > 3:
                        crate = parts[1]
                        hoist = parts[3]
                        
                        fid_clear = get_fact_id("clear", (crate,))
                        fid_avail = get_fact_id("available", (hoist,))
                        
                        if fid_clear is not None and fid_avail is not None:
                             self.optimistic_tasks[node.ID] = {
                                 "type": "lift_check",
                                 "penalty": base_penalty if base_penalty > 0 else 1.0,
                                 "fid_clear": fid_clear,
                                 "fid_avail": fid_avail,
                                 "extra_penalty": 1.0
                             }

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                type_ = info["type"]
                
                if type_ == "put_on_check":
                    fid_clear = info["fid_clear"]
                    is_clear = (state & (1 << fid_clear))
                    if not is_clear:
                        h_value += info["extra_penalty"]

                elif type_ == "load_truck_check":
                    fid_at = info["fid_at"]
                    is_at = (state & (1 << fid_at))
                    if not is_at:
                        h_value += info["extra_penalty"]

                elif type_ == "lift_check":
                    fid_clear = info["fid_clear"]
                    fid_avail = info["fid_avail"]
                    
                    is_clear = (state & (1 << fid_clear))
                    is_avail = (state & (1 << fid_avail))
                    
                    if not is_clear or not is_avail:
                        h_value += info["extra_penalty"]

        self.update_info(h_value)
        return h_value
