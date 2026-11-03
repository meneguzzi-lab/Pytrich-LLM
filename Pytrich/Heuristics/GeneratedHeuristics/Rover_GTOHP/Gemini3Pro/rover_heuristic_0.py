from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RoverGemini3ProHeuristic0(TaskDecompositionHeuristic):
    """
    Initial state-aware heuristic for the Rover domain (v0).
    Penalizes optimistic assumptions about:
    1. Rover Location (must be at waypoint to sample).
    2. Store Status (must be empty to sample).
    3. Calibration (must be calibrated to image).
    """
    def __init__(self, name="RoverGemini3ProHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}
        self.rovers = set()
        self.stores = set()
        self.cameras = set()

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & Entity Sets
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
            predicate = parts[0]
            args = tuple(parts[1:])
            key = (predicate, args)
            self.fact_map[key] = fact.global_id
            
            # Identify Rovers
            if predicate == "at" and len(args) == 2:
                self.rovers.add(args[0])
            
            # Identify Stores (from empty/full)
            if (predicate == "empty" or predicate == "full") and len(args) == 1:
                self.stores.add(args[0])
                
            # Identify Cameras (from calibrated)
            if predicate == "calibrated" and len(args) == 2:
                # calibrated camera rover
                self.cameras.add(args[0])

        def get_fact_id(pred, args_tuple):
            key_std = (pred, args_tuple)
            key_plus = ("+" + pred, args_tuple)
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

                # --- 1. GET DATA (Rock/Soil) ---
                if "get_rock_data" in task_type or "get_soil_data" in task_type:
                    # get_rock_data waypoint0
                    if len(parts) > 1:
                        wp = parts[1]
                        
                        # Collect all potential "at" facts for this waypoint
                        # (Checking if ANY rover is at this waypoint)
                        
                        at_facts = []
                        for rover in self.rovers:
                            fid = get_fact_id("at", (rover, wp))
                            if fid is not None:
                                at_facts.append(fid)
                                
                        # Collect all "empty" facts
                        empty_facts = []
                        for store in self.stores:
                            fid = get_fact_id("empty", (store,))
                            if fid is not None:
                                empty_facts.append(fid)
                        
                        self.optimistic_tasks[node.ID] = {
                            "type": "sample_check",
                            "penalty": base_penalty if base_penalty > 0 else 1.0,
                            "at_facts": at_facts,
                            "empty_facts": empty_facts,
                            "extra_penalty_nav": 1.0,
                            "extra_penalty_drop": 1.0
                        }

                # --- 2. GET IMAGE ---
                elif "get_image_data" in task_type:
                    # get_image_data objective1 low_res
                    # Methods often involve calibration.
                    # Check if ANY camera is calibrated?
                    
                    calib_facts = []
                    for cam in self.cameras:
                        # Need to know which rover owns the camera to check "calibrated camera rover"
                        # We iterate all rovers
                        for rover in self.rovers:
                            fid = get_fact_id("calibrated", (cam, rover))
                            if fid is not None:
                                calib_facts.append(fid)
                    
                    if calib_facts:
                         self.optimistic_tasks[node.ID] = {
                            "type": "image_check",
                            "penalty": base_penalty if base_penalty > 0 else 1.0,
                            "calib_facts": calib_facts,
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
                
                if type_ == "sample_check":
                    at_facts = info["at_facts"]
                    empty_facts = info["empty_facts"]
                    
                    # Check AT: Is any rover at the waypoint?
                    aligned = False
                    for fid in at_facts:
                        if (state & (1 << fid)):
                            aligned = True
                            break
                    if not aligned:
                        h_value += info["extra_penalty_nav"]
                        
                    # Check EMPTY: Is any store empty?
                    # (Strictly we need the SAME rover to be AT and EMPTY, but heuristic 'any' is safe lower bound approx)
                    # Use 'any' for loose coupling.
                    # If we enforce 'same', we need to couple (rover -> store).
                    # 'rover0store' implies 'rover0'.
                    # Let's verify 'any' first.
                    
                    has_empty = False
                    for fid in empty_facts:
                        if (state & (1 << fid)):
                            has_empty = True
                            break
                    if not has_empty:
                        h_value += info["extra_penalty_drop"]

                elif type_ == "image_check":
                    calib_facts = info["calib_facts"]
                    is_calib = False
                    for fid in calib_facts:
                        if (state & (1 << fid)):
                            is_calib = True
                            break
                    if not is_calib:
                        h_value += info["extra_penalty"]

        self.update_info(h_value)
        return h_value
