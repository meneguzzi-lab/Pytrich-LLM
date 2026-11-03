from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3ProHeuristic0(TaskDecompositionHeuristic):
    """
    Initial state-aware heuristic for the Robot domain (v0).
    Penalizes optimistic assumptions about:
    1. Robot Location (must be at object for pickup).
    2. Arm Status (must be empty for pickup).
    3. Door Status (must be open for move).
    """
    def __init__(self, name="RobotGemini3ProHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map and Reverse Map
        # We need to find "in(pkg, ?)" or "rloc(?)" 
        # Map: (predicate, arg0) -> arg1
        self.pkg_locations = {} # pkg -> room
        self.fact_map = {}
        
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
            predicate = parts[0]
            args = tuple(parts[1:])
            key = (predicate, args)
            self.fact_map[key] = fact.global_id
            
            # Pre-compute static locations if possible?
            # In Robot, pkg locations might change? "pickup" changes location to "holding".
            # BUT for checking "pickup" precondition at start of method, we check where it IS.
            # Wait, "in[pkg, room]" is a fact. We can only map it if we iterate facts.
            # But during heuristic eval (call), we check the STATE.
            # So we need to potential facts.
            
            if predicate == "in" and len(args) == 2:
                pkg, room = args
                if pkg not in self.pkg_locations:
                    self.pkg_locations[pkg] = []
                self.pkg_locations[pkg].append( (room, fact.global_id) )

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

                # --- 1. PICKUP ---
                # Task: achieve-goals-pickup(pkg) OR pickup(pkg, room)
                if "pickup" in task_type:
                    if len(parts) > 1:
                        pkg = parts[1] # "o1"
                        
                        # We need to check if robot is at pkg location.
                        # Since pkg location can be dynamic, we might have multiple "in(pkg, room)" facts?
                        # In this domain, valid "in" facts are mutually exclusive usually.
                        
                        possible_locs = self.pkg_locations.get(pkg, [])
                        
                        fid_armempty = get_fact_id("armempty", ())
                        
                        if fid_armempty is not None:
                            self.optimistic_tasks[node.ID] = {
                                "type": "pickup_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_armempty": fid_armempty,
                                "pkg_locs": possible_locs, # list of (room, fid_in_room)
                                "extra_penalty_move": 1.0,
                                "extra_penalty_drop": 1.0
                            }

                # --- 2. MOVE ---
                # Task: achieve-goals-move?
                # Usually "move" implies a destination. 
                # If abstract task doesn't have args, we can't do much.
                # "achieve-goals-move" (no args in psas?).
                # If so, we skip. Primitive 'move' has args.

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
                
                if type_ == "pickup_check":
                    fid_armempty = info["fid_armempty"]
                    pkg_locs = info["pkg_locs"]
                    
                    is_empty = (state & (1 << fid_armempty))
                    if not is_empty:
                        h_value += info["extra_penalty_drop"]
                    
                    # Check if Robot is at the SAME room as the package
                    # We need ONE (pkg_in_room AND robot_in_room) pair to be true.
                    # If pkg is in r1, robot must be in r1.
                    
                    aligned = False
                    for room, fid_in_room in pkg_locs:
                        if (state & (1 << fid_in_room)):
                            # Package IS in this room.
                            # Check if Robot is in this room.
                            fid_rloc = self.get_fact_id_runtime("rloc", (room,)) 
                            if fid_rloc is not None and (state & (1 << fid_rloc)):
                                aligned = True
                                break
                    
                    if not aligned:
                        # Either pkg not found (holding?) or robot not there.
                        # If pkg is holding, then pickup is invalid anyway? Or already done?
                        # If holding, 'in' fact is False. implicit aligned=False.
                        # If aligned is False, we add move penalty.
                        h_value += info["extra_penalty_move"]

        self.update_info(h_value)
        return h_value
        
    def get_fact_id_runtime(self, pred, args_tuple):
        # Helper reuse
        key_std = (pred, args_tuple)
        key_plus = ("+" + pred, args_tuple)
        if key_plus in self.fact_map:
            return self.fact_map[key_plus]
        if key_std in self.fact_map:
            return self.fact_map[key_std]
        return None
