from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BlocksworldGemini3ProHeuristic0(TaskDecompositionHeuristic):
    """
    Initial state-aware heuristic for the Blocksworld domain (v0).
    Penalizes optimistic assumptions about:
    1. Clear status (blocks must be clear to be moved).
    2. Holding status (must hold block to stack/putdown).
    3. Hand status (must be empty to pickup/unstack).
    """
    def __init__(self, name="BlocksworldGemini3ProHeuristic0"):
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
                
                # Check for "do_put_on", "do_on_table", "do_clear"
                # Task names might be "do_put_on", "do_move", etc.
                # PSAS lines: "1 do_put_on[b3,b1]", "1 do_move[b3,b1]"
                
                task_type = parts[0]
                
                base_penalty = 0
                cheapest = node.value
                alternative = float('inf')
                for child in node.successors:
                    if child.value > cheapest:
                        alternative = min(alternative, child.value)
                
                if alternative < float('inf'):
                    base_penalty = alternative - cheapest

                # --- 1. DO_PUT_ON(x, y) ---
                if "do_put_on" in task_type:
                    # do_put_on x y
                    if len(parts) > 2:
                        x = parts[1]
                        y = parts[2]
                        
                        fid_holding = get_fact_id("holding", (x,))
                        fid_clear = get_fact_id("clear", (y,))
                        
                        if fid_holding is not None:
                            self.optimistic_tasks[node.ID] = {
                                "type": "put_on_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_holding": fid_holding,
                                "fid_clear": fid_clear,
                                "extra_penalty": 1.0
                            }

                # --- 2. DO_ON_TABLE(x) ---
                elif "do_on_table" in task_type:
                    if len(parts) > 1:
                        x = parts[1]
                        
                        fid_holding = get_fact_id("holding", (x,))
                        if fid_holding is not None:
                            self.optimistic_tasks[node.ID] = {
                                "type": "on_table_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_holding": fid_holding,
                                "extra_penalty": 1.0
                            }
                            
                # --- 3. DO_CLEAR(x) ---
                elif "do_clear" in task_type:
                    if len(parts) > 1:
                        x = parts[1]
                        fid_clear = get_fact_id("clear", (x,))
                        if fid_clear is not None:
                            self.optimistic_tasks[node.ID] = {
                                "type": "clear_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0, 
                                "fid_clear": fid_clear,
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
                    fid_holding = info["fid_holding"]
                    fid_clear = info["fid_clear"]
                    
                    is_holding = (state & (1 << fid_holding))
                    
                    if not is_holding:
                         h_value += info["extra_penalty"]
                    
                    if fid_clear is not None:
                         is_clear = (state & (1 << fid_clear))
                         if not is_clear:
                             h_value += info["extra_penalty"]

                elif type_ == "on_table_check":
                    fid_holding = info["fid_holding"]
                    is_holding = (state & (1 << fid_holding))
                    # If not holding, we assume we need to unstack/pickup.
                    if not is_holding:
                        h_value += info["extra_penalty"]

                elif type_ == "clear_check":
                    fid_clear = info["fid_clear"]
                    is_clear = (state & (1 << fid_clear))
                    if not is_clear:
                        h_value += info["extra_penalty"]
        
        self.update_info(h_value)
        return h_value
