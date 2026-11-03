from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3ProHeuristic4(TaskDecompositionHeuristic):
    """
    A fifth-generation heuristic for the Barman domain (v4).
    Refines v3 by enforcing stricter preconditions for filling Shots.
    
    New Logic:
    - Fill(Shot): Requires Shot to be Held AND Clean AND Empty.
      (v3 only checked Held).
    
    This targets the specific inefficiency where the planner might optimisticly 
    plan to fill a used shot without cleaning it first.
    """
    def __init__(self, name="BarmanGemini3ProHeuristic4"):
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
                
                task_type = parts[0]
                
                base_penalty = 0
                cheapest = node.value
                alternative = float('inf')
                for child in node.successors:
                    if child.value > cheapest:
                        alternative = min(alternative, child.value)
                
                if alternative < float('inf'):
                    base_penalty = alternative - cheapest

                # --- 1. CLEANING (v3 Logic) ---
                if "achievecleanshaker" in task_type or "achievecleanshot" in task_type:
                    if len(parts) > 1:
                        target = parts[1]
                        fid_clean = get_fact_id("clean", (target,))
                        fid_empty = get_fact_id("empty", (target,))
                        fid_ontable = get_fact_id("ontable", (target,))
                        
                        if fid_clean is not None:
                             self.optimistic_tasks[node.ID] = {
                                "type": "clean_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_clean": fid_clean,
                                "fid_empty": fid_empty,
                                "fid_ontable": fid_ontable,
                                "extra_penalty_not_empty": 1.0,
                                "extra_penalty_not_held": 1.0
                            }

                # --- 2. FILLING (v4: Stricter Shot Logic) ---
                elif "achievecontains" in task_type: 
                    if len(parts) > 2:
                        container = parts[1]
                        ingredient = parts[2]
                        
                        fid_contains = get_fact_id("contains", (container, ingredient))
                        fid_ontable = get_fact_id("ontable", (container,))
                        
                        is_shaker = "shaker" in container
                        
                        # New for v4: Checking Clean/Empty for Shot filling
                        fid_clean = get_fact_id("clean", (container,))
                        fid_empty = get_fact_id("empty", (container,))
                        
                        if fid_contains is not None: 
                            self.optimistic_tasks[node.ID] = {
                                "type": "fill_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_contains": fid_contains,
                                "fid_ontable": fid_ontable,
                                "fid_clean": fid_clean,
                                "fid_empty": fid_empty,
                                "is_shaker": is_shaker,
                                "extra_penalty_pickup": 1.0,
                                "extra_penalty_prep": 1.0 # Cost to clean/empty
                            }

                # --- 3. HAND EMPTY ---
                elif "achievehandempty" in task_type:
                    if len(parts) > 1 and base_penalty > 0:
                        hand = parts[1]
                        fid = get_fact_id("handempty", (hand,))
                        if fid is not None:
                            self.optimistic_tasks[node.ID] = {
                                "type": "simple_fact",
                                "penalty": base_penalty,
                                "fid": fid
                            }

                # --- 4. HOLDING ---
                elif "achieveholding" in task_type:
                     if len(parts) > 2:
                         hand = parts[1]
                         obj = parts[2]
                         fid_holding = get_fact_id("holding", (hand, obj))
                         fid_empty = get_fact_id("handempty", (hand,))
                         
                         if fid_holding is not None:
                             self.optimistic_tasks[node.ID] = {
                                 "type": "holding_check",
                                 "base_val": node.value,
                                 "fid_holding": fid_holding,
                                 "fid_empty": fid_empty,
                                 "cost_holding": 0,
                                 "cost_empty": 1,
                                 "cost_full": 2
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
                
                if type_ == "simple_fact":
                    if not (state & (1 << info["fid"])):
                        h_value += info["penalty"]
                
                elif type_ == "holding_check":
                    base_tdg = info["base_val"]
                    fid_holding = info["fid_holding"]
                    fid_empty = info["fid_empty"]
                    is_holding = (state & (1 << fid_holding))
                    is_empty = (fid_empty is not None) and (state & (1 << fid_empty))
                    real_cost = 2 
                    if is_holding: real_cost = 0
                    elif is_empty: real_cost = 1
                    h_value += max(0, real_cost - base_tdg)

                elif type_ == "clean_check":
                    fid_clean = info["fid_clean"]
                    fid_empty = info["fid_empty"]
                    fid_ontable = info["fid_ontable"]
                    
                    is_clean = (state & (1 << fid_clean))
                    is_empty = (fid_empty is not None) and (state & (1 << fid_empty))
                    is_ontable = (fid_ontable is not None) and (state & (1 << fid_ontable))
                    
                    if not is_clean:
                        h_value += info["penalty"]
                        if not is_empty:
                             h_value += info["extra_penalty_not_empty"]
                        if is_ontable:
                             h_value += info["extra_penalty_not_held"]

                elif type_ == "fill_check":
                    fid_contains = info["fid_contains"]
                    fid_ontable = info["fid_ontable"]
                    is_shaker = info["is_shaker"]
                    
                    is_containing = (state & (1 << fid_contains))
                    
                    if not is_containing:
                        h_value += info["penalty"]
                        
                        if not is_shaker:
                            # SHOT Logic:
                            # 1. Must be Held (Penalty if OnTable)
                            if fid_ontable is not None and (state & (1 << fid_ontable)):
                                h_value += info["extra_penalty_pickup"]
                                
                            # 2. Must be Clean & Empty (New v4)
                            # If not clean or not empty, we need extra steps (clean it first)
                            fid_clean = info["fid_clean"]
                            fid_empty = info["fid_empty"]
                            
                            is_clean = (fid_clean is not None) and (state & (1 << fid_clean))
                            is_empty = (fid_empty is not None) and (state & (1 << fid_empty))
                            
                            if (not is_clean) or (not is_empty):
                                h_value += info["extra_penalty_prep"]

                        # Shaker Logic: (No changes from v3)
                        # - Can be OnTable.
                        # - We assume 'clean shaker' is handled by other tasks or TDG.

        self.update_info(h_value)
        return h_value
