from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3ProHeuristic2(TaskDecompositionHeuristic):
    """
    A third-generation heuristic for the Barman domain.
    Builds on v1 by adding interaction-aware penalties.
    
    v0 Awareness: Cleanliness penalties.
    v1 Awareness: + HandEmpty, Holding, OnTable penalties.
    v2 Awareness (This Heuristic):
    - Refined Cleaning: Penalizes 'Clean' tasks if the vessel is NOT empty (requires Empty action).
    - Refined Filling: Penalizes 'Contains' tasks if the vessel is NOT held (requires Pickup action).
    """
    def __init__(self, name="BarmanGemini3ProHeuristic2"):
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
                
                # --- Base Penalty Logic (from v1, simplified re-use) ---
                # We calculate a 'base_penalty' if the immediate optimistic check fails.
                base_penalty = 0
                cheapest = node.value
                alternative = float('inf')
                # Find alternative cost
                for child in node.successors:
                    if child.value > cheapest:
                        alternative = min(alternative, child.value)
                
                if alternative < float('inf'):
                    base_penalty = alternative - cheapest

                # --- 1. CLEANING (Extended) ---
                if "achievecleanshaker" in task_type or "achievecleanshot" in task_type:
                    if len(parts) > 1:
                        target = parts[1]
                        
                        # V1 Check: Is it clean?
                        fid_clean = get_fact_id("clean", (target,))
                        # V2 Check: Is it empty? (If not empty, we need to empty it first)
                        fid_empty = get_fact_id("empty", (target,))
                        
                        # We store a "compound_check"
                        if fid_clean is not None: # We need this to exist to track it
                             self.optimistic_tasks[node.ID] = {
                                "type": "clean_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0, # Default to 1.0 if not found
                                "fid_clean": fid_clean,
                                "fid_empty": fid_empty, # Might be None if predicate doesn't exist (but it should)
                                "extra_penalty_not_empty": 1.0 # Cost of splitting/emptying roughly
                            }

                # --- 2. FILLING (New in v2) ---
                elif "achievecontains" in task_type: 
                    # achievecontains(container, ingredient) usually
                    if len(parts) > 2:
                        container = parts[1]
                        ingredient = parts[2]
                        
                        # Optimization: Is it already containing it?
                        fid_contains = get_fact_id("contains", (container, ingredient))
                        
                        # To fill it, we usually need to HOLD the container.
                        # So if we are not holding it, we pay pickup cost.
                        # We assume "holding(hand, container)" exists for some hand.
                        # Since checking "any hand" is hard with specific fact IDs, 
                        # we check "ontable(container)". If it is on table, we are NOT holding it.
                        fid_ontable = get_fact_id("ontable", (container,))
                        
                        if fid_contains is not None: 
                            self.optimistic_tasks[node.ID] = {
                                "type": "fill_check",
                                "penalty": base_penalty if base_penalty > 0 else 1.0,
                                "fid_contains": fid_contains,
                                "fid_ontable": fid_ontable,
                                "extra_penalty_pickup": 1.0
                            }

                # --- 3. HAND EMPTY (From v1) ---
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

                # --- 4. HOLDING (From v1) ---
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
                
                # --- SIMPLE FACT (HandEmpty, OnTable) ---
                if type_ == "simple_fact":
                    if not (state & (1 << info["fid"])):
                        h_value += info["penalty"]
                
                # --- HOLDING CHECK ---
                elif type_ == "holding_check":
                    base_tdg = info["base_val"]
                    fid_holding = info["fid_holding"]
                    fid_empty = info["fid_empty"]
                    
                    is_holding = (state & (1 << fid_holding))
                    is_empty = (fid_empty is not None) and (state & (1 << fid_empty))
                    
                    real_cost = 2 # full
                    if is_holding: real_cost = 0
                    elif is_empty: real_cost = 1
                    
                    h_value += max(0, real_cost - base_tdg)

                # --- CLEAN CHECK (v2 Improved) ---
                elif type_ == "clean_check":
                    fid_clean = info["fid_clean"]
                    fid_empty = info["fid_empty"]
                    
                    is_clean = (state & (1 << fid_clean))
                    is_empty = (fid_empty is not None) and (state & (1 << fid_empty))
                    
                    if not is_clean:
                        # If not clean, we base pay the clean penalty
                        h_value += info["penalty"]
                        
                        # BUT, if it's also NOT empty, we pay extra for emptying
                        # Note: If it IS clean, we assume it's fine (cost 0).
                        if not is_empty:
                             h_value += info["extra_penalty_not_empty"]

                # --- FILL CHECK (v2 New) ---
                elif type_ == "fill_check":
                    fid_contains = info["fid_contains"]
                    fid_ontable = info["fid_ontable"]
                    
                    is_containing = (state & (1 << fid_contains))
                    
                    if not is_containing:
                        # We need to fill it.
                        # Base penalty? TDG usually assumes fill is cost 1 (pour).
                        # But if TDG assumes cost 0 (already full), apply base penalty.
                        # Usually for filling, TDG isn't 0-cost if empty. 
                        # So 'info["penalty"]' might be 0 if TDG accounts for pouring cost.
                        h_value += info["penalty"]
                        
                        # REQUIREMENT: Holding container.
                        # If container is on table, we must pick it up.
                        if fid_ontable is not None and (state & (1 << fid_ontable)):
                            h_value += info["extra_penalty_pickup"]

        self.update_info(h_value)
        return h_value
