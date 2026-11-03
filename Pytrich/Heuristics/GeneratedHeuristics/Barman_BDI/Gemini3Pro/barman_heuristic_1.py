from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3ProHeuristic1(TaskDecompositionHeuristic):
    """
    A second-generation heuristic for the Barman domain.
    Improves upon BarmanGemini3ProHeuristic0 ("v0") by extending state-awareness.
    
    v0 Awareness:
    - Cleanliness: Penalizes optimistic assumption that shakers/shots are clean.
    
    v1 Awareness (This Heuristic):
    - Cleanliness: Same as v0.
    - Hand State: Penalizes optimistic assumption that hands are empty or holding the target object.
    - Object Location: Penalizes optimistic assumption that objects are already on the table.
    
    This heuristic aims to guide the planner away from decompositions that require implicit
    "cleanup" or "setup" actions (like putting down an object to free a hand) that the 
    context-free TDG heuristic misses.
    """
    def __init__(self, name="BarmanGemini3ProHeuristic1"):
        # Correctly pass name as keyword arg to avoid 'use_satis' mixup
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        # Standard TDG initialization
        h_value = super().initialize(model, initial_node)
        
        # 1. Build a map of (predicate, arg_name) -> fact ID
        for fact in model.facts:
            # Normalize string: lowercase, remove commas/parens/brackets
            # Barman domain typically uses "+predicate" in SAS
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
                
            predicate = parts[0]
            args = tuple(parts[1:])
            
            key = (predicate, args)
            self.fact_map[key] = fact.global_id

        # Helper to safely find a fact ID from map
        def get_fact_id(pred, args_tuple):
            # Check for standard and "+"-prefixed versions
            key_std = (pred, args_tuple)
            key_plus = ("+" + pred, args_tuple)
            if key_plus in self.fact_map:
                return self.fact_map[key_plus]
            if key_std in self.fact_map:
                return self.fact_map[key_std]
            return None

        # 2. Identify Optimistic Tasks and Calculate Penalties
        for node in self.and_or_graph.nodes:
            if node and node.content_type == ContentType.ABSTRACT_TASK:
                task_name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                parts = task_name_clean.split()
                if not parts:
                    continue
                
                task_type = parts[0]
                
                # We need to find the "optimistic cost" (node.value) and compare it 
                # to the cost of alternative methods to determine the penalty.
                # However, for efficiency, we can often assume the optimistic cost is 0 
                # (doing nothing) vs cost > 0.
                
                # --- clean-shaker / clean-shot (from v0) ---
                if "achievecleanshaker" in task_type or "achievecleanshot" in task_type:
                    # Find cheapest LOCAL decomposition (usually 0)
                    # Find alternative cheapest (usually > 0)
                    # For simplicity/robustness, we compute the penalty gap dynamically
                    cheapest = node.value
                    alternative = float('inf')
                    for child in node.successors:
                        if child.value > cheapest:
                            alternative = min(alternative, child.value)
                    
                    if alternative < float('inf'):
                        penalty = alternative - cheapest
                        if len(parts) > 1:
                            target_arg = parts[1]
                            fact_id = get_fact_id("clean", (target_arg,))
                            if fact_id is not None:
                                self.optimistic_tasks[node.ID] = {
                                    "type": "simple_fact",
                                    "penalty": penalty,
                                    "required_fact": fact_id
                                }

                # --- hand-empty ---
                # Task: AchieveHandEmpty(hand)
                # Optimistic: Already empty (Cost 0)
                # Reality: If not empty, must put down (Cost > 0)
                elif "achievehandempty" in task_type:
                    cheapest = node.value
                    alternative = float('inf')
                    for child in node.successors:
                        if child.value > cheapest:
                            alternative = min(alternative, child.value)
                    
                    if alternative < float('inf'):
                        penalty = alternative - cheapest
                        if len(parts) > 1:
                            hand_arg = parts[1]
                            fact_id = get_fact_id("handempty", (hand_arg,))
                            if fact_id is not None:
                                self.optimistic_tasks[node.ID] = {
                                    "type": "simple_fact",
                                    "penalty": penalty,
                                    "required_fact": fact_id
                                }

                # --- on-table ---
                # Task: AchieveOnTable(obj)
                # Optimistic: Already on table (Cost 0)
                # Reality: If held, must put down.
                elif "achieveontable" in task_type:
                    cheapest = node.value
                    alternative = float('inf')
                    for child in node.successors:
                        if child.value > cheapest:
                            alternative = min(alternative, child.value)
                    
                    if alternative < float('inf'):
                        penalty = alternative - cheapest
                        if len(parts) > 1:
                            obj_arg = parts[1]
                            fact_id = get_fact_id("ontable", (obj_arg,))
                            if fact_id is not None:
                                self.optimistic_tasks[node.ID] = {
                                    "type": "simple_fact", # Standard check: if not ontable, apply penalty
                                    "penalty": penalty,
                                    "required_fact": fact_id
                                }

                # --- holding ---
                # Task: AchieveHolding(hand, obj)
                # This is more complex.
                # State 1: Holding obj -> Cost 0 (NoOp)
                # State 2: Hand Empty -> Cost 1 (PickUp)
                # State 3: Holding other -> Cost 2 (PutDown other + PickUp obj)
                #
                # TDG Value is likely 0 (assuming State 1).
                # Start simple: If TDG value is 0, we check if we are holding it.
                elif "achieveholding" in task_type:
                     # This usually requires specific logic because "holding" takes 2 args (hand, obj)
                     # or maybe just (obj)? In Barman, it's typically (hand, obj).
                     # Let's check parts.
                     if len(parts) > 2:
                         hand_arg = parts[1]
                         obj_arg = parts[2]
                         
                         fact_id_holding = get_fact_id("holding", (hand_arg, obj_arg))
                         fact_id_empty = get_fact_id("handempty", (hand_arg,))
                         
                         # We need to calculate penalties relative to the TDG value.
                         # Assuming TDG sees "0 cost" for holding.
                         
                         if fact_id_holding is not None: 
                             # We register a complex check
                             self.optimistic_tasks[node.ID] = {
                                 "type": "holding_check",
                                 "base_val": node.value, 
                                 # We assume standard costs: pickup=1, putdown=1. 
                                 # If TDG=0, then:
                                 #   If holding: 0 (Diff 0)
                                 #   If empty: 1 (Diff 1)
                                 #   If full: 2 (Diff 2)
                                 "holding_fact": fact_id_holding,
                                 "empty_fact": fact_id_empty,
                                 "cost_holding": 0,
                                 "cost_empty": 1,
                                 "cost_full": 2
                             }

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state # bitvector
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                type_ = info["type"]
                
                if type_ == "simple_fact":
                    required_fact_id = info["required_fact"]
                    if not (state & (1 << required_fact_id)):
                        h_value += info["penalty"]
                
                elif type_ == "holding_check":
                    base_tdg = info["base_val"]
                    holding_fid = info["holding_fact"]
                    empty_fid = info["empty_fact"]
                    
                    is_holding = (state & (1 << holding_fid))
                    is_empty = (empty_fid is not None) and (state & (1 << empty_fid))
                    
                    real_cost = 0
                    if is_holding:
                        real_cost = info["cost_holding"] # 0
                    elif is_empty:
                        real_cost = info["cost_empty"] # 1
                    else:
                        real_cost = info["cost_full"] # 2
                        
                    # The penalty is the difference between real reality and what TDG thought (base_tdg)
                    # Usually base_tdg is 0. If base_tdg is 1 (e.g. TDG knows pickup is needed?), 
                    # we calculate diff.
                    penalty = max(0, real_cost - base_tdg)
                    h_value += penalty
        
        self.update_info(h_value)
        return h_value
