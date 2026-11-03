from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3ProHeuristic0(TaskDecompositionHeuristic):
    """
    A heuristic for the Barman domain that improves upon the Task Decomposition Graph (TDG) heuristic
    by incorporating state context.
    
    The standard TDG heuristic is "optimistic" and context-free. It assumes that if a task *can* be 
    decomposed into a low-cost method (e.g., a method requiring no actions, cost 0), it *will* be, 
    regardless of the current state.
    
    For example, in Barman, `AchieveCleanShaker` can be achieved by:
    1. Doing nothing (if the shaker is already clean) -> Cost 0
    2. Cleaning the shaker (if it is dirty) -> Cost > 0
    
    TDG will always take the minimum, which is 0. This heuristic checks the current state. 
    If the "optimistic" condition (shaker is clean) is not met, it adds a penalty equivalent 
    to the cost of the actual work required.
    """
    def __init__(self, name="BarmanGemini3ProHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic.
        1. Call super().initialize() to build the TDG and compute base values.
        2. Map predicate/arg pairs to global fact IDs for fast state lookups.
        3. Identify "optimistic" tasks and pre-calculate the penalty for when the optimistic case fails.
        """
        # Standard TDG initialization
        h_value = super().initialize(model, initial_node)
        
        # 1. Build a map of (predicate, arg_name) -> fact ID (index in the state bitvector)
        # fact.name is a string (e.g., "+clean(shaker1)" or "clean shaker1")
        for fact in model.facts:
            # Normalize string: lowercase, remove commas/parens/brackets
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
                
            predicate = parts[0]
            # args are the rest
            args = tuple(parts[1:])
            
            key = (predicate, args)
            self.fact_map[key] = fact.global_id

        # 2. Identify Optimistic Tasks and Calculate Penalties
        for node in self.and_or_graph.nodes:
            if node and node.content_type == ContentType.ABSTRACT_TASK:
                # AndOrNode uses str_name, not name
                task_name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                parts = task_name_clean.split()
                if not parts:
                    continue
                
                task_type = parts[0]
                
                # Setup common logic for finding alternative cost
                # We want: min(cost(child)) where cost(child) > node.value (optimistic cost)
                # Usually node.value is 0 for these tasks.
                
                if "achievecleanshaker" in task_type:
                    alternative_cost = float('inf')
                    for child in node.successors:
                        if child.value > node.value:
                            alternative_cost = min(alternative_cost, child.value)
                    
                    if alternative_cost < float('inf'):
                        penalty = alternative_cost - node.value
                        # args: "achievecleanshaker shaker1" -> shaker1 is index 1
                        if len(parts) > 1:
                            shaker_arg = parts[1]
                            # Check for +clean first (common in grounded files), then clean
                            check_key_plus = ("+clean", (shaker_arg,))
                            check_key_std = ("clean", (shaker_arg,))
                            
                            fact_id = None
                            if check_key_plus in self.fact_map:
                                fact_id = self.fact_map[check_key_plus]
                            elif check_key_std in self.fact_map:
                                fact_id = self.fact_map[check_key_std]
                                
                            if fact_id is not None:
                                self.optimistic_tasks[node.ID] = {
                                    "penalty": penalty,
                                    "required_fact": fact_id
                                }

                elif "achievecleanshot" in task_type:
                     alternative_cost = float('inf')
                     for child in node.successors:
                        if child.value > node.value:
                            alternative_cost = min(alternative_cost, child.value)
                     
                     if alternative_cost < float('inf'):
                        penalty = alternative_cost - node.value
                        if len(parts) > 1:
                            shot_arg = parts[1]
                            check_key_plus = ("+clean", (shot_arg,))
                            check_key_std = ("clean", (shot_arg,))
                            
                            fact_id = None
                            if check_key_plus in self.fact_map:
                                fact_id = self.fact_map[check_key_plus]
                            elif check_key_std in self.fact_map:
                                fact_id = self.fact_map[check_key_std]

                            if fact_id is not None:
                                self.optimistic_tasks[node.ID] = {
                                    "penalty": penalty,
                                    "required_fact": fact_id
                                }

        return h_value

    def __call__(self, parent_node, node):
        """
        Compute heuristic value for a search node.
        """
        # 1. Get base TDG value (sum of TDG values of tasks in network)
        h_value = 0
        state = node.state # This is a bitvector (set of integers)
        
        for task in node.task_network:
            # Add base TDG cost
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            # Add context penalty
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                required_fact_id = info["required_fact"]
                
                # Check if the required fact is in the current state
                # node.state is a bitvector (int)
                if not (state & (1 << required_fact_id)):
                    # The "optimistic" 0-cost method (which requires this fact) is NOT applicable.
                    h_value += info["penalty"]
        
        self.update_info(h_value)
        return h_value
