from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3FlashHeuristic4(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v4 heuristic for Barman-BDI.
    - Extracts goal cocktails from initial Task Network.
    """
    def __init__(self, name="BarmanGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_cocktails = set()
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            predicate = parts[0].strip('+-')
            self.fact_map[(predicate, tuple(parts[1:]))] = fact.global_id

        # Extract goals from initial task network (since Barman-BDI is task-based)
        for task in initial_node.task_network:
            name_clean = task.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            # Task: achievecontainsshotcocktail shot5 cocktail4
            for p in parts:
                if "cocktail" in p:
                    self.goal_cocktails.add(p)

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) < 2: continue
            
            task_type = parts[0]
            if "shake" in task_type and len(parts) >= 2:
                self.optimistic_tasks[node.ID] = {"type": "shake", "cocktail": parts[1]}
            elif "pour" in task_type and "shaker" in task_type and len(parts) >= 2:
                self.optimistic_tasks[node.ID] = {"type": "pour", "cocktail": parts[1]}

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                # If cocktail is not in initial goals, penalize
                if info["cocktail"] not in self.goal_cocktails:
                    h_value += 20.0 
        self.update_info(h_value)
        return h_value
