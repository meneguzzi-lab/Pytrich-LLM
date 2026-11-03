from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3FlashHeuristic2(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v2 heuristic for Barman-BDI.
    Shaker-Goal awareness:
    - Identifies which beverages/cocktails are goals.
    - Penalizes operations that produce non-goal cocktails.
    """
    def __init__(self, name="BarmanGemini3FlashHeuristic2"):
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
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id

        # Extract goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                fact = next((f for f in model.facts if f.global_id == fid), None)
                if fact:
                    name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                    parts = name_clean.split()
                    if len(parts) >= 2 and parts[0].strip('+-') == "contains":
                        # contains[shot/shaker, beverage]
                        self.goal_cocktails.add(parts[2])
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) < 2: continue
            
            task_type = parts[0]
            # Tasks: do_shake[shaker, cocktail, hand1, hand2]
            if "shake" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "shake", "cocktail": parts[2]}
            # do_pour_shaker_to_shot[shaker, shot, cocktail, hand]
            elif "pour" in task_type and "shaker" in task_type and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {"type": "pour", "cocktail": parts[3]}

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                cocktail = info["cocktail"]
                
                # Penalty if producing a non-goal cocktail
                if cocktail not in self.goal_cocktails:
                    h_value += 10.0
        
        self.update_info(h_value)
        return h_value
