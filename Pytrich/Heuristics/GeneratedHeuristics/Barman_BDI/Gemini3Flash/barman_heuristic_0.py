from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class BarmanGemini3FlashHeuristic0(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v0 heuristic for Barman-BDI.
    Focuses on:
    1. Hand status (holding/empty).
    2. Shaker status (empty/contains).
    3. Container contents.
    """
    def __init__(self, name="BarmanGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'):
                predicate = predicate[1:]
            
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if len(parts) < 2: continue
            
            task_type = parts[0]
            if "grasp" in task_type and len(parts) >= 3:
                # grasp hand obj
                self.optimistic_tasks[node.ID] = {"type": "grasp", "hand": parts[1], "obj": parts[2]}
            elif "pour" in task_type and len(parts) >= 4:
                # pour hand source target
                self.optimistic_tasks[node.ID] = {"type": "pour", "hand": parts[1], "source": parts[2], "target": parts[3]}
            elif "shaking" in task_type and len(parts) >= 3:
                # shaking shaker shot1 shot2 hand1 hand2
                self.optimistic_tasks[node.ID] = {"type": "shake", "shaker": parts[1]}

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                t_type = info["type"]
                
                if t_type == "grasp":
                    # Penalty if hand not empty
                    empty_fid = self.fact_map.get(("hand_empty", (info["hand"],)))
                    if empty_fid is not None and not (state & (1 << empty_fid)):
                        h_value += 2.0
                        
                elif t_type == "pour":
                    # Penalty if not holding source
                    holding_fid = self.fact_map.get(("holding", (info["hand"], info["source"])))
                    if holding_fid is not None and not (state & (1 << holding_fid)):
                        h_value += 3.0
                        
                elif t_type == "shake":
                    # Penalty if shaker not closed (if applicable) or empty
                    empty_fid = self.fact_map.get(("shaker_empty", (info["shaker"],)))
                    if empty_fid is not None and (state & (1 << empty_fid)):
                        h_value += 5.0
        
        self.update_info(h_value)
        return h_value
