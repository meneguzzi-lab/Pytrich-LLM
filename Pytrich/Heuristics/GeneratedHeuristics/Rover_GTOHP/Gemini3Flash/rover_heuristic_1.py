from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RoverGemini3FlashHeuristic1(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v1 heuristic for Rover domain.
    Refined weights (lower) to avoid search distortion.
    """
    def __init__(self, name="RoverGemini3FlashHeuristic1"):
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
            if not parts: continue
            
            task_type = parts[0]
            
            if "take_image" in task_type and len(parts) >= 5:
                # take_image rover waypoint objective camera mode
                self.optimistic_tasks[node.ID] = {
                    "type": "take_image",
                    "camera": parts[4],
                    "rover": parts[1]
                }
            elif ("sample_soil" in task_type or "sample_rock" in task_type) and len(parts) >= 4:
                # sample_soil/rock rover store waypoint
                self.optimistic_tasks[node.ID] = {
                    "type": "sample",
                    "store": parts[2],
                    "rover": parts[1]
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
                t_type = info["type"]
                
                if t_type == "take_image":
                    cal_fid = self.fact_map.get(("calibrated", (info["camera"], info["rover"])))
                    if cal_fid is not None and not (state & (1 << cal_fid)):
                        h_value += 1.0 # Lowered from 10.0
                        
                elif t_type == "sample":
                    empty_fid = self.fact_map.get(("empty", (info["store"],)))
                    if empty_fid is not None and not (state & (1 << empty_fid)):
                        h_value += 1.0 # Lowered from 10.0

        self.update_info(h_value)
        return h_value
