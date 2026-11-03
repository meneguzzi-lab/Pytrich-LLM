from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RoverGemini3FlashHeuristic3(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v3 heuristic for Rover.
    - Waypoint-Goal awareness (sample).
    - Objective-Goal awareness (take_image).
    """
    def __init__(self, name="RoverGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_waypoints = set()
        self.goal_objectives = set()
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            predicate = parts[0].strip('+-')
            self.fact_map[(predicate, tuple(parts[1:]))] = fact.global_id

        # Extract goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                fact = next((f for f in model.facts if f.global_id == fid), None)
                if fact:
                    name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                    parts = name_clean.split()
                    if parts:
                        if "soil_data" in parts[0] or "rock_data" in parts[0]:
                            self.goal_waypoints.add(parts[1])
                        elif "image_data" in parts[0]:
                            self.goal_objectives.add(parts[1])
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            if ("sample_soil" in task_type or "sample_rock" in task_type) and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {"type": "sample", "val": parts[3]}
            elif "take_image" in task_type and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {"type": "image", "val": parts[3]}

        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                if info["type"] == "sample":
                    if info["val"] not in self.goal_waypoints: h_value += 10.0
                elif info["type"] == "image":
                    if info["val"] not in self.goal_objectives: h_value += 10.0
        self.update_info(h_value)
        return h_value
