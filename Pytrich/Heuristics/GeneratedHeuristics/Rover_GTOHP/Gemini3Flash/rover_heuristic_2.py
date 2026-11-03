from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RoverGemini3FlashHeuristic2(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v2 heuristic for Rover domain.
    Waypoint-Goal awareness:
    - Identifies waypoints that need soil/rock samples communicated.
    - Penalizes sampling at non-goal waypoints.
    """
    def __init__(self, name="RoverGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_waypoints = set()
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
                    if parts and "data" in parts[0] and len(parts) >= 2:
                        self.goal_waypoints.add(parts[1])
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            if ("sample_soil" in task_type or "sample_rock" in task_type) and len(parts) >= 4:
                # sample_soil/rock rover store waypoint
                self.optimistic_tasks[node.ID] = {
                    "type": "sample",
                    "waypoint": parts[3]
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
                if info["waypoint"] not in self.goal_waypoints:
                    h_value += 5.0

        self.update_info(h_value)
        return h_value
