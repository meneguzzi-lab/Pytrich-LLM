from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3FlashHeuristic3(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v3 heuristic for Robot.
    - Delivery Focus: Penalizes moving to the wrong room when carrying a goal object.
    """
    def __init__(self, name="RobotGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_in = {} # obj -> target_room
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
                    if parts and parts[0].strip('+-') == "in" and len(parts) == 3:
                        self.goal_in[parts[1]] = parts[2]
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            task_type = parts[0]
            if "move" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "move", "dst": parts[2]}
        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                # Check if holding something
                for obj, target_room in self.goal_in.items():
                    holding_fid = self.fact_map.get(("holding", (obj,)))
                    if holding_fid is not None and (state & (1 << holding_fid)):
                        # If moving and dst is not target and dst is not corridor
                        if info["dst"] != target_room and info["dst"] != 'c':
                            h_value += 5.0
                        break # Only one arm anyway
        self.update_info(h_value)
        return h_value
