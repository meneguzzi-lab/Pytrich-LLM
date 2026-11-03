from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3FlashHeuristic2(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v2 heuristic for Robot domain.
    Goal-Room Awareness:
    - Identifies rooms involved in goals (target locations).
    - Penalizes moving to rooms that are not goal targets.
    """
    def __init__(self, name="RobotGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_rooms = set()
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map & Extract Goals
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'):
                predicate = predicate[1:]
                
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id

        # Extract goals to find goal rooms
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                # Find the fact
                fact = next((f for f in model.facts if f.global_id == fid), None)
                if fact:
                    name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                    parts = name_clean.split()
                    if parts and parts[0].strip('+-') == "in" and len(parts) == 3:
                        self.goal_rooms.add(parts[2])
            goals_bitwise >>= 1
            fid += 1

        # 2. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            
            if "move" in task_type and len(parts) >= 3:
                # move src dst door
                self.optimistic_tasks[node.ID] = {
                    "type": "move",
                    "dst": parts[2]
                }
            elif "pickup" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {
                    "type": "pickup",
                    "obj": parts[1],
                    "room": parts[2]
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
                
                if t_type == "move":
                    # Penalty if moving to a room that is not a goal room (and not corridor 'c' which is connector)
                    # Note: corridor 'c' is often neutral.
                    if info["dst"] not in self.goal_rooms and info["dst"] != 'c':
                        h_value += 5.0
                        
                elif t_type == "pickup":
                    # Basic state checks from v1
                    arm_fid = self.fact_map.get(("armempty", ()))
                    if arm_fid is not None and not (state & (1 << arm_fid)):
                        h_value += 1.0
                    
                    loc_fid = self.fact_map.get(("rloc", (info["room"],)))
                    if loc_fid is not None and not (state & (1 << loc_fid)):
                        h_value += 1.0
                        
                    in_fid = self.fact_map.get(("in", (info["obj"], info["room"])))
                    if in_fid is not None and not (state & (1 << in_fid)):
                        h_value += 2.0

        self.update_info(h_value)
        return h_value
