from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3FlashHeuristic5(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v5 heuristic for Robot domain.
    Adds Room Clearance Priority:
    - Penalizes leaving a room if it contains goal objects that haven't been moved.
    - Discourages picking up objects already in their goal room (v4).
    - Includes Door-Status and Arm-Empty constraints from v1-v3.
    """
    def __init__(self, name="RobotGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_in = {} 
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

        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                target_fact = next((f for f in model.facts if f.global_id == fid), None)
                if target_fact:
                    parts = target_fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ").split()
                    if parts[0].strip('+-') == "in" and len(parts) >= 3:
                        self.goal_in[parts[1]] = parts[2]
            goals_bitwise >>= 1
            fid += 1

        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            task_type = parts[0]
            if "move" in task_type and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {"type": "move", "src": parts[1], "dst": parts[2], "door": parts[3]}
            elif "pickup" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {"type": "pickup", "obj": parts[1], "room": parts[2]}
        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        # Current status
        holding_obj = None
        for fact_key, fid in self.fact_map.items():
            if fact_key[0] == "holding" and (state & (1 << fid)):
                holding_obj = fact_key[1][0]; break

        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                t_type = info["type"]
                
                if t_type == "move":
                    # 1. Room Clearance (v5 Enhancement)
                    if not holding_obj:
                        src_room = info["src"]
                        # Check for ANY object currently in src_room that should be somewhere else
                        for obj, goal_room in self.goal_in.items():
                            in_fid = self.fact_map.get(("in", (obj, src_room)))
                            if in_fid is not None and (state & (1 << in_fid)) and goal_room != src_room:
                                h_value += 60.0 # Leaving room while goals remain is very bad
                                break
                    
                    # 2. Door/Efficiency (v1-v4)
                    door_fid = self.fact_map.get(("closed", (info["door"],)))
                    if door_fid is not None and (state & (1 << door_fid)): h_value += 40.0
                    
                    if holding_obj:
                        goal_room = self.goal_in.get(holding_obj)
                        if goal_room and info["dst"] != goal_room and goal_room == info["src"]:
                            h_value += 50.0 
                        
                elif t_type == "pickup":
                    obj, room = info["obj"], info["room"]
                    if self.goal_in.get(obj) == room: h_value += 50.0 
                    if holding_obj: h_value += 30.0 
                    in_fid = self.fact_map.get(("in", (obj, room)))
                    if in_fid is not None and not (state & (1 << in_fid)): h_value += 45.0

        self.update_info(h_value)
        return h_value
