from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3FlashHeuristic0(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v0 heuristic for Robot domain.
    Penalizes:
    1. move(src, dst, door) if door is CLOSED.
    2. pickup(obj, room) if:
        - arm is NOT empty.
        - robot is NOT in room.
        - object is NOT in room.
    """
    def __init__(self, name="RobotGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'):
                predicate = predicate[1:]
                
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id

        # 2. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            
            if "move" in task_type and len(parts) >= 4:
                # move src dst door
                self.optimistic_tasks[node.ID] = {
                    "type": "move",
                    "door": parts[3]
                }
            elif "pickup" in task_type and len(parts) >= 3:
                # pickup obj room
                self.optimistic_tasks[node.ID] = {
                    "type": "pickup",
                    "obj": parts[1],
                    "room": parts[2]
                }
            # Catch some abstract tasks
            elif "achieve-goals-pickup" in task_type and len(parts) >= 3:
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
                    # Penalty if door is closed
                    door_fid = self.fact_map.get(("closed", (info["door"],)))
                    if door_fid is not None and (state & (1 << door_fid)):
                        h_value += 10.0
                        
                elif t_type == "pickup":
                    # Penalty if arm not empty
                    arm_fid = self.fact_map.get(("armempty", ()))
                    if arm_fid is not None and not (state & (1 << arm_fid)):
                        h_value += 10.0
                    
                    # Penalty if robot not in room
                    loc_fid = self.fact_map.get(("rloc", (info["room"],)))
                    if loc_fid is not None and not (state & (1 << loc_fid)):
                        h_value += 10.0
                        
                    # Penalty if object not in room
                    in_fid = self.fact_map.get(("in", (info["obj"], info["room"])))
                    if in_fid is not None and not (state & (1 << in_fid)):
                        h_value += 20.0 

        self.update_info(h_value)
        return h_value
