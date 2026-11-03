from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3FlashHeuristic4(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v4 heuristic for Robot domain.
    Adds Room-to-Room Efficiency and Goal-Room Awareness:
    - Penalizes redundant movements between rooms.
    - Discourages picking up objects already in their goal room.
    - Includes Door-Status and Arm-Empty constraints from v1-v3.
    """
    def __init__(self, name="RobotGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.fact_map = {}
        self.goal_in = {} # obj -> room
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map
        for fact in model.facts:
            name_clean = fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            predicate = parts[0]
            if predicate.startswith('+') or predicate.startswith('-'): predicate = predicate[1:]
            args = tuple(parts[1:])
            self.fact_map[(predicate, args)] = fact.global_id

        # 2. Extract Goal Locations
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

        # 3. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            if "move" in task_type and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {
                    "type": "move", "src": parts[1], "dst": parts[2], "door": parts[3]
                }
            elif "pickup" in task_type and len(parts) >= 3:
                self.optimistic_tasks[node.ID] = {
                    "type": "pickup", "obj": parts[1], "room": parts[2]
                }
        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        # Current robot status
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
                    door_fid = self.fact_map.get(("closed", (info["door"],)))
                    if door_fid is not None and (state & (1 << door_fid)):
                        h_value += 30.0 # Strict door check
                    
                    # Room-to-Room Efficiency (v4)
                    if holding_obj:
                        goal_room = self.goal_in.get(holding_obj)
                        if goal_room and info["dst"] != goal_room and goal_room == info["src"]:
                            h_value += 40.0 # Moving AWAY from goal room while holding item
                    else:
                        # Empty handed move
                        # Are there goal items in current room that we should have picked up?
                        items_in_src = [o for o, r in self.goal_in.items() if r != info["src"]] # items that need to be moved FROM here
                        # Simplified: check for ANY "in" fact in current room for items not in their goal room
                        for o, gr in self.goal_in.items():
                            in_fid = self.fact_map.get(("in", (o, info["src"])))
                            if in_fid is not None and (state & (1 << in_fid)) and gr != info["src"]:
                                h_value += 20.0 # Leaving room while goal objects are still there to be picked up
                                break
                        
                elif t_type == "pickup":
                    obj, room = info["obj"], info["room"]
                    # Goal-Room Awareness (v4)
                    if self.goal_in.get(obj) == room:
                        h_value += 50.0 # Don't pickup if already in goal room

                    # Standard checks (v1)
                    if holding_obj: h_value += 25.0 # Arm not empty
                    
                    in_fid = self.fact_map.get(("in", (obj, room)))
                    if in_fid is not None and not (state & (1 << in_fid)):
                        h_value += 40.0 # Obj not in room

        self.update_info(h_value)
        return h_value
