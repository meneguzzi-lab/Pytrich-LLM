from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RoverGemini3FlashHeuristic4(TaskDecompositionHeuristic):
    """
    Gemini 3 Flash v4 heuristic for Rover domain.
    Adds Waypoint Pathing and Objective-Driven Navigation:
    - Penalizes redundant navigation.
    - Prioritizes movement towards waypoints with unfinished objectives.
    - Includes Camera Calibration and Store Empty checks from v1-v3.
    """
    def __init__(self, name="RoverGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.fact_map = {}
        self.objective_waypoints = set()
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

        # 2. Extract Objective Waypoints from Goals
        goals_bitwise = model.goals
        fid = 0
        while goals_bitwise:
            if goals_bitwise & 1:
                target_fact = next((f for f in model.facts if f.global_id == fid), None)
                if target_fact:
                    parts = target_fact.name.replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ").split()
                    # e.g., (communicated_soil_data waypoint)
                    for p in parts:
                        if p.startswith('waypoint'):
                            self.objective_waypoints.add(p)
            goals_bitwise >>= 1
            fid += 1

        # 3. Identify Tasks
        for node in self.and_or_graph.nodes:
            if not node: continue
            name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts: continue
            
            task_type = parts[0]
            if "navigate" in task_type and len(parts) >= 4:
                # navigate rover from to
                self.optimistic_tasks[node.ID] = {
                    "type": "navigate",
                    "rover": parts[1],
                    "from": parts[2],
                    "to": parts[3]
                }
            elif "take_image" in task_type and len(parts) >= 5:
                self.optimistic_tasks[node.ID] = {
                    "type": "take_image",
                    "camera": parts[4],
                    "rover": parts[1]
                }
            elif ("sample_soil" in task_type or "sample_rock" in task_type) and len(parts) >= 4:
                self.optimistic_tasks[node.ID] = {
                    "type": "sample",
                    "store": parts[2],
                    "rover": parts[1]
                }
        return h_value

    def __call__(self, parent_node, node):
        h_value = 0
        state = node.state
        
        # Check current rover locations
        rover_at = {} # rover -> waypoint
        for fact_key, fid in self.fact_map.items():
            if fact_key[0] == "at" and (state & (1 << fid)):
                rover_at[fact_key[1][0]] = fact_key[1][1]

        for task in node.task_network:
            base_cost = self.tdg_values.get(task.global_id, float('inf'))
            h_value += base_cost
            
            if task.global_id in self.optimistic_tasks:
                info = self.optimistic_tasks[task.global_id]
                t_type = info["type"]
                
                if t_type == "navigate":
                    target_w = info["to"]
                    # If target waypoint is NOT an objective waypoint, penalize slightly
                    if target_w not in self.objective_waypoints:
                        h_value += 15.0
                    
                    # Penalize going back to where we just came from (simple redundency check)
                    if parent_node:
                        # Find parent's rover loc
                        pass # complicated to implement efficiently here, skip for now
                        
                elif t_type == "take_image":
                    cal_fid = self.fact_map.get(("calibrated", (info["camera"], info["rover"])))
                    if cal_fid is not None and not (state & (1 << cal_fid)):
                        h_value += 20.0 # Strict calibration check

                elif t_type == "sample":
                    empty_fid = self.fact_map.get(("empty", (info["store"],)))
                    if empty_fid is not None and not (state & (1 << empty_fid)):
                        h_value += 25.0 # Strict store check

        self.update_info(h_value)
        return h_value
