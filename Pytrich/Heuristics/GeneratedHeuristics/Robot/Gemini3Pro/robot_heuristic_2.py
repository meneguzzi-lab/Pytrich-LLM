from Pytrich.Heuristics.tdg_heuristic import TaskDecompositionHeuristic
from Pytrich.ProblemRepresentation.and_or_graph import ContentType

class RobotGemini3ProHeuristic2(TaskDecompositionHeuristic):
    """
    Round 2 state-aware heuristic for Robot (v2).
    Penalizes:
    1. MOVE tasks if the connecting door is closed.
    2. PICKUP tasks if:
       - Arm is not empty.
       - Robot is not in the same room as the object.
       - Object is not in that room.
    """
    def __init__(self, name="RobotGemini3ProHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}
        self.optimistic_tasks = {}

    def initialize(self, model, initial_node):
        h_value = super().initialize(model, initial_node)
        
        # 1. Build Fact Map
        for fact in model.facts:
            name_clean = fact.name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
            parts = name_clean.split()
            if not parts:
                continue
            predicate = parts[0]
            args = tuple(parts[1:])
            key = (predicate, args)
            self.fact_map[key] = fact.global_id

        def get_fact_id(pred, args_tuple):
            key_std = (pred, args_tuple)
            key_plus = ("+" + pred, args_tuple)
            if key_plus in self.fact_map:
                return self.fact_map[key_plus]
            if key_std in self.fact_map:
                return self.fact_map[key_std]
            return None

        # 2. Identify Optimistic Tasks
        for node in self.and_or_graph.nodes:
            if node and node.content_type in [ContentType.ABSTRACT_TASK, ContentType.OPERATOR]:
                task_name_clean = node.str_name.lower().replace("(", " ").replace(")", " ").replace(",", " ").replace("[", " ").replace("]", " ")
                parts = task_name_clean.split()
                if not parts:
                    continue
                
                task_type = parts[0]
                
                # We target: pickup, move, achieve-goals-pickup
                if "pickup" in task_type and len(parts) > 2:
                    # pickup o1 r4
                    obj, room = parts[1], parts[2]
                    self.optimistic_tasks[node.ID] = {
                        "type": "pickup",
                        "obj": obj,
                        "room": room,
                        "fid_armempty": get_fact_id("armempty", ()),
                        "fid_rloc_room": get_fact_id("rloc", (room,)),
                        "fid_obj_in_room": get_fact_id("in", (obj, room))
                    }
                elif "move" in task_type and len(parts) > 3:
                    # move r1 r4 d14
                    door = parts[3]
                    self.optimistic_tasks[node.ID] = {
                        "type": "move",
                        "door": door,
                        "fid_closed": get_fact_id("closed", (door,))
                    }
                elif "achieve-goals-pickup" in task_type and len(parts) > 2:
                    # achieve-goals-pickup_split_split r1 o4
                    # Abstract task: target_room, obj
                    t_room, obj = parts[1], parts[2]
                    # This task usually involves getting it. 
                    # If arm is already full with something else, it's a blocker.
                    self.optimistic_tasks[node.ID] = {
                        "type": "achieve_pickup",
                        "obj": obj,
                        "target_room": t_room,
                        "fid_armempty": get_fact_id("armempty", ())
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
                
                if t_type == "pickup":
                    # 1. Arm empty check
                    fid_arm = info["fid_armempty"]
                    if fid_arm is not None and not (state & (1 << fid_arm)):
                        h_value += 1.0 # Need to putdown something
                    
                    # 2. Robot loc check
                    fid_rloc = info["fid_rloc_room"]
                    if fid_rloc is not None and not (state & (1 << fid_rloc)):
                        h_value += 1.0 # Need to move here
                        
                    # 3. Object loc check
                    fid_in = info["fid_obj_in_room"]
                    if fid_in is not None and not (state & (1 << fid_in)):
                        h_value += 10.0 # Object not here! Massive penalty for wrong branch.

                elif t_type == "move":
                    # Check door
                    fid_closed = info["fid_closed"]
                    if fid_closed is not None and (state & (1 << fid_closed)):
                        h_value += 1.0 # Need to open door

                elif t_type == "achieve_pickup":
                    # If holding something else, must putdown.
                    # (Hard to check 'not armempty AND not holding obj')
                    # But if not armempty, it's a cost anyway.
                    fid_arm = info["fid_armempty"]
                    if fid_arm is not None and not (state & (1 << fid_arm)):
                        # Check if holding the DESIRED obj
                        # We don't have fid_holding_obj in 'achieve_pickup' yet but let's just 
                        # skip penalty if it looks like we might already have it? 
                        # Actually achieve_pickup usually starts with armempty.
                        h_value += 1.0

        self.update_info(h_value)
        return h_value
