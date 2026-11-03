from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RoverGTOHPGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.

    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is the requirement for rovers to visit specific waypoints 
    for sampling or imaging, and then navigate to a line-of-sight location with a lander for data transmission.
    Each data task follows a strict sequence: Move to Target -> Acquire Data -> Move to LOS -> Communicate.

    Admissible Lower Bounds:
    1. Goal Progress Term (h_goals): Each unsatisfied communication goal requires at least one 'communicate' action. 
       If the data isn't currently held by any rover, a 'sample' or 'take_image' action is also mandatory. 
       For images, if no rover is calibrated, a 'calibrate' action is also required.
    2. Navigation Term (h_nav): For the current active task in the network, the rover must perform at least 
       dist(current_pos, target_waypoint) navigation actions. This is additive as 'navigate' is a distinct action type.
    """

    def __init__(self, rover_gemini3flash_heuristic_10="rover_gemini3flash_heuristic_10"):
        super().__init__(rover_gemini3flash_heuristic_10)
        self.at_ids = []  # [rover_idx][waypoint_idx]
        self.calibrated_ids = [] # [rover_idx][objective_idx]
        self.goal_info = [] # List of dicts
        self.task_to_target = {} # global_id -> (type, index, rover_idx_or_None)
        self.rover_to_idx = {}
        self.waypoint_to_idx = {}
        self.obj_to_idx = {}
        self.dist_matrix = []
        self.dist_to_obj = [] # [waypoint_idx][obj_idx]

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify Objects
        rover_names = []
        waypoint_names = []
        obj_names = []
        camera_to_obj = {} # camera -> objective
        camera_on_rover = {} # camera -> rover

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                if args[0] not in rover_names: rover_names.append(args[0])
                if args[1] not in waypoint_names: waypoint_names.append(args[1])
            elif pred == "calibration_target":
                camera_to_obj[args[0]] = args[1]
                if args[1] not in obj_names: obj_names.append(args[1])
            elif pred == "on_board":
                camera_on_rover[args[0]] = args[1]

        self.rover_to_idx = {name: i for i, name in enumerate(rover_names)}
        self.waypoint_to_idx = {name: i for i, name in enumerate(waypoint_names)}
        self.obj_to_idx = {name: i for i, name in enumerate(obj_names)}
        num_r, num_w, num_o = len(rover_names), len(waypoint_names), len(obj_names)

        # 2. Build Fact Maps
        self.at_ids = [[-1] * num_w for _ in range(num_r)]
        self.goal_to_analysis_ids = {}
        self.goal_to_calibration_ids = {}
        self.goal_ids = []
        
        analysis_map = {} # (pred, target) -> [ids]
        calibrated_map = {} # (rover, obj) -> [ids]

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_ids[self.rover_to_idx[args[0]]][self.waypoint_to_idx[args[1]]] = f.global_id
            elif pred in ["have_soil_analysis", "have_rock_analysis"]:
                key = (pred, args[1])
                if key not in analysis_map: analysis_map[key] = []
                analysis_map[key].append(f.global_id)
            elif pred == "have_image":
                key = (pred, args[1]) # obj is index 1
                if key not in analysis_map: analysis_map[key] = []
                analysis_map[key].append(f.global_id)
            elif pred == "calibrated":
                # Search camera info
                for cam, obj in camera_to_obj.items():
                    if camera_on_rover.get(cam) == args[1]: # args[1] is rover
                        o_key = (args[1], obj)
                        if o_key not in calibrated_map: calibrated_map[o_key] = []
                        calibrated_map[o_key].append(f.global_id)

        # 3. Parse Goals
        explicit_goals = model.state_explicit_repr(model.goals)
        for g_name in explicit_goals:
            pred, args = self._parse_fact_name(g_name)
            g_id = -1
            for f in model.facts: 
                if f.name == g_name: 
                    g_id = f.global_id
                    break
            if g_id == -1: continue
            
            self.goal_ids.append(g_id)
            info = {"goal_id": g_id, "analysis_ids": [], "calib_ids": []}
            if pred == "communicated_soil_data":
                info["analysis_ids"] = analysis_map.get(("have_soil_analysis", args[0]), [])
            elif pred == "communicated_rock_data":
                info["analysis_ids"] = analysis_map.get(("have_rock_analysis", args[0]), [])
            elif pred == "communicated_image_data":
                info["analysis_ids"] = analysis_map.get(("have_image", args[0]), [])
                # obj = args[0]. Find any rover that has calibrated camera for this obj
                for r_name in rover_names:
                    info["calib_ids"].extend(calibrated_map.get((r_name, args[0]), []))
            self.goal_info.append(info)

        # 4. Navigation matrix
        adj = [[] for _ in range(num_w)]
        for f in model.facts:
            if "can_traverse" in f.name and f.global_id <= model.ifacts_end:
                if (initial_node.state >> f.global_id) & 1:
                    p, args = self._parse_fact_name(f.name)
                    u, v = self.waypoint_to_idx[args[1]], self.waypoint_to_idx[args[2]]
                    if v not in adj[u]: adj[u].append(v)

        self.dist_matrix = [[99] * num_w for _ in range(num_w)]
        for i in range(num_w):
            self.dist_matrix[i][i] = 0
            q = deque([(i, 0)])
            visited = {i}
            while q:
                u, d = q.popleft()
                self.dist_matrix[i][u] = d
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append((v, d+1))

        self.dist_to_obj = [[99] * num_o for _ in range(num_w)]
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "visible_from":
                o_idx = self.obj_to_idx.get(args[0])
                w_idx = self.waypoint_to_idx.get(args[1])
                if o_idx is not None and w_idx is not None:
                    for start_w in range(num_w):
                        self.dist_to_obj[start_w][o_idx] = min(self.dist_to_obj[start_w][o_idx], self.dist_matrix[start_w][w_idx])

        # 5. Task Targets
        for t in list(model.abstract_tasks) + list(model.operators):
            p, args = self._parse_fact_name("+" + t.name)
            t_info = (None, None, None) # Type, TargetIdx, RoverIdx
            r_idx = next((self.rover_to_idx[a] for a in args if a in self.rover_to_idx), None)
            if args:
                if args[-1] in self.waypoint_to_idx: t_info = ("waypoint", self.waypoint_to_idx[args[-1]], r_idx)
                elif args[0] in self.obj_to_idx: t_info = ("objective", self.obj_to_idx[args[0]], r_idx)
            self.task_to_target[t.global_id] = t_info

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network) -> int:
        h = 0
        # Term 1: Goals
        for g in self.goal_info:
            if not (state & (1 << g["goal_id"])):
                h += 1  # communicate
                if not any((state & (1 << a_id)) for a_id in g["analysis_ids"]):
                    h += 1  # sample/image
                    if g["calib_ids"] and not any((state & (1 << c_id)) for c_id in g["calib_ids"]):
                        h += 1  # calibrate
        
        # Term 2: Current Task Navigation
        if task_network:
            task = task_network[0]
            t_type, t_idx, r_fixed_idx = self.task_to_target.get(task.global_id, (None, None, None))
            if t_type:
                min_dist = 99
                ro_indices = [r_fixed_idx] if r_fixed_idx is not None else range(len(self.at_ids))
                for ri in ro_indices:
                    for wi in range(len(self.at_ids[ri])):
                        if (state >> self.at_ids[ri][wi]) & 1:
                            d = self.dist_matrix[wi][t_idx] if t_type == "waypoint" else self.dist_to_obj[wi][t_idx]
                            if d < min_dist: min_dist = d
                            break
                if min_dist < 99: h += min_dist

        # Tie-breaking for guidance
        h_weighted = h + 0.001 * len(task_network)
        return int(h_weighted)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
