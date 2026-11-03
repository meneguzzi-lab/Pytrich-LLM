import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    BOTTLENECK ANALYSIS:
    The serialization in this domain is dominated by the limited communication resources and the 
    sequential nature of rover navigation. Each rover must travel to specific waypoints to sample soil/rocks 
    or take images, and then travel to a waypoint visible to a lander to transmit that data.

    ADMISSIBLE LOWER BOUNDS:
    1. Mission Action Bound: Each uncommunicated data goal (soil, rock, image) requires a fixed sequence of 
       logical actions: 1 Sample/Image + 1 Communicate. This is a lower bound because these physical actions 
       are mandatory to satisfy the goal state.
    2. Navigation Shortest Path: The number of 'navigate' actions required to reach a target waypoint and then 
       a communication waypoint. Using BFS distances ensures this is the absolute minimum moves required.
    3. Preparation Bound: If a rover's store is full, an additional 'drop' action is required for soil/rock 
       collection. If a camera is not calibrated, a 'calibrate' action is required at a specific target waypoint.
    """

    def __init__(self, rover_gemini3flash_heuristic_1=None, name="RoverGTOHPGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.dist_map = {}  # (rover, wp1, wp2) -> min_nav_actions
        self.min_dist_comm = {}  # (rover, wp_source) -> min_dist_to_visible_lander
        self.fact_ids = {}  # pred -> tuple(args) -> id
        self.rover_capabilities = defaultdict(dict) # rover -> {'soil': bool, 'rock': bool, 'imaging': bool}
        self.rover_cameras = defaultdict(list) # rover -> [camera_names]
        self.camera_targets = {} # camera -> [waypoint_indices]
        self.obj_view_wps = defaultdict(list) # objective -> [waypoint_names]
        self.landers = [] # list of (lander_name, wp_name)
        self.rovers = []
        self.waypoints = []
        self.at_fact_ids = {} # rover -> {wp: fid}
        
    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _preprocess(self):
        # 1. Map Facts and extract Objects
        self.fact_ids = defaultdict(dict)
        temp_can_traverse = defaultdict(list)
        temp_visible = defaultdict(set)
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            self.fact_ids[pred][tuple(args)] = fact.global_id

            if pred == "at_lander": self.landers.append((args[0], args[1]))
            elif pred == "at" and len(args) == 2: 
                if args[0] not in self.rovers: self.rovers.append(args[0])
                if args[1] not in self.waypoints: self.waypoints.append(args[1])
            elif pred == "can_traverse": temp_can_traverse[args[0]].append((args[1], args[2]))
            elif pred == "visible": temp_visible[args[0]].add(args[1])
            elif pred == "on_board": self.rover_cameras[args[1]].append(args[0])
            elif pred == "equipped_for_soil_analysis": self.rover_capabilities[args[0]]['soil'] = True
            elif pred == "equipped_for_rock_analysis": self.rover_capabilities[args[0]]['rock'] = True
            elif pred == "equipped_for_imaging": self.rover_capabilities[args[0]]['imaging'] = True
            elif pred == "visible_from": self.obj_view_wps[args[0]].append(args[1])

        # Process At facts for fast lookup
        for r in self.rovers:
            self.at_fact_ids[r] = {wp: self.fact_ids['at'].get((r, wp)) for wp in self.waypoints if (r, wp) in self.fact_ids['at']}

        # 2. Build Multi-Rover Navigation Graphs and BFS
        self.dist_map = defaultdict(lambda: defaultdict(lambda: 999))
        for r in self.rovers:
            self.dist_map[r] = {wp: {wp2: 999 for wp2 in self.waypoints} for wp in self.waypoints}
            for start_wp in self.waypoints:
                self.dist_map[r][start_wp][start_wp] = 0
                queue = deque([(start_wp, 0)])
                visited = {start_wp}
                while queue:
                    curr, d = queue.popleft()
                    for u, v in temp_can_traverse[r]:
                        if u == curr and v not in visited and v in temp_visible[curr]:
                            visited.add(v)
                            self.dist_map[r][start_wp][v] = d + 1
                            queue.append((v, d + 1))

        # 3. Precompute Min Comms
        self.min_dist_comm = defaultdict(lambda: defaultdict(lambda: 999))
        comm_wps_per_wp = defaultdict(list)
        for l_name, l_wp in self.landers:
            for w1 in self.waypoints:
                if l_wp in temp_visible[w1]:
                    comm_wps_per_wp[l_wp].append(w1)

        for r in self.rovers:
            for w_source in self.waypoints:
                best = 999
                for l_name, l_wp in self.landers:
                    for w_comm in comm_wps_per_wp[l_wp]:
                        d = self.dist_map[r][w_source][w_comm]
                        if d < best: best = d
                self.min_dist_comm[r][w_source] = best

    def _get_rover_pos(self, state, rover):
        if rover not in self.at_fact_ids: return None
        for wp, fid in self.at_fact_ids[rover].items():
            if fid is not None and (state >> fid) & 1: return wp
        return None

    def _is_empty(self, state, rover):
        for s_name, r_name in self.fact_ids['store_of'].items():
            if r_name == rover:
                fid = self.fact_ids['empty'].get((s_name,))
                if fid is not None and (state >> fid) & 1: return True
        return False

    def _estimate_task(self, name, args, state):
        # Normalized names for matching
        name = name.lower()
        # Cost estimates based on remaining physical work
        if "get_soil_data" in name or "get_rock_data" in name:
            wp = args[0]
            best = 999
            for r in self.rovers:
                if ("soil" in name and not self.rover_capabilities[r].get('soil')) or \
                   ("rock" in name and not self.rover_capabilities[r].get('rock')): continue
                r_pos = self._get_rover_pos(state, r)
                if not r_pos: continue
                cost = self.dist_map[r][r_pos][wp] + 1 + self.min_dist_comm[r][wp] + 1
                if not self._is_empty(state, r): cost += 1
                if cost < best: best = cost
            return best if best < 999 else 5
        
        if "get_image_data" in name:
            obj, mode = args[0], args[1]
            best = 999
            for r in self.rovers:
                if not self.rover_capabilities[r].get('imaging'): continue
                r_pos = self._get_rover_pos(state, r)
                if not r_pos: continue
                # Simple image cost: dist to view + 1 (calibrate/take) + dist to comm + 1
                wp_views = self.obj_view_wps.get(obj, [])
                for wp in wp_views:
                    cost = self.dist_map[r][r_pos][wp] + 2 + self.min_dist_comm[r][wp] + 1
                    if cost < best: best = cost
            return best if best < 999 else 7
            
        if "do_navigate" in name:
            # args[0] is rover, args[-1] is to_wp
            r, to_wp = args[0], args[-1]
            r_pos = self._get_rover_pos(state, r)
            return self.dist_map[r][r_pos][to_wp] if r_pos else 1
            
        if "sample_" in name or "take_image" in name: return 2
        if "communicate_" in name or "send_" in name: return 1
        return 0

    def _compute(self, state, task_network):
        if not task_network: return 0
        
        h = 0
        for task in task_network:
            # Standard Pytrich grounded task name: pred[arg1,arg2] or pred(arg1,arg2)
            name = task.name
            args = []
            if "[" in name:
                parts = name.split("[")
                pred = parts[0]
                args = parts[1].replace("]", "").split(",")
            else:
                pred = name
            
            h += self._estimate_task(pred, args, state)
        
        # Symmetry breaking: small penalty for number of tasks remaining
        h += 0.01 * len(task_network)
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
