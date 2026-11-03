from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.

    1. Bottleneck Analysis:
       The primary bottleneck in this domain is rover navigation. Rovers must move between sample collection sites, 
       calibration targets, and communication points visible from landers. Because the task network strictly 
       orders these activities, navigation cost dominates the plan length.

    2. Admissible Lower Bounds:
       - Term 1 (Task Completion Cost): Each remaining high-level task in the network (soil, rock, image) 
         requires at least one collection action and one communication action (+2). 
       - Term 2 (Navigation Cost): For any task, a rover must be at specific waypoints. The distance (hops)
         between the rover's current position and these target sites is a strict lower bound on the number 
         of 'navigate' actions required.
       - Term 3 (Preparation Cost): Tasks like 'get_image_data' require an additional 'calibrate' action (+1).
         If the store is full, an 'empty_store' action (+1) is needed before sampling.

    3. Design Strategy:
       - Precompute all-pairs shortest paths for each rover (hops) in initialize().
       - Map objectives, cameras, and transmit sites to waypoints based on grounded operators.
       - In __call__, iterate through the current task network and sum the estimated minimum cost to 
         complete each remaining task branch.
    """

    def __init__(self, rover_gemini3flash_heuristic_6="rover_gemini3flash_heuristic_6"):
        super().__init__(rover_gemini3flash_heuristic_6)
        self.at_fact_ids = {} # rover -> waypoint -> global_id
        self.full_fact_ids = {} # store -> global_id
        self.rover_to_store = {} # rover -> store
        self.rover_capabilities = {'soil': set(), 'rock': set(), 'image': set()}
        self.rover_distances = {} # rover -> wp1 -> wp2 -> dist
        self.transmit_points = {} # rover -> set(waypoints)
        self.visibility_map = {} # objective -> set(waypoints)
        self.calibration_map = {} # camera -> set(waypoints)
        self.camera_to_rover = {} # camera -> rover
        self.camera_to_modes = {} # camera -> set(modes)
        self.rovers = []
        self.waypoints = []

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
        # 1. Identify all objects
        rover_objs = set()
        wp_objs = set()
        camera_objs = set()
        objective_objs = set()
        store_objs = set()

        # 2. Parse grounded facts to find positions and capabilities
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                rover_objs.add(args[0])
                wp_objs.add(args[1])
                if args[0] not in self.at_fact_ids: self.at_fact_ids[args[0]] = {}
                self.at_fact_ids[args[0]][args[1]] = f.global_id
            elif pred == "full":
                store_objs.add(args[0])
                self.full_fact_ids[args[0]] = f.global_id
            elif pred == "store_of":
                self.rover_to_store[args[1]] = args[0]

        self.rovers = list(rover_objs)
        self.waypoints = list(wp_objs)

        # 3. Parse operators to find static connectivity and device locations
        adj = {r: {wp: set() for wp in self.waypoints} for r in self.rovers}
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred == "navigate":
                adj[args[0]][args[1]].add(args[2])
            elif pred == "sample_soil":
                self.rover_capabilities['soil'].add(args[0])
            elif pred == "sample_rock":
                self.rover_capabilities['rock'].add(args[0])
            elif pred == "take_image":
                self.rover_capabilities['image'].add(args[0])
                obj, cam, mode = args[2], args[3], args[4]
                if obj not in self.visibility_map: self.visibility_map[obj] = set()
                self.visibility_map[obj].add(args[1])
                self.camera_to_rover[cam] = args[0]
                if cam not in self.camera_to_modes: self.camera_to_modes[cam] = set()
                self.camera_to_modes[cam].add(mode)
            elif pred == "calibrate":
                cam, obj, wp = args[1], args[2], args[3]
                if cam not in self.calibration_map: self.calibration_map[cam] = set()
                self.calibration_map[cam].add(wp)
            elif pred.startswith("communicate_"):
                r, wp = args[0], args[3] if "1" in pred else args[2]
                if r not in self.transmit_points: self.transmit_points[r] = set()
                self.transmit_points[r].add(wp)

        # 4. Precompute Hops via BFS
        for r in self.rovers:
            self.rover_distances[r] = {}
            for start_wp in self.waypoints:
                dists = {start_wp: 0}
                queue = collections.deque([start_wp])
                while queue:
                    u = queue.popleft()
                    for v in adj[r][u]:
                        if v not in dists:
                            dists[v] = dists[u] + 1
                            queue.append(v)
                self.rover_distances[r][start_wp] = dists

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_dist(self, r, wp1, wp2):
        if r not in self.rover_distances: return 999
        return self.rover_distances[r].get(wp1, {}).get(wp2, 999)

    def _get_rover_positions(self, state):
        positions = {}
        for r in self.rovers:
            for wp in self.waypoints:
                if (state >> self.at_fact_ids[r][wp]) & 1:
                    positions[r] = wp
                    break
        return positions

    def _get_task_cost(self, task_name, state, rover_pos):
        name_clean = task_name.replace("+", "").replace("-", "")
        pred = name_clean.split("[")[0]
        args = []
        if "[" in name_clean:
            args = name_clean.split("[")[1][:-1].split(",")

        # High Level Goals
        if pred == "get_soil_data":
            target_wp = args[0]
            costs = []
            for r in self.rover_capabilities['soil']:
                pos = rover_pos.get(r)
                if not pos: continue
                d1 = self._get_dist(r, pos, target_wp)
                d2 = min([self._get_dist(r, target_wp, twp) for twp in self.transmit_points.get(r, [])], default=999)
                costs.append(d1 + 1 + d2 + 1)
            return min(costs, default=50)

        if pred == "get_rock_data":
            target_wp = args[0]
            costs = []
            for r in self.rover_capabilities['rock']:
                pos = rover_pos.get(r)
                if not pos: continue
                d1 = self._get_dist(r, pos, target_wp)
                d2 = min([self._get_dist(r, target_wp, twp) for twp in self.transmit_points.get(r, [])], default=999)
                costs.append(d1 + 1 + d2 + 1)
            return min(costs, default=50)

        if pred == "get_image_data":
            obj, mode = args[0], args[1]
            costs = []
            for r in self.rover_capabilities['image']:
                pos = rover_pos.get(r)
                if not pos: continue
                for cam, rv in self.camera_to_rover.items():
                    if rv == r and mode in self.camera_to_modes.get(cam, set()):
                        d_cal = min([self._get_dist(r, pos, cwp) for cwp in self.calibration_map.get(cam, [])], default=999)
                        d_img = min([self._get_dist(r, pos, iwp) for iwp in self.visibility_map.get(obj, [])], default=999)
                        d_trans = min([self._get_dist(r, pos, twp) for twp in self.transmit_points.get(r, [])], default=999)
                        costs.append(d_cal + 1 + d_img + 1 + d_trans + 1)
            return min(costs, default=80)

        # Intermediate Tasks
        if pred == "send_soil_data" or pred == "send_rock_data":
            r, target_wp = args[0], args[1]
            pos = rover_pos.get(r)
            if not pos: return 10
            d = min([self._get_dist(r, pos, twp) for twp in self.transmit_points.get(r, [])], default=999)
            return d + 1

        if pred == "do_navigate1":
            r, to = args[0], args[1]
            pos = rover_pos.get(r)
            if not pos: return 5
            return self._get_dist(r, pos, to)

        if pred in ["visit", "unvisit", "navigate", "sample_soil", "sample_rock", "take_image", "drop", "calibrate", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data"]:
            return 1
        
        return 0

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        rover_pos = self._get_rover_positions(state)
        total_h = 0
        for task in task_network:
            # The task in Pytrich can be an Operator or AbstractTask
            total_h += self._get_task_cost(task.name, state, rover_pos)
            
        return total_h
