from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RoverGTOHPGemini3FlashHeuristic11(Heuristic):
    """
    Rover Domain Heuristic (GTOHP version).

    1. Bottleneck: Rover navigation between sampling sites and lander communication points.
       A rover can only be in one place, and navigation is the most action-heavy part of the plan.

    2. Admissible Lower Bounds:
       - Term 1 (Navigation): BFS distance (d) multiplied by 3 actions per hop (visit+navigate+unvisit) 
         plus 1 for the terminal task decomposition. Total: 3*d + 1.
       - Term 2 (Task Count): Each task in the network requires at least one action to complete.

    3. Strategy:
       - Precompute all-pairs shortest paths for every rover based on navigate actions (static facts stripped).
       - Identify capability-based waypoints (comm-ready, sampling spots, imaging sites).
       - Compute cost based on current task network contents, using current rover positions from bitwise state.
    """

    def __init__(self, rover_gemini3flash_heuristic_11="rover_gemini3flash_heuristic_11"):
        super().__init__(rover_gemini3flash_heuristic_11)
        self.at_fact_ids = {} # rover_name -> wp_name -> global_id
        self.comm_fact_ids = {"soil": {}, "rock": {}, "image": {}} # type -> wp/obj -> global_id
        self.store_empty_ids = {} # rover_name -> global_id
        
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        
        self.can_traverse = {} # rover -> wp -> set(wp)
        self.comm_wps = {} # rover -> set(wp)
        self.visible_from = {} # obj -> set(wp)
        self.rover_cameras = {} # rover -> list of (cam, target, modes)
        self.rover_equipped = {"soil": set(), "rock": set(), "image": set()}
        
        self.dist = {} # rover -> wp -> wp -> int
        self.dist_to_comm = {} # rover -> wp -> int

    def _parse_item_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx == -1:
            return name, []
        predicate = name[:bracket_idx]
        args = name[bracket_idx+1:-1].split(",")
        return predicate, [a.strip() for a in args]

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Predicates
        for f in model.facts:
            p, args = self._parse_item_name(f.name)
            if p == "at":
                r, wp = args[0], args[1]
                self.rovers.add(r)
                self.waypoints.add(wp)
                if r not in self.at_fact_ids: self.at_fact_ids[r] = {}
                self.at_fact_ids[r][wp] = f.global_id
            elif p == "communicated_soil_data":
                self.comm_fact_ids["soil"][args[0]] = f.global_id
            elif p == "communicated_rock_data":
                self.comm_fact_ids["rock"][args[0]] = f.global_id
            elif p == "communicated_image_data":
                self.comm_fact_ids["image"][(args[0], args[1])] = f.global_id
            elif p == "empty":
                # Use store_of to link to rover later via operators
                pass 

        # 2. Extract Static Info from Operators
        for op in model.operators:
            p, args = self._parse_item_name(op.name)
            if p == "navigate":
                r, w1, w2 = args
                if r not in self.can_traverse: self.can_traverse[r] = {}
                if w1 not in self.can_traverse[r]: self.can_traverse[r][w1] = set()
                self.can_traverse[r][w1].add(w2)
            elif p.startswith("communicate_soil"):
                r = args[0]
                wp = args[3] if "1" in p else args[2]
                if r not in self.comm_wps: self.comm_wps[r] = set()
                self.comm_wps[r].add(wp)
                self.rover_equipped["soil"].add(r)
            elif p.startswith("communicate_rock"):
                r = args[0]
                wp = args[3] if "1" in p else args[2]
                if r not in self.comm_wps: self.comm_wps[r] = set()
                self.comm_wps[r].add(wp)
                self.rover_equipped["rock"].add(r)
            elif p == "communicate_image_data":
                r, wp = args[0], args[4]
                if r not in self.comm_wps: self.comm_wps[r] = set()
                self.comm_wps[r].add(wp)
                self.rover_equipped["image"].add(r)
            elif p == "sample_soil":
                self.rover_equipped["soil"].add(args[0])
            elif p == "sample_rock":
                self.rover_equipped["rock"].add(args[0])
            elif p == "calibrate":
                r, cam, obj, wp = args
                if r not in self.rover_equipped["image"]: self.rover_equipped["image"].add(r)
                if r not in self.rover_cameras: self.rover_cameras[r] = []
                self.rover_cameras[r].append((cam, obj, set()))
                if obj not in self.visible_from: self.visible_from[obj] = set()
                self.visible_from[obj].add(wp)
            elif p == "take_image":
                r, wp, obj = args[0], args[1], args[2]
                if obj not in self.visible_from: self.visible_from[obj] = set()
                self.visible_from[obj].add(wp)
                # Update camera modes
                if r in self.rover_cameras:
                    for cam_data in self.rover_cameras[r]:
                        if cam_data[0] == args[3]: cam_data[2].add(args[4])

        # 3. BFS All-Pairs Shortest Paths
        for r in self.rovers:
            self.dist[r] = {}
            for start in self.waypoints:
                self.dist[r][start] = {w: 1000 for w in self.waypoints}
                self.dist[r][start][start] = 0
                if r not in self.can_traverse or start not in self.can_traverse[r]:
                    continue
                q = deque([(start, 0)])
                visited = {start}
                while q:
                    curr, d = q.popleft()
                    self.dist[r][start][curr] = d
                    for neighbor in self.can_traverse[r].get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))

        # 4. Precompute helper distances
        for r in self.rovers:
            self.dist_to_comm[r] = {}
            for w in self.waypoints:
                cwps = self.comm_wps.get(r, set())
                self.dist_to_comm[r][w] = min([self.dist[r][w][cw] for cw in cwps]) if cwps else 1000

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_rover_pos(self, rover, state):
        for wp, fid in self.at_fact_ids.get(rover, {}).items():
            if (state >> fid) & 1: return wp
        return None

    def _compute(self, state, task_network):
        if not task_network: return 0
        
        h = 0
        rover_pos = {r: self._get_rover_pos(r, state) for r in self.rovers}

        for i, task in enumerate(task_network):
            p, args = self._parse_item_name(task.name)
            task_cost = 1 # Minimum actions
            
            if p == "do_navigate1":
                r, to_wp = args[0], args[1]
                curr = rover_pos.get(r)
                if curr and to_wp and curr != to_wp:
                    task_cost = 3 * self.dist[r][curr][to_wp] + 1
            
            elif p == "get_soil_data":
                wp = args[0]
                if not ((state >> self.comm_fact_ids["soil"].get(wp, 0)) & 1):
                    min_nav = 1000
                    for r in self.rover_equipped["soil"]:
                        curr = rover_pos.get(r)
                        if curr:
                            d = self.dist[r][curr][wp] + self.dist_to_comm[r][wp]
                            min_nav = min(min_nav, 3 * d + 3)
                    task_cost = min_nav if min_nav < 1000 else 10
            
            elif p == "get_rock_data":
                wp = args[0]
                if not ((state >> self.comm_fact_ids["rock"].get(wp, 0)) & 1):
                    min_nav = 1000
                    for r in self.rover_equipped["rock"]:
                        curr = rover_pos.get(r)
                        if curr:
                            d = self.dist[r][curr][wp] + self.dist_to_comm[r][wp]
                            min_nav = min(min_nav, 3 * d + 3)
                    task_cost = min_nav if min_nav < 1000 else 10
            
            elif p == "get_image_data":
                obj, mode = args[0], args[1]
                if not ((state >> self.comm_fact_ids["image"].get((obj, mode), 0)) & 1):
                    min_nav = 1000
                    for r in self.rover_equipped["image"]:
                        # Check if rover has camera for mode
                        has_mode = False
                        for _, _, modes in self.rover_cameras.get(r, []):
                            if mode in modes: has_mode = True; break
                        if not has_mode: continue
                        
                        curr = rover_pos.get(r)
                        if curr:
                            # Cost to calibrate + cost to image + cost to comm
                            # Using nearest visible spots as admissible lower bound
                            v_wps = self.visible_from.get(obj, set())
                            d_img = min([self.dist[r][curr][vw] for vw in v_wps]) if v_wps else 1000
                            d_comm = self.dist_to_comm[r][curr]
                            min_nav = min(min_nav, 3*d_img + 3*d_comm + 6)
                    task_cost = min_nav if min_nav < 1000 else 15

            # Ordered sum: First task in network gets full navigation assessment
            # Subsequent tasks add minimum work overhead
            if i == 0:
                h += task_cost
            else:
                h += 1

        return int(h)
