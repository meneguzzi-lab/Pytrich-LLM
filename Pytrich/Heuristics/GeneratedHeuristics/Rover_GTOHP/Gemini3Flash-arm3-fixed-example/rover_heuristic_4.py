"""
Domain-dependent heuristic for the Rover GTOHP domain.

Bottleneck Analysis:
The primary bottleneck in the Rover domain is the physical navigation of rovers across a constrained waypoint graph 
to perform sequential tasks. Each rover is limited by its specialized equipment (soil/rock tools, cameras) and 
a store that can only hold one physical sample at a time. This requires repeated trips between sampling 
sites and communication waypoints visible to landers.

Admissibility Sketches:
1. Action Count Term: Each uncommunicated data point requires a minimum set of unique actions (Sample/TakeImage, 
   Communicate, and potentially Calibrate). These are mandatory actions that cannot be collapsed, 
   providing a lower bound on cost.
2. Navigation Term: To fulfill a data task, a rover must travel to the data source and then to a lander-visible 
   waypoint. Using precomputed shortest-path distances on the waypoint graph (BFS per rover) gives a tight 
   lower bound on the number of 'navigate' actions required for these transitions.
"""

import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic4(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_4=None, name="RoverGTOHPGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.rovers = []
        self.waypoint_names = []
        self.goal_soil_ids = {}      # wp -> goal_gid
        self.goal_rock_ids = {}      # wp -> goal_gid
        self.goal_image_ids = {}     # (obj, mode) -> goal_gid
        
        self.have_soil_ids = {}      # rover -> {wp: gid}
        self.have_rock_ids = {}      # rover -> {wp: gid}
        self.have_image_ids = {}     # rover -> {(obj, mode): gid}
        
        self.at_rover_ids = {}       # rover -> {wp: gid}
        self.calibrated_ids = {}     # rover -> {camera: gid}
        
        self.dist = {}               # rover -> wp1 -> wp2 -> int
        self.comm_wps = []           # list of wps visible to any lander
        self.obj_visible_wps = {}    # obj -> list of wps where obj is visible
        self.rover_cameras = {}      # rover -> list of cameras
        
        self.inf = 1000000

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, [a.strip() for a in args]
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify objects and static relationships
        waypoint_set = set()
        rover_set = set()
        lander_wp_set = set()
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at":
                rover_set.add(args[0])
                waypoint_set.add(args[1])
            elif pred == "can_traverse":
                rover_set.add(args[0])
                waypoint_set.add(args[1])
                waypoint_set.add(args[2])
            elif pred == "at_lander":
                lander_wp_set.add(args[1])
            elif pred == "visible_from":
                if args[0] not in self.obj_visible_wps: self.obj_visible_wps[args[0]] = []
                self.obj_visible_wps[args[0]].append(args[1])
            elif pred == "on_board":
                if args[1] not in self.rover_cameras: self.rover_cameras[args[1]] = []
                self.rover_cameras[args[1]].append(args[0])

        self.rovers = sorted(list(rover_set))
        self.waypoint_names = sorted(list(waypoint_set))

        # 2. Build waypoint adjacency per rover
        adj = {r: {w: [] for w in self.waypoint_names} for r in self.rovers}
        vis = {w: set() for w in self.waypoint_names}
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "visible":
                vis[args[0]].add(args[1])
                vis[args[1]].add(args[0]) # Assuming visibility is symmetric in this domain
            elif pred == "can_traverse":
                adj[args[0]][args[1]].append(args[2])

        # 3. BFS for shortest navigation paths
        for r in self.rovers:
            self.dist[r] = {w1: {w2: self.inf for w2 in self.waypoint_names} for w1 in self.waypoint_names}
            for start_node in self.waypoint_names:
                self.dist[r][start_node][start_node] = 0
                queue = deque([start_node])
                while queue:
                    curr = queue.popleft()
                    d = self.dist[r][start_node][curr]
                    for neighbor in adj[r].get(curr, []):
                        # Check traverse precondition: must be visible and can_traverse
                        if neighbor in vis[curr] and self.dist[r][start_node][neighbor] == self.inf:
                            self.dist[r][start_node][neighbor] = d + 1
                            queue.append(neighbor)

        # 4. Identify Communication Waypoints (points from which rover can see a lander)
        for w in self.waypoint_names:
            if any(w == lw or w in vis[lw] for lw in lander_wp_set):
                self.comm_wps.append(w)

        # 5. Map Goal Predicates and Dynamic Facts to Global IDs
        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data":
                self.goal_soil_ids[args[0]] = None
            elif pred == "communicated_rock_data":
                self.goal_rock_ids[args[0]] = None
            elif pred == "communicated_image_data":
                self.goal_image_ids[(args[0], args[1])] = None

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            gid = fact.global_id
            
            if pred == "communicated_soil_data" and args[0] in self.goal_soil_ids:
                self.goal_soil_ids[args[0]] = gid
            elif pred == "communicated_rock_data" and args[0] in self.goal_rock_ids:
                self.goal_rock_ids[args[0]] = gid
            elif pred == "communicated_image_data" and (args[0], args[1]) in self.goal_image_ids:
                self.goal_image_ids[(args[0], args[1])] = gid
            elif pred == "have_soil_analysis":
                if args[0] not in self.have_soil_ids: self.have_soil_ids[args[0]] = {}
                self.have_soil_ids[args[0]][args[1]] = gid
            elif pred == "have_rock_analysis":
                if args[0] not in self.have_rock_ids: self.have_rock_ids[args[0]] = {}
                self.have_rock_ids[args[0]][args[1]] = gid
            elif pred == "have_image":
                if args[0] not in self.have_image_ids: self.have_image_ids[args[0]] = {}
                self.have_image_ids[args[0]][(args[1], args[2])] = gid
            elif pred == "at":
                if args[0] not in self.at_rover_ids: self.at_rover_ids[args[0]] = {}
                self.at_rover_ids[args[0]][args[1]] = gid
            elif pred == "calibrated":
                if args[0] not in self.calibrated_ids: self.calibrated_ids[args[0]] = {}
                self.calibrated_ids[args[0]][args[1]] = gid

        h_val = self._compute(initial_node.state)
        return super().initialize(model, h_val)

    def _get_min_dist(self, r, start_wp, target_list):
        min_d = self.inf
        for tw in target_list:
            d = self.dist[r][start_wp].get(tw, self.inf)
            if d < min_d:
                min_d = d
        return min_d

    def _compute(self, state: int) -> int:
        h = 0
        
        # Locate rovers
        rover_pos = {}
        for r in self.rovers:
            for w, gid in self.at_rover_ids.get(r, {}).items():
                if (state >> gid) & 1:
                    rover_pos[r] = w
                    break
        
        # 1. Action Costs (Communicate=1, Sample=1, Calibrate+TakeImage=2)
        # 2. Navigation Costs (Min distance transitions required)
        
        # Soil Goals
        for wp, gid in self.goal_soil_ids.items():
            if not ((state >> gid) & 1):
                h += 1 # communicate action
                any_have = False
                best_nav = self.inf
                for r in self.rovers:
                    if (state >> self.have_soil_ids.get(r, {}).get(wp, 0)) & 1:
                        any_have = True
                        if r in rover_pos:
                            best_nav = min(best_nav, self._get_min_dist(r, rover_pos[r], self.comm_wps))
                        break
                
                if not any_have:
                    h += 1 # sample action
                    for r in self.rovers:
                        if r in rover_pos:
                            d1 = self.dist[r][rover_pos[r]].get(wp, self.inf)
                            d2 = self._get_min_dist(r, wp, self.comm_wps)
                            if d1 + d2 < best_nav: best_nav = d1 + d2
                h += best_nav if best_nav < self.inf else 1

        # Rock Goals
        for wp, gid in self.goal_rock_ids.items():
            if not ((state >> gid) & 1):
                h += 1 # communicate action
                any_have = False
                best_nav = self.inf
                for r in self.rovers:
                    if (state >> self.have_rock_ids.get(r, {}).get(wp, 0)) & 1:
                        any_have = True
                        if r in rover_pos:
                            best_nav = min(best_nav, self._get_min_dist(r, rover_pos[r], self.comm_wps))
                        break
                
                if not any_have:
                    h += 1 # sample action
                    for r in self.rovers:
                        if r in rover_pos:
                            d1 = self.dist[r][rover_pos[r]].get(wp, self.inf)
                            d2 = self._get_min_dist(r, wp, self.comm_wps)
                            if d1 + d2 < best_nav: best_nav = d1 + d2
                h += best_nav if best_nav < self.inf else 1

        # Image Goals
        for (obj, mode), gid in self.goal_image_ids.items():
            if not ((state >> gid) & 1):
                h += 1 # communicate action
                any_have = False
                best_nav = self.inf
                for r in self.rovers:
                    if (state >> self.have_image_ids.get(r, {}).get((obj, mode), 0)) & 1:
                        any_have = True
                        if r in rover_pos:
                            best_nav = min(best_nav, self._get_min_dist(r, rover_pos[r], self.comm_wps))
                        break
                
                if not any_have:
                    h += 2 # Calibrate + TakeImage actions
                    sample_wps = self.obj_visible_wps.get(obj, [])
                    for r in self.rovers:
                        if r not in rover_pos: continue
                        for swp in sample_wps:
                            d1 = self.dist[r][rover_pos[r]].get(swp, self.inf)
                            d2 = self._get_min_dist(r, swp, self.comm_wps)
                            if d1 + d2 < best_nav: best_nav = d1 + d2
                h += best_nav if best_nav < self.inf else 1

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
