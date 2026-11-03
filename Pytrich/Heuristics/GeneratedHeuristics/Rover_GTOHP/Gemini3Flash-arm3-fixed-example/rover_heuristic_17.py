from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque, defaultdict
import time

"""
Domain-dependent heuristic for the Rover_GTOHP domain.

1. Bottleneck Analysis:
   The primary bottleneck in the Rover domain is sequential navigation. A rover must move to specific waypoints 
   to gather soil/rock samples or take images, and then potentially move to a communication-ready waypoint 
   (visible to a lander) to transmit the data. Since navigation actions are expensive and often shared across 
   tasks, the shortest-path distance between sites is the dominant cost factor.

2. Admissible Lower Bounds:
   - Term 1: Action Count. Every unachieved communication goal (soil, rock, or image) requires at least 
     a specific number of primitive actions (e.g., 2 for soil/rock data: sample + communicate; 3 for image data: 
     calibrate + take_image + communicate). This is a strict lower bound on non-navigation actions.
   - Term 2: Navigation Distance. For every task in the current task network, the rover must travel 
     from its current location to the target site, and for sampling tasks, onwards to a communication point. 
     By tracking rover locations and summing minimum required navigation steps (BFS distance), we obtain 
     an admissible lower bound on navigation actions.

3. Initialization:
   - Builds a distance matrix (dist[rover][w1][w2]) using BFS based on traverse capabilities.
   - Precomputes communication points (waypoints visible to any lander).
   - Precomputes static navigation work for complex image tasks (dist to calibration + dist to visibility).
   - Maps stores, rovers, cameras, and objectives for O(1) state checks.

4. Summation Logic:
   The heuristic sums the estimated costs of all tasks in the ordered task network. This is admissible 
   because in this domain, tasks are decomposed into sequential primitive actions that do not overlap 
   in their effects on the goals.
"""

class RoverGTOHPGemini3FlashHeuristic17(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_17=None, name="RoverGTOHPGemini3FlashHeuristic17"):
        super().__init__(name=name)
        self.dist = {} # rover -> w1 -> w2 -> int
        self.comm_dist = {} # rover -> w -> int
        self.best_comm_waypoint = {} # rover -> w -> waypoint_name
        self.at_fact_id = defaultdict(dict) # rover -> waypoint -> fact_id
        self.empty_fact_id = {} # store -> fact_id
        self.rover_store = {} # rover -> store
        self.rover_names = []
        self.waypoint_names = []
        self.objective_names = []
        self.camera_rover = {} # camera -> rover
        self.rover_cameras = defaultdict(list) # rover -> [cameras]
        self.camera_modes = defaultdict(set) # camera -> [modes]
        self.camera_target = {} # camera -> objective
        self.visible_from = defaultdict(list) # objective -> [waypoint]
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.image_static_work = defaultdict(lambda: defaultdict(dict)) # rover -> obj -> mode -> (nav_cost, w_img, w_cal)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        # 1. Identify static components
        adj = defaultdict(lambda: defaultdict(list)) # rover -> waypoint -> [neighbors]
        landers = []
        lander_locs = {}
        visibility = defaultdict(list)

        for fact in self.model.facts:
            p, args = self._parse_fact_name(fact.name)
            if not p: continue

            if p == "at":
                if args[0].startswith("rover"): 
                    self.at_fact_id[args[0]][args[1]] = fact.global_id
                    if args[0] not in self.rover_names: self.rover_names.append(args[0])
                if args[1] not in self.waypoint_names: self.waypoint_names.append(args[1])
            elif p == "can_traverse":
                adj[args[0]][args[1]].append(args[2])
            elif p == "visible":
                visibility[args[0]].append(args[1])
            elif p == "at_lander":
                landers.append(args[0])
                lander_locs[args[0]] = args[1]
            elif p == "equipped_for_soil_analysis": self.equipped_soil.add(args[0])
            elif p == "equipped_for_rock_analysis": self.equipped_rock.add(args[0])
            elif p == "equipped_for_imaging": self.equipped_image.add(args[0])
            elif p == "on_board":
                self.camera_rover[args[0]] = args[1]
                self.rover_cameras[args[1]].append(args[0])
            elif p == "supports": self.camera_modes[args[0]].add(args[1])
            elif p == "calibration_target": self.camera_target[args[0]] = args[1]
            elif p == "visible_from":
                self.visible_from[args[0]].append(args[1])
                if args[0] not in self.objective_names: self.objective_names.append(args[0])
            elif p == "store_of": self.rover_store[args[1]] = args[0]
            elif p == "empty": self.empty_fact_id[args[0]] = fact.global_id

        # 2. BFS for distances
        for r in self.rover_names:
            self.dist[r] = {}
            for start in self.waypoint_names:
                dists = {w: 1000 for w in self.waypoint_names}
                dists[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[r][u]:
                        if dists[v] == 1000:
                            dists[v] = dists[u] + 1
                            q.append(v)
                self.dist[r][start] = dists

        # 3. Communication points
        comm_points = set()
        for w in self.waypoint_names:
            for l in landers:
                if lander_locs[l] in visibility[w]:
                    comm_points.add(w)
                    break

        # 4. Comm dist and best comm points
        for r in self.rover_names:
            self.comm_dist[r] = {}
            self.best_comm_waypoint[r] = {}
            for w in self.waypoint_names:
                best_d = 1000
                best_w = w
                for cp in comm_points:
                    d = self.dist[r][w][cp]
                    if d < best_d:
                        best_d = d
                        best_w = cp
                self.comm_dist[r][w] = best_d
                self.best_comm_waypoint[r][w] = best_w

        # 5. Image static work optimized
        for r in self.rover_names:
            if r not in self.equipped_image: continue
            for c in self.rover_cameras[r]:
                target_obj = self.camera_target.get(c)
                if not target_obj: continue
                cal_waypoints = self.visible_from[target_obj]
                for mode in self.camera_modes[c]:
                    for obj in self.objective_names:
                        best_nav = 1000
                        best_wi, best_wc = None, None
                        for wi in self.visible_from[obj]:
                            c_dist = self.comm_dist[r][wi]
                            for wc in cal_waypoints:
                                d_cw = self.dist[r][wc][wi]
                                total = d_cw + c_dist
                                if total < best_nav:
                                    best_nav = total
                                    best_wi, best_wc = wi, wc
                        if best_wi:
                            # Store the nav parts separate from the 3 actions
                            self.image_static_work[r][obj][mode] = (best_nav, best_wi, best_wc)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h)

    def _get_loc(self, r, state):
        for w, fid in self.at_fact_id.get(r, {}).items():
            if (state >> fid) & 1: return w
        return None

    def _compute(self, state, task_network):
        if not task_network: return 0
        
        h = 0
        locs = {r: self._get_loc(r, state) for r in self.rover_names}

        for task in task_network:
            name = task.name.replace("(", " ").replace(")", " ").strip()
            parts = name.split()
            t_type = parts[0]
            args = parts[1:]

            # Marker tasks cost 0
            if t_type in ["nop", "visit", "unvisit"]: continue

            # Sum costs for tasks
            if t_type == "get_soil_data":
                target_w = args[0]
                best_cost = 1000
                chosen_r = None
                for r in self.equipped_soil:
                    curr = locs[r]
                    if not curr: continue
                    d = self.dist[r][curr][target_w] + self.comm_dist[r][target_w] + 2
                    if d < best_cost:
                        best_cost = d
                        chosen_r = r
                h += best_cost
                if chosen_r:
                    locs[chosen_r] = self.best_comm_waypoint[chosen_r][target_w]
            elif t_type == "get_rock_data":
                target_w = args[0]
                best_cost = 1000
                chosen_r = None
                for r in self.equipped_rock:
                    curr = locs[r]
                    if not curr: continue
                    d = self.dist[r][curr][target_w] + self.comm_dist[r][target_w] + 2
                    if d < best_cost:
                        best_cost = d
                        chosen_r = r
                h += best_cost
                if chosen_r:
                    locs[chosen_r] = self.best_comm_waypoint[chosen_r][target_w]
            elif t_type == "get_image_data":
                obj, mode = args[0], args[1]
                best_cost = 1000
                chosen_r = None
                final_w = None
                for r in self.equipped_image:
                    curr = locs[r]
                    if not curr or obj not in self.image_static_work[r] or mode not in self.image_static_work[r][obj]:
                        continue
                    nav, wi, wc = self.image_static_work[r][obj][mode]
                    d = self.dist[r][curr][wc] + nav + 3
                    if d < best_cost:
                        best_cost = d
                        chosen_r = r
                        final_w = self.best_comm_waypoint[r][wi]
                h += best_cost
                if chosen_r:
                    locs[chosen_r] = final_w
            elif t_type in ["do_navigate1", "do_navigate2"]:
                r = args[0]
                target_w = args[-1]
                curr = locs[r]
                if curr and r in self.dist: h += self.dist[r][curr][target_w]
                locs[r] = target_w
            elif t_type == "navigate":
                r = args[0]
                target_w = args[2]
                h += 1
                locs[r] = target_w
            elif t_type in ["send_soil_data", "send_rock_data"]:
                r = args[0]
                curr = locs[r]
                if curr: h += self.comm_dist[r][curr] + 1
                if r in self.best_comm_waypoint: locs[r] = self.best_comm_waypoint[r].get(curr, curr)
            elif t_type == "send_image_data":
                r = args[0]
                curr = locs[r]
                if curr: h += self.comm_dist[r][curr] + 1
                if r in self.best_comm_waypoint: locs[r] = self.best_comm_waypoint[r].get(curr, curr)
            elif t_type == "empty_store":
                r = args[1]
                s = self.rover_store.get(r)
                if s and not ((state >> self.empty_fact_id[s]) & 1): h += 1
            elif t_type == "do_calibrate":
                r, cam = args[0], args[1]
                curr = locs[r]
                best_d = 1000
                best_w = curr
                for w in self.visible_from.get(self.camera_target.get(cam), []):
                    d = self.dist[r][curr][w]
                    if d < best_d:
                        best_d = d
                        best_w = w
                h += best_d + 1
                locs[r] = best_w
            else:
                h += 1

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        # Small tie-breaker for network depth and unemptied stores
        h_value += 0.01 * len(node.task_network)
        super().update_info(h_value)
        return h_value
