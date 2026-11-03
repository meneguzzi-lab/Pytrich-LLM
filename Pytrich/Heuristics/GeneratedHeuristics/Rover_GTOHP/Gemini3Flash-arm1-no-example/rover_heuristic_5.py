from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

# BOTTLENECK ANALYSIS:
# The primary bottleneck in the Rover domain is navigation. To fulfill goals, rovers must travel to 
# specific waypoints to collect samples or take images, and then move to waypoints with line-of-sight 
# to a lander to communicate that data. Storage management (dropping samples) and camera 
# calibration add secondary sequential constraints.

# ADMISSIBILITY SKETCH:
# 1. Navigation cost (_nav_cost): For a distance d (hops), the HTN methods m1, m3, m4 require exactly 
#    3d-2 primitive actions (navigate, visit, unvisit). Using hops as 'd' provides a lower bound.
# 2. Task completion: Each high-level task like get_soil_data requires a sequence of navigation, 
#    sampling, and communication. The minimum number of primitive actions required to complete 
#    a task from its current state (e.g., 1 for comm, 1 for sample, plus min-navigation) is used.
# 3. Summing task costs: Since the task network in this domain is ordered and tasks typically 
#    refer to distinct data gathering requirements, summing the lower bounds for each task 
#    remains an admissible estimate of total remaining plan length.

class RoverGTOHPGemini3FlashHeuristic5(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_5="rover_gemini3flash_heuristic_5"):
        super().__init__(name=rover_gemini3flash_heuristic_5)
        self.rover_ids = []
        self.wp_ids = []
        self.store_ids = []
        self.camera_ids = []
        self.obj_ids = []
        self.distances = {}  # dist[rover][wp1][wp2]
        self.comm_waypoints = set()
        self.obj_src_waypoints = {} # obj -> [wp1, wp2, ...]
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.rover_store = {} # rover -> store
        self.camera_rover = {} # camera -> rover
        self.camera_targets = {} # camera -> objective
        self.camera_modes = {} # camera -> set(modes)
        self.at_id = {} # rover -> wp -> gid
        self.full_id = {} # store -> gid
        self.have_soil_id = {} # rover -> wp -> gid
        self.have_rock_id = {} # rover -> wp -> gid
        self.have_image_id = {} # rover -> obj -> mode -> gid
        self.calibrated_id = {} # camera -> rover -> gid

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self.rover_ids = [f.name for f in model.facts if "rover" in f.name and "store" not in f.name]
        self.wp_ids = [f.name for f in model.facts if "waypoint" in f.name]
        
        # Process Static Facts and IDs
        lander_wps = []
        adj = {}
        visible_pairs = set()
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "at": self.at_id.setdefault(args[0], {})[args[1]] = f.global_id
            elif pred == "full": self.full_id[args[0]] = f.global_id
            elif pred == "have_soil_analysis": self.have_soil_id.setdefault(args[0], {})[args[1]] = f.global_id
            elif pred == "have_rock_analysis": self.have_rock_id.setdefault(args[0], {})[args[1]] = f.global_id
            elif pred == "have_image": self.have_image_id.setdefault(args[0], {}).setdefault(args[1], {})[args[2]] = f.global_id
            elif pred == "calibrated": self.calibrated_id.setdefault(args[0], {})[args[1]] = f.global_id
            elif pred == "at_lander": lander_wps.append(args[1])
            elif pred == "visible": visible_pairs.add((args[0], args[1])); visible_pairs.add((args[1], args[0]))
            elif pred == "visible_from": self.obj_src_waypoints.setdefault(args[0], []).append(args[1])
            elif pred == "equipped_for_soil_analysis": self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis": self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging": self.equipped_imaging.add(args[0])
            elif pred == "store_of": self.rover_store[args[1]] = args[0]
            elif pred == "on_board": self.camera_rover[args[0]] = args[1]
            elif pred == "calibration_target": self.camera_targets[args[0]] = args[1]
            elif pred == "supports": self.camera_modes.setdefault(args[0], set()).add(args[1])
            elif pred == "can_traverse":
                r, w1, w2 = args
                adj.setdefault(r, {}).setdefault(w1, []).append(w2)

        # Identify Communication Waypoints
        for wp in visible_pairs:
            if wp[1] in lander_wps: self.comm_waypoints.add(wp[0])
            if wp[0] in lander_wps: self.comm_waypoints.add(wp[1])

        # Precompute Multi-Rover Distance BFS
        rovers = set(adj.keys())
        waypoints = set()
        for r in adj: waypoints.update(adj[r].keys())
        
        for r in rovers:
            self.distances[r] = {}
            for start in waypoints:
                self.distances[r][start] = {start: 0}
                queue = deque([start])
                while queue:
                    curr = queue.popleft()
                    for neighbor in adj[r].get(curr, []):
                        if neighbor not in self.distances[r][start]:
                            self.distances[r][start][neighbor] = self.distances[r][start][curr] + 1
                            queue.append(neighbor)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        if not task_network: return 0
        h = 0
        r_pos = self._get_rover_positions(state)
        
        for task in task_network:
            t_name, args = self._parse_task(task.name)
            h += self._estimate_task_cost(t_name, args, state, r_pos)
        
        # Admissibility check: Goals remaining (simple count)
        goal_count = 0
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            if not self._is_true_by_name(state, g_name):
                goal_count += 1
        
        return max(h, goal_count)

    def _estimate_task_cost(self, t_type, args, state, r_pos):
        if t_type == "get_soil_data":
            wp = args[0]
            return self._min_rover_data_cost(wp, "soil", state, r_pos)
        elif t_type == "get_rock_data":
            wp = args[0]
            return self._min_rover_data_cost(wp, "rock", state, r_pos)
        elif t_type == "get_image_data":
            obj, mode = args[0], args[1]
            return self._min_rover_image_cost(obj, mode, state, r_pos)
        elif t_type == "do_navigate1":
            r, to_wp = args[0], args[1]
            return self._nav_cost(self._dist(r, r_pos.get(r), to_wp))
        elif t_type == "send_soil_data":
            return 1 + self._min_nav_to_comm(args[0], r_pos.get(args[0]))
        elif t_type == "send_rock_data":
            return 1 + self._min_nav_to_comm(args[0], r_pos.get(args[0]))
        elif t_type == "send_image_data":
            return 1 + self._min_nav_to_comm(args[0], r_pos.get(args[0]))
        elif t_type in ["navigate", "sample_soil", "sample_rock", "take_image", "calibrate", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data", "drop", "visit", "unvisit"]:
            return 1
        return 0

    def _min_rover_data_cost(self, target_wp, d_type, state, r_pos):
        costs = []
        capable = self.equipped_soil if d_type == "soil" else self.equipped_rock
        id_map = self.have_soil_id if d_type == "soil" else self.have_rock_id
        for r in capable:
            r_wp = r_pos.get(r)
            if self._is_true(state, id_map.get(r, {}).get(target_wp)):
                costs.append(1 + self._min_nav_to_comm(r, r_wp))
            else:
                cost = 2 + self._nav_cost(self._dist(r, r_wp, target_wp)) + self._min_nav_to_comm(r, target_wp)
                if self._is_true(state, self.full_id.get(self.rover_store.get(r))):
                    cost += 1
                costs.append(cost)
        return min(costs) if costs else 5

    def _min_rover_image_cost(self, obj, mode, state, r_pos):
        costs = []
        for r in self.equipped_imaging:
            r_wp = r_pos.get(r)
            # Check if any camera on board supports mode
            cams = [c for c, rv in self.camera_rover.items() if rv == r and mode in self.camera_modes.get(c, set())]
            if not cams: continue
            
            if any(self._is_true(state, self.have_image_id.get(r, {}).get(obj, {}).get(mode)) for r in self.rover_ids):
                costs.append(1 + self._min_nav_to_comm(r, r_wp))
            else:
                # Needs take_image: requires calibration + src_wp
                for c in cams:
                    for src_wp in self.obj_src_waypoints.get(obj, []):
                        cal_target = self.camera_targets.get(c)
                        cal_wps = self.obj_src_waypoints.get(cal_target, [])
                        min_cal_nav = min([self._dist(r, r_wp, cw) for cw in cal_wps] + [10])
                        is_cal = self._is_true(state, self.calibrated_id.get(c, {}).get(r))
                        cal_cost = 0 if is_cal else (self._nav_cost(min_cal_nav) + 1)
                        
                        take_nav = self._nav_cost(self._dist(r, r_wp if is_cal else (cal_wps[0] if cal_wps else r_wp), src_wp))
                        costs.append(cal_cost + take_nav + 1 + self._min_nav_to_comm(r, src_wp) + 1)
        return min(costs) if costs else 5

    def _min_nav_to_comm(self, r, start_wp):
        dists = [self._nav_cost(self._dist(r, start_wp, cwp)) for cwp in self.comm_waypoints]
        return min(dists) if dists else 10

    def _dist(self, r, w1, w2):
        if w1 == w2: return 0
        return self.distances.get(r, {}).get(w1, {}).get(w2, 10)

    def _nav_cost(self, d):
        return 3 * d - 2 if d > 0 else 0

    def _get_rover_positions(self, state):
        r_pos = {}
        for r, wps in self.at_id.items():
            for wp, gid in wps.items():
                if (state >> gid) & 1:
                    r_pos[r] = wp
                    break
        return r_pos

    def _is_true(self, state, gid):
        return gid is not None and (state >> gid) & 1

    def _is_true_by_name(self, state, name):
        # For goal facts which might not be in our precomputed ID maps
        # Not used in __call__ for speed, only for initialization or global goal check
        for f in self.model.facts:
            if f.name == name:
                return (state >> f.global_id) & 1
        return False

    def _parse_fact_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return None, []

    def _parse_task(self, name):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []
