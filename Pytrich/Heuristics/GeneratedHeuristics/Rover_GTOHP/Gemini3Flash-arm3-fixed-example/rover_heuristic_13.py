import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Rover_GTOHP Domain Bottleneck Analysis:
---------------------------------------
The primary bottleneck in the Rover domain is navigation. Rovers must serialize their movements between 
sampling waypoints (soil/rock), calibration waypoints (cameras), imaging waypoints, and communication 
waypoints (visible to landers). The store resource (one per rover) also forces sequential drop/sample 
actions when multiple samples are needed.

Admissible Lower Bound Sketches:
---------------------------------
1. Non-Nav Actions: A lower bound on necessary primitive actions (sample_soil, sample_rock, calibrate, 
   take_image, communicate, drop). Each unfulfilled goal requires a specific sequence (e.g., sample + communicate).
2. Max Navigation: For any pending goal, a rover must travel from its current position to the 
   sampling/imaging site and then to a communication site. The maximum of these minimum required 
   travel distances across all goals is a valid lower bound on the number of 'navigate' steps.
"""

class RoverGTOHPGemini3FlashHeuristic13(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_13=None, name="RoverGTOHPGemini3FlashHeuristic13"):
        super().__init__(name=name)
        self.fact_map = defaultdict(dict)
        self.rovers = []
        self.waypoints = []
        self.soil_goals = []
        self.rock_goals = []
        self.image_goals = []
        self.dist = {} # rover -> wp1 -> wp2 -> dist
        self.comm_dist = {} # rover -> wp -> dist_to_nearest_comm_site
        self.image_nav_table = {} # rover -> obj -> mode -> {wp: total_dist}
        self.image_nav_calibrated = {} # rover -> obj -> {wp: total_dist}
        self.rover_store = {}
        self.rover_cams = defaultdict(list)
        self.cam_target = {}
        self.cam_modes = defaultdict(set)
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.at_ids = {} # rover -> wp -> global_id
        self.have_soil_ids = {} # rover -> wp -> global_id
        self.have_rock_ids = {} # rover -> wp -> global_id
        self.have_image_ids = {} # rover -> obj -> mode -> global_id
        self.comm_soil_ids = {} # wp -> global_id
        self.comm_rock_ids = {} # wp -> global_id
        self.comm_image_ids = {} # (obj, mode) -> global_id
        self.full_ids = {} # store -> global_id
        self.calibrated_ids = {} # (cam, rover) -> global_id

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
        self.model = model
        
        # 1. Parse all facts to build maps
        obj_types = defaultdict(set)
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[pred][tuple(args)] = fact.global_id

        # 2. Extract objects and types
        landers = set()
        waypoint_set = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'at_lander': landers.add(args[0])
            if pred == 'at': self.rovers.append(args[0])
            if pred == 'visible': 
                waypoint_set.add(args[0])
                waypoint_set.add(args[1])
        
        self.rovers = sorted(list(set(self.rovers)))
        self.waypoints = sorted(list(waypoint_set))
        
        # 3. Parse Static State for caps and connections
        init_state_explicit = model.state_explicit_repr(model.initial_state)
        adj = {r: defaultdict(list) for r in self.rovers}
        visible = defaultdict(set)
        visible_from = defaultdict(list) # obj -> [wp]
        lander_at = {}
        
        for name in init_state_explicit:
            pred, args = self._parse_fact_name(name)
            if pred == 'can_traverse': adj[args[0]][args[1]].append(args[2])
            elif pred == 'visible': visible[args[0]].add(args[1])
            elif pred == 'visible_from': visible_from[args[0]].append(args[1])
            elif pred == 'at_lander': lander_at[args[0]] = args[1]
            elif pred == 'equipped_for_soil_analysis': self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis': self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging': self.equipped_image.add(args[0])
            elif pred == 'store_of': self.rover_store[args[1]] = args[0]
            elif pred == 'on_board': self.rover_cams[args[1]].append(args[0])
            elif pred == 'supports': self.cam_supports[args[0]].add(args[1])
            elif pred == 'calibration_target': self.cam_target[args[0]] = args[1]

        # 4. Precompute distances for each rover
        for r in self.rovers:
            self.dist[r] = {}
            for start in self.waypoints:
                dists = {wp: 1000 for wp in self.waypoints}
                dists[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[r].get(u, []):
                        if dists[v] == 1000:
                            dists[v] = dists[u] + 1
                            q.append(v)
                self.dist[r][start] = dists

        # 5. Precompute communication distances
        lander_wps = list(lander_at.values())
        comm_ready_wps = [w1 for w1 in self.waypoints if any(visible[w1].intersection({w2}) for w2 in lander_wps)]
        for r in self.rovers:
            self.comm_dist[r] = {}
            for w1 in self.waypoints:
                self.comm_dist[r][w1] = min([self.dist[r][w1][w2] for w2 in comm_ready_wps] + [1000])

        # 6. Map Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'communicated_soil_data': self.soil_goals.append(args[0])
            elif pred == 'communicated_rock_data': self.rock_goals.append(args[0])
            elif pred == 'communicated_image_data': self.image_goals.append((args[0], args[1]))

        # 7. Precompute image nav tables
        for r in self.rovers:
            self.image_nav_table[r] = defaultdict(lambda: defaultdict(dict))
            self.image_nav_calibrated[r] = defaultdict(dict)
            if r not in self.equipped_image: continue
            for obj, mode in self.image_goals:
                cams = [c for c in self.rover_cams[r] if mode in self.cam_supports[c]]
                if not cams: continue
                img_wps = visible_from[obj]
                for w_curr in self.waypoints:
                    min_all = 1000
                    min_cal = 1000
                    for cam in cams:
                        cal_target = self.cam_target.get(cam)
                        cal_wps = visible_from.get(cal_target, [])
                        for w_img in img_wps:
                            c_dist = self.comm_dist[r][w_img]
                            min_cal = min(min_cal, self.dist[r][w_curr][w_img] + c_dist)
                            for w_cal in cal_wps:
                                min_all = min(min_all, self.dist[r][w_curr][w_cal] + self.dist[r][w_cal][w_img] + c_dist)
                    self.image_nav_table[r][obj][mode][w_curr] = min_all
                    self.image_nav_calibrated[r][obj][w_curr] = min_cal

        # 8. Store ID shortcuts for fast bitwise checking
        for r in self.rovers:
            self.at_ids[r] = {wp: self.fact_map['at'].get((r, wp)) for wp in self.waypoints}
            for wp in self.waypoints:
                self.have_soil_ids[(r, wp)] = self.fact_map['have_soil_analysis'].get((r, wp))
                self.have_rock_ids[(r, wp)] = self.fact_map['have_rock_analysis'].get((r, wp))
            for obj, mode in self.image_goals:
                self.have_image_ids[(r, obj, mode)] = self.fact_map['have_image'].get((r, obj, mode))
            for cam in self.rover_cams[r]:
                self.calibrated_ids[(cam, r)] = self.fact_map['calibrated'].get((cam, r))

        for wp in self.soil_goals: self.comm_soil_ids[wp] = self.fact_map['communicated_soil_data'].get((wp,))
        for wp in self.rock_goals: self.comm_rock_ids[wp] = self.fact_map['communicated_rock_data'].get((wp,))
        for obj, mode in self.image_goals: self.comm_image_ids[(obj, mode)] = self.fact_map['communicated_image_data'].get((obj, mode))
        for r, s in self.rover_store.items(): self.full_ids[s] = self.fact_map['full'].get((s,))

        h_value = self._calculate(initial_node.state)
        return super().initialize(model, h_value)

    def _calculate(self, state):
        action_h = 0
        max_nav_h = 0
        
        # Find current rover positions
        rover_pos = {}
        for r in self.rovers:
            for wp in self.waypoints:
                fid = self.at_ids[r][wp]
                if fid is not None and (state >> fid) & 1:
                    rover_pos[r] = wp
                    break
        
        # Soil Goals
        for wp in self.soil_goals:
            if (state >> self.comm_soil_ids[wp]) & 1: continue
            action_h += 1
            holding_rover = None
            for r in self.rovers:
                fid = self.have_soil_ids.get((r, wp))
                if fid is not None and (state >> fid) & 1:
                    holding_rover = r; break
            if holding_rover:
                max_nav_h = max(max_nav_h, self.comm_dist[holding_rover][rover_pos.get(holding_rover, self.waypoints[0])])
            else:
                action_h += 1
                capable = [r for r in self.rovers if r in self.equipped_soil]
                best_nav = min([self.dist[r][rover_pos[r]][wp] + self.comm_dist[r][wp] for r in capable if r in rover_pos] + [1000])
                max_nav_h = max(max_nav_h, best_nav)
                if all((state >> self.full_ids[self.rover_store[r]]) & 1 for r in capable if r in self.rover_store):
                    action_h += 1

        # Rock Goals
        for wp in self.rock_goals:
            if (state >> self.comm_rock_ids[wp]) & 1: continue
            action_h += 1
            holding_rover = None
            for r in self.rovers:
                fid = self.have_rock_ids.get((r, wp))
                if fid is not None and (state >> fid) & 1:
                    holding_rover = r; break
            if holding_rover:
                max_nav_h = max(max_nav_h, self.comm_dist[holding_rover][rover_pos.get(holding_rover, self.waypoints[0])])
            else:
                action_h += 1
                capable = [r for r in self.rovers if r in self.equipped_rock]
                best_nav = min([self.dist[r][rover_pos[r]][wp] + self.comm_dist[r][wp] for r in capable if r in rover_pos] + [1000])
                max_nav_h = max(max_nav_h, best_nav)
                if all((state >> self.full_ids[self.rover_store[r]]) & 1 for r in capable if r in self.rover_store):
                    action_h += 1

        # Image Goals
        for obj, mode in self.image_goals:
            if (state >> self.comm_image_ids[(obj, mode)]) & 1: continue
            action_h += 1
            holding_rover = None
            for r in self.rovers:
                fid = self.have_image_ids.get((r, obj, mode))
                if fid is not None and (state >> fid) & 1:
                    holding_rover = r; break
            if holding_rover:
                max_nav_h = max(max_nav_h, self.comm_dist[holding_rover][rover_pos.get(holding_rover, self.waypoints[0])])
            else:
                action_h += 2 # take + comm
                best_nav = 1000
                for r in self.rovers:
                    if r not in self.equipped_image or r not in rover_pos: continue
                    cams = [c for c in self.rover_cams[r] if mode in self.cam_supports[c]]
                    if not cams: continue
                    
                    curr_wp = rover_pos[r]
                    is_calibrated = any((state >> self.calibrated_ids[(c, r)]) & 1 for c in cams)
                    if is_calibrated:
                        best_nav = min(best_nav, self.image_nav_calibrated[r][obj].get(curr_wp, 1000))
                    else:
                        action_h += 1 # need to calibrate
                        best_nav = min(best_nav, self.image_nav_table[r][obj][mode].get(curr_wp, 1000))
                max_nav_h = max(max_nav_h, best_nav)

        return action_h + max_nav_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._calculate(node.state)
        # Small tie-breaker for depth/progress
        h_val += 0.0001 * len(node.task_network)
        super().update_info(h_val)
        return h_val
