import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    BOTTLENECK ANALYSIS:
    The primary bottleneck is Rover navigation. Tasks (soil/rock/image) require the rover to visit 
    specific waypoints to sample or calibrate, and then often visit other waypoints visible to 
    a lander to communicate the data. Since navigation is a serializing resource 
    (one rover can only move along edges), travel distance dominates the plan length.

    ADMISSIBLE LOWER BOUNDS:
    1. h_actions: Each pending goal (communicated_soil/rock/image) requires a specific set of 
       non-navigational actions (sample_soil, sample_rock, calibrate, take_image, communicate). 
       Summing the minimum required actions for each distinct goal provides a safe lower bound.
    2. h_nav: For each rover, the distance to its nearest next required interaction site (sample site, 
       calibration point, or lander-visible point) is an unavoidable cost to satisfy the next step 
       of a goal it is capable of performing. We take the maximum of these over rovers as an 
       admissible travel term (at least one rover must cover its distance).

    ADMISSIBILITY SKETCH:
    H = h_actions + max(h_nav_per_rover). 
    This is admissible because every goal requires at least its primitive actions, and the 
    rovers must travel at least the distance to their first requirement to make any progress.
    """

    def __init__(self, rover_gemini3flash_heuristic_3=None, name="RoverGTOHPHeuristic"):
        super().__init__(name=name)
        self.preprocessing_done = False
        
        # Static knowledge maps
        self.rovers = []
        self.waypoints = []
        self.landers = []
        self.objectives = []
        
        # Capabilities
        self.rover_soil_cap = []
        self.rover_rock_cap = []
        self.rover_image_cap = defaultdict(list) # rover -> list of (camera_id, modes)
        
        # Facts mapping (global_id)
        self.at_bit = {} # (rover, wp) -> bit
        self.have_soil_bit = {} # (rover, wp) -> bit
        self.have_rock_bit = {} # (rover, wp) -> bit
        self.have_image_bit = {} # (rover, obj, mode) -> bit
        self.calibrated_bit = {} # (camera, rover) -> bit
        self.comm_soil_goal_bit = {} # wp -> bit
        self.comm_rock_goal_bit = {} # wp -> bit
        self.comm_image_goal_bit = {} # (obj, mode) -> bit
        self.store_full_bit = {} # rover -> bit

        # Targets
        self.soil_sample_sites = []
        self.rock_sample_sites = []
        self.obj_visible_from = defaultdict(list) # obj -> [wp]
        self.lander_visible_wps = [] # [wp] that can see a lander
        self.cal_target_of = {} # camera -> objective
        
        # Precomputed distances
        self.adj = defaultdict(list)
        self.dist = {} # rover -> w1 -> w2 -> cost

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_val = self._compute(initial_node.state)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state)
        super().update_info(h_val)
        return h_val

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
        # 1. Identify objects and static types
        rover_names = []
        wp_names = []
        camera_names = []
        objective_names = []

        # Pre-scan objects via facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at" and len(args) == 2:
                if args[0] not in rover_names: rover_names.append(args[0])
                if args[1] not in wp_names: wp_names.append(args[1])
            elif pred == "on_board":
                if args[0] not in camera_names: camera_names.append(args[0])
            elif pred == "calibration_target":
                if args[1] not in objective_names: objective_names.append(args[1])
                self.cal_target_of[args[0]] = args[1]
            elif pred == "visible_from":
                self.obj_visible_from[args[0]].append(args[1])
            elif pred == "can_traverse":
                self.adj[args[0]].append((args[1], args[2]))

        self.rovers = rover_names
        self.waypoints = wp_names

        # 2. Build Distance Matrices (BFS per rover)
        for r in self.rovers:
            self.dist[r] = {w1: {w2: 999 for w2 in self.waypoints} for w1 in self.waypoints}
            for w in self.waypoints: self.dist[r][w][w] = 0
            # BFS
            for start_wp in self.waypoints:
                q = deque([(start_wp, 0)])
                visited = {start_wp}
                while q:
                    curr, d = q.popleft()
                    self.dist[r][start_wp][curr] = d
                    for u, v in self.adj[r]:
                        if u == curr and v not in visited:
                            visited.add(v)
                            q.append((v, d + 1))

        # 3. Static Sites and Landers
        lander_wps = []
        visible_pairs = []
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at_lander": lander_wps.append(args[1])
            elif pred == "visible": visible_pairs.append((args[0], args[1]))
            elif pred == "equipped_for_soil_analysis": self.rover_soil_cap.append(args[0])
            elif pred == "equipped_for_rock_analysis": self.rover_rock_cap.append(args[0])
            elif pred == "supports": # camera, mode
                pass # will be handled in Image capability

        for w1, w2 in visible_pairs:
            if w2 in lander_wps: self.lander_visible_wps.append(w1)

        # 4. Map Bits
        camera_to_rover = {}
        camera_supports = defaultdict(list)
        store_of = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at": self.at_bit[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis": self.have_soil_bit[(args[0], args[1])] = f.global_id
            elif pred == "have_rock_analysis": self.have_rock_bit[(args[0], args[1])] = f.global_id
            elif pred == "have_image": self.have_image_bit[(args[0], args[1], args[2])] = f.global_id
            elif pred == "calibrated": self.calibrated_bit[(args[0], args[1])] = f.global_id
            elif pred == "communicated_soil_data": self.comm_soil_goal_bit[args[0]] = f.global_id
            elif pred == "communicated_rock_data": self.comm_rock_goal_bit[args[0]] = f.global_id
            elif pred == "communicated_image_data": self.comm_image_goal_bit[(args[0], args[1])] = f.global_id
            elif pred == "full": self.store_full_bit[args[1]] = f.global_id # store, rover
            elif pred == "on_board": camera_to_rover[args[0]] = args[1]
            elif pred == "supports": camera_supports[args[0]].append(args[1])
            elif pred == "store_of": store_of[args[1]] = args[0]

        for cam, r in camera_to_rover.items():
            self.rover_image_cap[r].append((cam, camera_supports[cam]))

        # Extract Goal structure
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self.pending_goals_data = []
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred in ["communicated_soil_data", "communicated_rock_data"]:
                self.pending_goals_data.append((pred, args[0]))
            elif pred == "communicated_image_data":
                self.pending_goals_data.append((pred, args[0], args[1]))

    def _compute(self, state: int) -> int:
        h_actions = 0
        rover_at = {}
        for r in self.rovers:
            for wp in self.waypoints:
                if (state >> self.at_bit.get((r, wp), 0)) & 1:
                    rover_at[r] = wp
                    break
        
        # Term 1: Action Progress
        rover_min_nav = {r: 999 for r in self.rovers}
        any_pending = False

        for ginfo in self.pending_goals_data:
            g_type = ginfo[0]
            
            if g_type == "communicated_soil_data":
                wp = ginfo[1]
                if (state >> self.comm_soil_goal_bit[wp]) & 1: continue
                any_pending = True
                
                is_had = any((state >> self.have_soil_bit.get((r, wp), 0)) & 1 for r in self.rovers)
                if is_had:
                    h_actions += 1 # communicate
                    for r in self.rover_soil_cap:
                        # dist to nearest lander visible point
                        d = min([self.dist[r][rover_at.get(r, wp)][lw] for lw in self.lander_visible_wps] + [999])
                        rover_min_nav[r] = min(rover_min_nav[r], d)
                else:
                    h_actions += 2 # sample + communicate
                    for r in self.rover_soil_cap:
                        d = self.dist[r][rover_at.get(r, wp)][wp] + min([self.dist[r][wp][lw] for lw in self.lander_visible_wps] + [999])
                        rover_min_nav[r] = min(rover_min_nav[r], d)
            
            elif g_type == "communicated_rock_data":
                wp = ginfo[1]
                if (state >> self.comm_rock_goal_bit[wp]) & 1: continue
                any_pending = True
                
                is_had = any((state >> self.have_rock_bit.get((r, wp), 0)) & 1 for r in self.rovers)
                if is_had:
                    h_actions += 1
                    for r in self.rover_rock_cap:
                        d = min([self.dist[r][rover_at.get(r, wp)][lw] for lw in self.lander_visible_wps] + [999])
                        rover_min_nav[r] = min(rover_min_nav[r], d)
                else:
                    h_actions += 2
                    for r in self.rover_rock_cap:
                        d = self.dist[r][rover_at.get(r, wp)][wp] + min([self.dist[r][wp][lw] for lw in self.lander_visible_wps] + [999])
                        rover_min_nav[r] = min(rover_min_nav[r], d)
            
            elif g_type == "communicated_image_data":
                obj, mode = ginfo[1], ginfo[2]
                if (state >> self.comm_image_goal_bit[(obj, mode)]) & 1: continue
                any_pending = True
                
                is_had = any((state >> self.have_image_bit.get((r, obj, mode), 0)) & 1 for r in self.rovers)
                if is_had:
                    h_actions += 1
                    for r, cams in self.rover_image_cap.items():
                        if any(mode in c[1] for c in cams):
                            d = min([self.dist[r][rover_at.get(r, "")][lw] for lw in self.lander_visible_wps] + [999])
                            rover_min_nav[r] = min(rover_min_nav[r], d)
                else:
                    h_actions += 3 # calib + take + comm
                    for r, cams in self.rover_image_cap.items():
                        for cam, modes in cams:
                            if mode in modes:
                                cal_obj = self.cal_target_of[cam]
                                # Dist: at -> cal_site -> photo_site -> lander_site
                                d_cal = min([self.dist[r][rover_at.get(r, "")][cw] for wp_l in self.obj_visible_from[cal_obj] for cw in [wp_l]] + [999])
                                d_photo = min([self.dist[r]["any"][pw] for pw in self.obj_visible_from[obj]] + [0]) # simplification
                                rover_min_nav[r] = min(rover_min_nav[r], d_cal + d_photo)

        if not any_pending: return 0

        h_nav = max([d for d in rover_min_nav.values() if d < 999] + [0])
        return h_actions + h_nav
