import sys
from typing import List, Set, Dict, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RoverGemini3ProHeuristic4(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain.
    
    Design Rationale:
    1. Structure: The domain involves traversing a graph (waypoints) to perform tasks (sampling, imaging) 
       and then communicating results. The costs are dominated by movement distances.
    2. Decomposition Cost: We estimate the cost of high-level tasks (get_data) by summing:
       - Movement to the target/resource.
       - Processing (sample/calibrate/image).
       - Movement to a communication point.
       - Communication action.
    3. Resource Awareness: We consider specific rover capabilities (equipped_for_*) and current state 
       (at, calibrated, store status) to assign the cheapest capable rover to each task.
    4. Precomputation: We precalculate the All-Pairs Shortest Paths (APSP) for the navigation graph 
       and identify valid communication points (locations visible from a lander) to fast-track distance lookups.
    """

    def __init__(self, model, rover_gemini3pro_heuristic_3=None):
        super().__init__(model)
        self.INF = 9999
        
        # Static Data Structures
        self.graph = {} # u -> [v]
        self.dist_matrix = {} # (u, v) -> int
        self.comm_points = set() # Waypoints valid for communication
        self.min_dist_to_comm = {} # waypoint -> int
        
        self.rover_capabilities = {
            'soil': set(),
            'rock': set(),
            'imaging': set()
        }
        # cam -> {'rover': r, 'modes': [], 'target': obj}
        self.camera_info = {}
        # obj -> [waypoints]
        self.objective_visibility = {}
        
        # Fact ID Caches for fast bitwise state checking
        self.f_at_map = {} # rover -> [(waypoint, global_id)]
        self.f_full = {} # store -> global_id
        self.f_empty = {} # store -> global_id
        self.f_have_soil = {} # (rover, wp) -> global_id
        self.f_have_rock = {} # (rover, wp) -> global_id
        self.f_have_image = {} # (rover, obj, mode) -> global_id
        self.f_calibrated = {} # (cam, rover) -> global_id
        self.f_communicated_soil = {} # wp -> global_id
        self.f_communicated_rock = {} # wp -> global_id
        self.f_communicated_image = {} # (obj, mode) -> global_id
        
        self._initialize_static_data()

    def _parse_fact_name(self, name: str):
        """Parses Pytrich grounded fact format: +predicate[arg1,arg2]"""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return predicate, args
        return None, []

    def _initialize_static_data(self):
        waypoints = set()
        landers_at = []
        visible_pairs = set()

        # 1. Parse static facts and build caches
        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            if not pred: continue
            
            gid = fact.global_id

            if pred == 'can_traverse':
                # args: rover, from, to
                # Assumes map is same for all rovers (standard in Rover domain)
                if len(args) >= 3:
                    u, v = args[1], args[2]
                    self._add_edge(u, v)
                    waypoints.add(u)
                    waypoints.add(v)
            
            elif pred == 'equipped_for_soil_analysis':
                self.rover_capabilities['soil'].add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_capabilities['rock'].add(args[0])
            elif pred == 'equipped_for_imaging':
                self.rover_capabilities['imaging'].add(args[0])
            
            elif pred == 'on_board':
                cam, rov = args[0], args[1]
                if cam not in self.camera_info: self.camera_info[cam] = {'rover': None, 'modes': [], 'target': None}
                self.camera_info[cam]['rover'] = rov
            
            elif pred == 'calibration_target':
                cam, obj = args[0], args[1]
                if cam not in self.camera_info: self.camera_info[cam] = {'rover': None, 'modes': [], 'target': None}
                self.camera_info[cam]['target'] = obj
                
            elif pred == 'supports':
                cam, mode = args[0], args[1]
                if cam not in self.camera_info: self.camera_info[cam] = {'rover': None, 'modes': [], 'target': None}
                self.camera_info[cam]['modes'].append(mode)
                
            elif pred == 'visible_from':
                obj, wp = args[0], args[1]
                if obj not in self.objective_visibility: self.objective_visibility[obj] = []
                self.objective_visibility[obj].append(wp)
                
            elif pred == 'at_lander':
                landers_at.append(args[1])
            elif pred == 'visible':
                visible_pairs.add((args[0], args[1]))
            
            # Dynamic ID Caching
            elif pred == 'at':
                r, w = args[0], args[1]
                if r not in self.f_at_map: self.f_at_map[r] = []
                self.f_at_map[r].append((w, gid))
            elif pred == 'full':
                self.f_full[args[0]] = gid
            elif pred == 'empty':
                self.f_empty[args[0]] = gid
            elif pred == 'have_soil_analysis':
                self.f_have_soil[(args[0], args[1])] = gid
            elif pred == 'have_rock_analysis':
                self.f_have_rock[(args[0], args[1])] = gid
            elif pred == 'have_image':
                self.f_have_image[(args[0], args[1], args[2])] = gid
            elif pred == 'calibrated':
                self.f_calibrated[(args[0], args[1])] = gid
            elif pred == 'communicated_soil_data':
                self.f_communicated_soil[args[0]] = gid
            elif pred == 'communicated_rock_data':
                self.f_communicated_rock[args[0]] = gid
            elif pred == 'communicated_image_data':
                self.f_communicated_image[(args[0], args[1])] = gid

        # 2. Compute Distances (APSP)
        self.waypoints = list(waypoints)
        self._compute_apsp()

        # 3. Identify Communication Points
        # Comm allowed at p1 if exists lander at p2 AND visible(p1, p2)
        for p2 in landers_at:
            for p1 in self.waypoints:
                if (p1, p2) in visible_pairs:
                    self.comm_points.add(p1)
        
        # 4. Precompute min distance to communication for every waypoint
        for wp in self.waypoints:
            min_d = self.INF
            for cp in self.comm_points:
                d = self._get_dist(wp, cp)
                if d < min_d:
                    min_d = d
            self.min_dist_to_comm[wp] = min_d

    def _add_edge(self, u, v):
        if u not in self.graph: self.graph[u] = []
        if v not in self.graph: self.graph[v] = []
        if v not in self.graph[u]: self.graph[u].append(v)

    def _compute_apsp(self):
        """Floyd-Warshall for All-Pairs Shortest Paths"""
        n = len(self.waypoints)
        wp_map = {w: i for i, w in enumerate(self.waypoints)}
        dist = [[self.INF] * n for _ in range(n)]
        
        for i in range(n):
            dist[i][i] = 0
            
        for u in self.graph:
            if u not in wp_map: continue
            u_i = wp_map[u]
            for v in self.graph[u]:
                if v not in wp_map: continue
                dist[u_i][wp_map[v]] = 1
        
        # FW Algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Store in dict
        for i, w1 in enumerate(self.waypoints):
            for j, w2 in enumerate(self.waypoints):
                self.dist_matrix[(w1, w2)] = dist[i][j]

    def _get_dist(self, w1, w2):
        return self.dist_matrix.get((w1, w2), self.INF)

    def _is_true(self, state, global_id):
        if global_id is None: return False
        return (state >> global_id) & 1

    def _get_rover_loc(self, state, rover):
        if rover in self.f_at_map:
            for wp, gid in self.f_at_map[rover]:
                if (state >> gid) & 1:
                    return wp
        return None

    def infer(self, node):
        state = node.state
        tn = node.task_network
        cost = 0
        
        for task in tn:
            t_name = task if isinstance(task, str) else task.name
            pred, args = self._parse_fact_name(t_name)
            
            if not pred:
                cost += 1
                continue
                
            # --- NAVIGATION ---
            if 'navigate' in pred:
                # Args: rover, [from], to
                rover = args[0]
                to_wp = args[-1]
                r_loc = self._get_rover_loc(state, rover)
                if r_loc:
                    cost += self._get_dist(r_loc, to_wp)
                else:
                    cost += 5 # Penalty if location unknown
            
            # --- SOIL / ROCK DATA ---
            elif pred == 'get_soil_data' or pred == 'get_rock_data':
                wp = args[0]
                type_ = 'soil' if 'soil' in pred else 'rock'
                
                # Check if already done
                comm_dict = self.f_communicated_soil if type_ == 'soil' else self.f_communicated_rock
                if self._is_true(state, comm_dict.get(wp)):
                    continue
                
                # Estimate cost: Find cheapest capable rover
                min_r_cost = self.INF
                capable_rovers = self.rover_capabilities[type_]
                
                for r in capable_rovers:
                    r_loc = self._get_rover_loc(state, r)
                    if not r_loc: continue
                    
                    # Analysis check
                    have_dict = self.f_have_soil if type_ == 'soil' else self.f_have_rock
                    has_analysis = self._is_true(state, have_dict.get((r, wp)))
                    
                    if has_analysis:
                        # Just need to communicate
                        dist_comm = self.min_dist_to_comm.get(r_loc, self.INF)
                        c = dist_comm + 1
                    else:
                        # Go to sample -> sample -> go to comm -> comm
                        dist_sample = self._get_dist(r_loc, wp)
                        dist_comm_from_sample = self.min_dist_to_comm.get(wp, self.INF)
                        # +1 for sample action, +1 for comm action, +1 for potential drop if full
                        c = dist_sample + 1 + dist_comm_from_sample + 1 + 1
                    
                    if c < min_r_cost:
                        min_r_cost = c
                
                cost += min_r_cost if min_r_cost < self.INF else 50

            # --- IMAGE DATA ---
            elif pred == 'get_image_data':
                obj, mode = args[0], args[1]
                
                if self._is_true(state, self.f_communicated_image.get((obj, mode))):
                    continue
                
                min_img_cost = self.INF
                
                for cam, info in self.camera_info.items():
                    if mode in info['modes']:
                        r = info['rover']
                        if not r: continue
                        r_loc = self._get_rover_loc(state, r)
                        if not r_loc: continue
                        
                        # Check if already have image
                        if self._is_true(state, self.f_have_image.get((r, obj, mode))):
                            d = self.min_dist_to_comm.get(r_loc, self.INF)
                            if (d + 1) < min_img_cost: min_img_cost = d + 1
                            continue
                        
                        # Need to acquire
                        # Cost: (Calibrate) + Goto Visible + Take + Goto Comm + Comm
                        
                        # Calibration
                        calib_cost = 0
                        if not self._is_true(state, self.f_calibrated.get((cam, r))):
                            calib_cost = 5 # heuristic penalty for calibration detour
                        
                        # Visibility
                        vis_wps = self.objective_visibility.get(obj, [])
                        best_vis_cost = self.INF
                        
                        for vwp in vis_wps:
                            d_to_vis = self._get_dist(r_loc, vwp)
                            d_to_comm = self.min_dist_to_comm.get(vwp, self.INF)
                            total = d_to_vis + 1 + d_to_comm + 1
                            if total < best_vis_cost:
                                best_vis_cost = total
                        
                        c = calib_cost + best_vis_cost
                        if c < min_img_cost:
                            min_img_cost = c
                            
                cost += min_img_cost if min_img_cost < self.INF else 50

            # --- SEND ACTIONS ---
            elif pred.startswith('send_'):
                # args: rover, ...
                r = args[0]
                r_loc = self._get_rover_loc(state, r)
                if r_loc:
                    dist = self.min_dist_to_comm.get(r_loc, self.INF)
                    cost += dist + 1 if dist < self.INF else 10
                else:
                    cost += 10
            
            else:
                # Default small cost for atomic actions or other tasks
                cost += 1
                
        return cost
