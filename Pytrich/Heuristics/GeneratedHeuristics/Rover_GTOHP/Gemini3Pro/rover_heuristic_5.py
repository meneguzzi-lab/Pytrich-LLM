import sys
import re
from typing import List, Dict, Set, Tuple, Optional
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic5(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_4=None):
        super().__init__(model)
        self.model = model
        
        # --- Precomputation Structures ---
        # Rovers and their capabilities
        self.rovers: Set[str] = set()
        self.equipped_soil: Set[str] = set()
        self.equipped_rock: Set[str] = set()
        self.equipped_imaging: Set[str] = set()
        
        # Camera info: rover -> [camera_names]
        self.rover_cameras: Dict[str, List[str]] = {}
        # Camera props: camera -> {'modes': set(), 'target': objective}
        self.camera_props: Dict[str, Dict] = {}
        
        # Navigation: rover -> {u -> {v: weight}}
        self.rover_graphs: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.all_waypoints: Set[str] = set()
        
        # Visibility: loc -> set(locs)
        self.visibility: Dict[str, Set[str]] = {}
        # Objective visibility: objective -> set(waypoints)
        self.objective_visible_from: Dict[str, Set[str]] = {}
        
        # Lander info
        self.lander_locs: Set[str] = set()
        
        # Store mapping: rover -> store
        self.rover_store: Dict[str, str] = {}
        
        # Dynamic Fact IDs mappings (name -> global_id)
        # We store these to quickly check state bits
        self.fact_ids: Dict[str, int] = {}
        self.dynamic_predicates_idx: Dict[str, Dict[str, int]] = {
            'at': {},                 # rover -> {wp -> id}
            'have_soil_analysis': {}, # rover -> {wp -> id}
            'have_rock_analysis': {}, # rover -> {wp -> id}
            'have_image': {},         # rover -> {obj -> {mode -> id}}
            'calibrated': {},         # camera -> {rover -> id}
            'full': {},               # store -> id
            'empty': {},              # store -> id
            'communicated_soil': {},  # wp -> id
            'communicated_rock': {},  # wp -> id
            'communicated_image': {}  # obj -> {mode -> id}
        }

        # Goal Identification
        # We will parse the goal bitmask once to identify high-level goals
        self.parsed_goals: List[Dict] = []

        self._parse_model()
        self._precompute_distances()
        self._identify_comm_spots()
        self._parse_goals()

    def _parse_fact_name(self, name: str) -> Tuple[str, List[str]]:
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _parse_model(self):
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            self.fact_ids[fact.name] = fact.global_id
            
            # Static Parsing
            if pred == 'can_traverse':
                # can_traverse[rover, from, to]
                if len(args) == 3:
                    r, u, v = args
                    self.rovers.add(r)
                    self.all_waypoints.add(u)
                    self.all_waypoints.add(v)
                    if r not in self.rover_graphs:
                        self.rover_graphs[r] = {}
                    if u not in self.rover_graphs[r]:
                        self.rover_graphs[r][u] = {}
                    self.rover_graphs[r][u][v] = 1 # Unit cost

            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(args[0])
                self.rovers.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(args[0])
                self.rovers.add(args[0])
            
            elif pred == 'on_board':
                # on_board[camera, rover]
                c, r = args
                if r not in self.rover_cameras:
                    self.rover_cameras[r] = []
                self.rover_cameras[r].append(c)
                if c not in self.camera_props:
                    self.camera_props[c] = {'modes': set(), 'target': None}
            
            elif pred == 'calibration_target':
                # calibration_target[camera, objective]
                c, o = args
                if c not in self.camera_props:
                    self.camera_props[c] = {'modes': set(), 'target': None}
                self.camera_props[c]['target'] = o
                
            elif pred == 'supports':
                # supports[camera, mode]
                c, m = args
                if c not in self.camera_props:
                    self.camera_props[c] = {'modes': set(), 'target': None}
                self.camera_props[c]['modes'].add(m)
            
            elif pred == 'visible':
                # visible[p1, p2]
                u, v = args
                if u not in self.visibility: self.visibility[u] = set()
                self.visibility[u].add(v)
            
            elif pred == 'visible_from':
                # visible_from[objective, waypoint]
                o, w = args
                if o not in self.objective_visible_from: self.objective_visible_from[o] = set()
                self.objective_visible_from[o].add(w)
            
            elif pred == 'at_lander':
                # at_lander[lander, waypoint]
                self.lander_locs.add(args[1])
            
            elif pred == 'store_of':
                # store_of[store, rover]
                s, r = args
                self.rover_store[r] = s

            # Dynamic IDs Mapping
            elif pred == 'at':
                r, w = args
                if r not in self.dynamic_predicates_idx['at']:
                    self.dynamic_predicates_idx['at'][r] = {}
                self.dynamic_predicates_idx['at'][r][w] = fact.global_id
            
            elif pred == 'have_soil_analysis':
                r, w = args
                if r not in self.dynamic_predicates_idx['have_soil_analysis']:
                    self.dynamic_predicates_idx['have_soil_analysis'][r] = {}
                self.dynamic_predicates_idx['have_soil_analysis'][r][w] = fact.global_id

            elif pred == 'have_rock_analysis':
                r, w = args
                if r not in self.dynamic_predicates_idx['have_rock_analysis']:
                    self.dynamic_predicates_idx['have_rock_analysis'][r] = {}
                self.dynamic_predicates_idx['have_rock_analysis'][r][w] = fact.global_id
            
            elif pred == 'have_image':
                # have_image[rover, objective, mode]
                r, o, m = args
                if r not in self.dynamic_predicates_idx['have_image']:
                    self.dynamic_predicates_idx['have_image'][r] = {}
                if o not in self.dynamic_predicates_idx['have_image'][r]:
                    self.dynamic_predicates_idx['have_image'][r][o] = {}
                self.dynamic_predicates_idx['have_image'][r][o][m] = fact.global_id

            elif pred == 'calibrated':
                c, r = args
                if c not in self.dynamic_predicates_idx['calibrated']:
                    self.dynamic_predicates_idx['calibrated'][c] = {}
                self.dynamic_predicates_idx['calibrated'][c][r] = fact.global_id

            elif pred == 'full':
                self.dynamic_predicates_idx['full'][args[0]] = fact.global_id
            
            elif pred == 'empty':
                self.dynamic_predicates_idx['empty'][args[0]] = fact.global_id

            elif pred == 'communicated_soil_data':
                self.dynamic_predicates_idx['communicated_soil'][args[0]] = fact.global_id
            
            elif pred == 'communicated_rock_data':
                self.dynamic_predicates_idx['communicated_rock'][args[0]] = fact.global_id
                
            elif pred == 'communicated_image_data':
                o, m = args
                if o not in self.dynamic_predicates_idx['communicated_image']:
                    self.dynamic_predicates_idx['communicated_image'][o] = {}
                self.dynamic_predicates_idx['communicated_image'][o][m] = fact.global_id

    def _precompute_distances(self):
        # Compute All-Pairs Shortest Paths for each rover using Floyd-Warshall
        # self.distances[rover][u][v] = int
        self.distances = {}
        
        for rover in self.rovers:
            if rover not in self.rover_graphs:
                self.distances[rover] = {}
                continue
                
            graph = self.rover_graphs[rover]
            dist = {}
            waypoints = list(self.all_waypoints)
            
            # Init
            for u in waypoints:
                dist[u] = {}
                for v in waypoints:
                    if u == v:
                        dist[u][v] = 0
                    elif u in graph and v in graph[u]:
                        dist[u][v] = graph[u][v]
                    else:
                        dist[u][v] = float('inf')
            
            # Floyd-Warshall
            for k in waypoints:
                for i in waypoints:
                    if dist[i][k] == float('inf'): continue
                    for j in waypoints:
                        if dist[k][j] == float('inf'): continue
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
            
            self.distances[rover] = dist

    def _identify_comm_spots(self):
        # A waypoint is a comm spot if it is visible to a waypoint with a lander
        self.comm_spots = set()
        for l_loc in self.lander_locs:
            # Add l_loc itself if visible to itself (usually yes)
            # Check visibility map
            for u, visible_neighbors in self.visibility.items():
                if l_loc in visible_neighbors:
                    self.comm_spots.add(u)
            # Often visibility is symmetric, but we check what's defined

    def _parse_goals(self):
        # The goal is a bitwise int. We need to find which bits are set.
        # However, checking every bit is slow. 
        # We iterate over the known goal predicate IDs we found during parsing.
        
        # Soil Goals
        for wp, gid in self.dynamic_predicates_idx['communicated_soil'].items():
            if (self.model.goals >> gid) & 1:
                self.parsed_goals.append({'type': 'soil', 'wp': wp, 'id': gid})
        
        # Rock Goals
        for wp, gid in self.dynamic_predicates_idx['communicated_rock'].items():
            if (self.model.goals >> gid) & 1:
                self.parsed_goals.append({'type': 'rock', 'wp': wp, 'id': gid})
                
        # Image Goals
        for obj, modes in self.dynamic_predicates_idx['communicated_image'].items():
            for mode, gid in modes.items():
                if (self.model.goals >> gid) & 1:
                    self.parsed_goals.append({'type': 'image', 'obj': obj, 'mode': mode, 'id': gid})

    def get_rover_location(self, rover, state_bitwise):
        # Scan possible locations for this rover
        # This is linear in number of waypoints, but unavoidable unless we cache state
        if rover in self.dynamic_predicates_idx['at']:
            for wp, gid in self.dynamic_predicates_idx['at'][rover].items():
                if (state_bitwise >> gid) & 1:
                    return wp
        return None

    def get_dist(self, rover, u, v):
        if u == v: return 0
        if rover in self.distances and u in self.distances[rover] and v in self.distances[rover][u]:
            d = self.distances[rover][u][v]
            return d if d != float('inf') else 9999
        return 9999

    def get_min_dist_to_set(self, rover, u, targets):
        min_d = 9999
        for t in targets:
            d = self.get_dist(rover, u, t)
            if d < min_d:
                min_d = d
        return min_d

    def compute(self, state_bitwise: int, task_network: List = None) -> int:
        total_cost = 0
        
        # Cache rover locations for this state
        rover_locs = {}
        for r in self.rovers:
            loc = self.get_rover_location(r, state_bitwise)
            if loc:
                rover_locs[r] = loc

        for goal in self.parsed_goals:
            # Check if already achieved
            if (state_bitwise >> goal['id']) & 1:
                continue

            min_goal_cost = 9999

            if goal['type'] == 'soil':
                wp = goal['wp']
                for r in self.equipped_soil:
                    if r not in rover_locs: continue
                    r_loc = rover_locs[r]
                    
                    # Check if already carrying
                    carrying = False
                    if r in self.dynamic_predicates_idx['have_soil_analysis']:
                        gid = self.dynamic_predicates_idx['have_soil_analysis'][r].get(wp)
                        if gid and ((state_bitwise >> gid) & 1):
                            carrying = True
                    
                    cost = 0
                    if carrying:
                        # Navigate to comm spot + communicate
                        dist_comm = self.get_min_dist_to_set(r, r_loc, self.comm_spots)
                        cost = dist_comm + 1 # +1 for communicate action
                    else:
                        # Navigate to sample + sample + navigate to comm + communicate
                        # + potentially empty store
                        dist_sample = self.get_dist(r, r_loc, wp)
                        dist_comm = self.get_min_dist_to_set(r, wp, self.comm_spots)
                        
                        # Check store
                        store_cost = 0
                        s = self.rover_store.get(r)
                        if s and s in self.dynamic_predicates_idx['full']:
                            fid = self.dynamic_predicates_idx['full'][s]
                            if (state_bitwise >> fid) & 1:
                                store_cost = 1 # drop
                        
                        cost = dist_sample + store_cost + 1 + dist_comm + 1 # +1 sample, +1 comm
                    
                    if cost < min_goal_cost:
                        min_goal_cost = cost

            elif goal['type'] == 'rock':
                wp = goal['wp']
                for r in self.equipped_rock:
                    if r not in rover_locs: continue
                    r_loc = rover_locs[r]
                    
                    carrying = False
                    if r in self.dynamic_predicates_idx['have_rock_analysis']:
                        gid = self.dynamic_predicates_idx['have_rock_analysis'][r].get(wp)
                        if gid and ((state_bitwise >> gid) & 1):
                            carrying = True
                    
                    cost = 0
                    if carrying:
                        dist_comm = self.get_min_dist_to_set(r, r_loc, self.comm_spots)
                        cost = dist_comm + 1
                    else:
                        dist_sample = self.get_dist(r, r_loc, wp)
                        dist_comm = self.get_min_dist_to_set(r, wp, self.comm_spots)
                        
                        store_cost = 0
                        s = self.rover_store.get(r)
                        if s and s in self.dynamic_predicates_idx['full']:
                            fid = self.dynamic_predicates_idx['full'][s]
                            if (state_bitwise >> fid) & 1:
                                store_cost = 1
                        
                        cost = dist_sample + store_cost + 1 + dist_comm + 1
                    
                    if cost < min_goal_cost:
                        min_goal_cost = cost

            elif goal['type'] == 'image':
                obj = goal['obj']
                mode = goal['mode']
                
                # Find capable rovers
                capable_rovers = []
                for r in self.equipped_imaging:
                    if r not in rover_locs: continue
                    # Check cameras on board
                    if r in self.rover_cameras:
                        for cam in self.rover_cameras[r]:
                            if cam in self.camera_props and mode in self.camera_props[cam]['modes']:
                                capable_rovers.append((r, cam))
                                
                for r, cam in capable_rovers:
                    r_loc = rover_locs[r]
                    
                    carrying = False
                    if r in self.dynamic_predicates_idx['have_image'] and obj in self.dynamic_predicates_idx['have_image'][r]:
                        gid = self.dynamic_predicates_idx['have_image'][r][obj].get(mode)
                        if gid and ((state_bitwise >> gid) & 1):
                            carrying = True
                    
                    cost = 0
                    if carrying:
                        dist_comm = self.get_min_dist_to_set(r, r_loc, self.comm_spots)
                        cost = dist_comm + 1
                    else:
                        # Need to take image
                        # 1. Check calibration
                        calibrated = False
                        if cam in self.dynamic_predicates_idx['calibrated'] and r in self.dynamic_predicates_idx['calibrated'][cam]:
                            cid = self.dynamic_predicates_idx['calibrated'][cam][r]
                            if (state_bitwise >> cid) & 1:
                                calibrated = True
                        
                        vis_spots = self.objective_visible_from.get(obj, set())
                        
                        if calibrated:
                            # Goto vis_spot -> take -> goto comm -> comm
                            dist_to_vis = self.get_min_dist_to_set(r, r_loc, vis_spots)
                            # We approximate distance from vis_spot to comm spot
                            # Since vis_spots is a set, to be admissible we take min(dist(v, c) for v in vis, c in comm)
                            # But that decouples the path. For FF/h_add, we can be greedy.
                            # Let's assume we go to the *closest* vis spot, then to *closest* comm spot from there.
                            
                            # Finding specific closest vis spot
                            best_vis_spot = None
                            min_d_v = 9999
                            for v in vis_spots:
                                d = self.get_dist(r, r_loc, v)
                                if d < min_d_v:
                                    min_d_v = d
                                    best_vis_spot = v
                            
                            dist_img_to_comm = 9999
                            if best_vis_spot:
                                dist_img_to_comm = self.get_min_dist_to_set(r, best_vis_spot, self.comm_spots)
                            
                            cost = min_d_v + 1 + dist_img_to_comm + 1
                        else:
                            # Calibrate -> Take -> Comm
                            cal_target = self.camera_props[cam]['target']
                            cal_vis_spots = self.objective_visible_from.get(cal_target, set())
                            
                            # 1. Goto cal spot
                            best_cal_spot = None
                            min_d_cal = 9999
                            for c_spot in cal_vis_spots:
                                d = self.get_dist(r, r_loc, c_spot)
                                if d < min_d_cal:
                                    min_d_cal = d
                                    best_cal_spot = c_spot
                            
                            # 2. Goto img spot from cal spot
                            dist_cal_to_img = 9999
                            best_img_spot = None
                            if best_cal_spot:
                                for i_spot in vis_spots:
                                    d = self.get_dist(r, best_cal_spot, i_spot)
                                    if d < dist_cal_to_img:
                                        dist_cal_to_img = d
                                        best_img_spot = i_spot
                            
                            # 3. Goto comm
                            dist_img_to_comm = 9999
                            if best_img_spot:
                                dist_img_to_comm = self.get_min_dist_to_set(r, best_img_spot, self.comm_spots)
                                
                            cost = min_d_cal + 1 + dist_cal_to_img + 1 + dist_img_to_comm + 1
                            
                    if cost < min_goal_cost:
                        min_goal_cost = cost

            if min_goal_cost < 9999:
                total_cost += min_goal_cost
            else:
                # If a goal is unreachable (e.g. no rover equipped), return infinity or a large penalty
                # For satisficing, a large number guides away from dead ends
                return 999999

        return total_cost
