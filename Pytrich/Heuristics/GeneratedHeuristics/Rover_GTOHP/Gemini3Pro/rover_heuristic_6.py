import sys
from collections import defaultdict, deque
import math

# Ideally we would import Heuristic from the framework, but we assume it's available
# or we define a dummy one if running standalone. For the planner, it will be injected.
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic6(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_5=None):
        super().__init__(model)
        self.solver_name = "RoverGemini3ProHeuristic6"
        
        # --- Precomputation Structures ---
        self.waypoints = set()
        self.adjacency = defaultdict(list)
        self.distances = {}  # (w1, w2) -> int
        
        self.rover_configs = defaultdict(lambda: {
            'soil': False, 'rock': False, 'imaging': False, 
            'store': None, 'cameras': []
        })
        self.camera_configs = defaultdict(lambda: {'target': None, 'modes': set()})
        self.objective_visibility = defaultdict(list) # obj -> [waypoints]
        self.lander_locations = set()
        self.comm_waypoints = set() # Waypoints from which a lander is visible
        self.waypoint_soil = set()
        self.waypoint_rock = set()
        
        # Maps for bitwise ID lookups
        self.fact_id_map = {}
        
        # --- Static Analysis ---
        self._analyze_static_facts()
        self._compute_all_pairs_shortest_paths()
        self._precompute_comm_distances()

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
        return name, []

    def _analyze_static_facts(self):
        for fact in self.model.facts:
            self.fact_id_map[fact.name] = fact.global_id
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == 'can_traverse':
                if len(args) == 3:
                    u, v = args[1], args[2]
                    self.waypoints.add(u)
                    self.waypoints.add(v)
                    self.adjacency[u].append(v)
            
            elif pred == 'equipped_for_soil_analysis':
                self.rover_configs[args[0]]['soil'] = True
            elif pred == 'equipped_for_rock_analysis':
                self.rover_configs[args[0]]['rock'] = True
            elif pred == 'equipped_for_imaging':
                self.rover_configs[args[0]]['imaging'] = True
            elif pred == 'store_of':
                self.rover_configs[args[1]]['store'] = args[0]
            elif pred == 'on_board':
                self.rover_configs[args[1]]['cameras'].append(args[0])
            
            elif pred == 'calibration_target':
                self.camera_configs[args[0]]['target'] = args[1]
            elif pred == 'supports':
                self.camera_configs[args[0]]['modes'].add(args[1])
            
            elif pred == 'visible_from':
                self.objective_visibility[args[0]].append(args[1])
            
            elif pred == 'at_lander':
                self.lander_locations.add(args[1])
            
            elif pred == 'at_soil_sample':
                self.waypoint_soil.add(args[0])
            elif pred == 'at_rock_sample':
                self.waypoint_rock.add(args[0])

        # Process visibility for communication points
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'visible':
                w1, w2 = args[0], args[1]
                if w2 in self.lander_locations:
                    self.comm_waypoints.add(w1)
                if w1 in self.lander_locations:
                    self.comm_waypoints.add(w2)
        
        for l_loc in self.lander_locations:
            self.comm_waypoints.add(l_loc)

    def _compute_all_pairs_shortest_paths(self):
        all_nodes = list(self.waypoints)
        for start_node in all_nodes:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0
            
            while q:
                curr, d = q.popleft()
                self.distances[(start_node, curr)] = d
                
                for neighbor in self.adjacency[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

    def _get_dist(self, w1, w2):
        return self.distances.get((w1, w2), float('inf'))

    def _precompute_comm_distances(self):
        self.nearest_comm_dist = {}
        for w in self.waypoints:
            min_dist = float('inf')
            for cp in self.comm_waypoints:
                d = self._get_dist(w, cp)
                if d < min_dist:
                    min_dist = d
            self.nearest_comm_dist[w] = min_dist

    def _get_fact_bit(self, predicate, args, state):
        name = f"+{predicate}[" + ",".join(args) + "]"
        if name in self.fact_id_map:
            gid = self.fact_id_map[name]
            return (state >> gid) & 1
        return 0

    def _get_rover_loc(self, state, rover):
        for w in self.waypoints:
            if self._get_fact_bit('at', [rover, w], state):
                return w
        return None

    def compute(self, state, task_network):
        h_val = 0
        
        for task in task_network:
            t_name = task.name
            pred, args = self._parse_fact_name(t_name)
            
            if pred == 'get_soil_data':
                h_val += self._h_get_soil_data(state, args[0])
            elif pred == 'get_rock_data':
                h_val += self._h_get_rock_data(state, args[0])
            elif pred == 'get_image_data':
                h_val += self._h_get_image_data(state, args[0], args[1])
            elif pred == 'do_navigate1':
                # args: rover, to
                r_loc = self._get_rover_loc(state, args[0])
                if r_loc:
                    d = self._get_dist(r_loc, args[1])
                    h_val += d if d != float('inf') else 10
            elif pred == 'do_navigate2':
                # args: rover, from, to. But rover might be at 'from' or mid path.
                # Heuristic assumes rover is at args[1] or we check state
                r_loc = self._get_rover_loc(state, args[0])
                target = args[2]
                if r_loc:
                    d = self._get_dist(r_loc, target)
                    h_val += d if d != float('inf') else 10
            elif pred == 'navigate':
                h_val += 1
            elif pred in ['sample_soil', 'sample_rock', 'take_image', 'calibrate', 
                          'communicate_soil_data1', 'communicate_soil_data2', 
                          'communicate_rock_data1', 'communicate_rock_data2',
                          'communicate_image_data']:
                h_val += 1
                
        return h_val

    def _h_get_soil_data(self, state, target_wp):
        min_cost = float('inf')
        for r, config in self.rover_configs.items():
            if not config['soil']: continue
            r_loc = self._get_rover_loc(state, r)
            if not r_loc: continue
            
            dist_to_sample = self._get_dist(r_loc, target_wp)
            if dist_to_sample == float('inf'): continue
            
            # nav + sample
            cost = dist_to_sample + 1
            
            store = config['store']
            if store and self._get_fact_bit('full', [store], state):
                cost += 1 # drop
            
            dist_to_comm = self.nearest_comm_dist.get(target_wp, float('inf'))
            if dist_to_comm == float('inf'): continue
            
            # nav to comm + communicate
            cost += dist_to_comm + 1
            
            if cost < min_cost:
                min_cost = cost
        return min_cost if min_cost != float('inf') else 20

    def _h_get_rock_data(self, state, target_wp):
        min_cost = float('inf')
        for r, config in self.rover_configs.items():
            if not config['rock']: continue
            r_loc = self._get_rover_loc(state, r)
            if not r_loc: continue
            
            dist_to_sample = self._get_dist(r_loc, target_wp)
            if dist_to_sample == float('inf'): continue
            
            cost = dist_to_sample + 1
            
            store = config['store']
            if store and self._get_fact_bit('full', [store], state):
                cost += 1
            
            dist_to_comm = self.nearest_comm_dist.get(target_wp, float('inf'))
            if dist_to_comm == float('inf'): continue
            
            cost += dist_to_comm + 1
            
            if cost < min_cost:
                min_cost = cost
        return min_cost if min_cost != float('inf') else 20

    def _h_get_image_data(self, state, objective, mode):
        min_cost = float('inf')
        vis_points = self.objective_visibility.get(objective, [])
        if not vis_points: return 20
        
        for r, config in self.rover_configs.items():
            if not config['imaging']: continue
            r_loc = self._get_rover_loc(state, r)
            if not r_loc: continue
            
            for cam in config['cameras']:
                cam_conf = self.camera_configs[cam]
                if mode in cam_conf['modes']:
                    is_calibrated = self._get_fact_bit('calibrated', [cam, r], state)
                    
                    this_cam_cost = float('inf')
                    
                    if is_calibrated:
                        # Path: r -> img -> comm
                        for ip in vis_points:
                            d1 = self._get_dist(r_loc, ip)
                            d2 = self.nearest_comm_dist.get(ip, float('inf'))
                            if d1 != float('inf') and d2 != float('inf'):
                                # nav + take + nav + comm
                                total = d1 + 1 + d2 + 1
                                if total < this_cam_cost:
                                    this_cam_cost = total
                    else:
                        # Path: r -> calib -> img -> comm
                        target = cam_conf['target']
                        target_vis = self.objective_visibility.get(target, [])
                        
                        for cp in target_vis:
                            d1 = self._get_dist(r_loc, cp)
                            if d1 == float('inf'): continue
                            
                            for ip in vis_points:
                                d2 = self._get_dist(cp, ip)
                                d3 = self.nearest_comm_dist.get(ip, float('inf'))
                                if d2 != float('inf') and d3 != float('inf'):
                                    # nav(cal) + cal + nav(img) + img + nav(comm) + comm
                                    total = d1 + 1 + d2 + 1 + d3 + 1
                                    if total < this_cam_cost:
                                        this_cam_cost = total
                    
                    if this_cam_cost < min_cost:
                        min_cost = this_cam_cost
                        
        return min_cost if min_cost != float('inf') else 20
