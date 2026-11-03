import sys
import math
from collections import defaultdict, deque

from Pytrich.Heuristics.heuristic import Heuristic
class RoverGemini3ProHeuristic9(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_8=None):
        super().__init__(model)
        self.rover_gemini3pro_heuristic_8 = rover_gemini3pro_heuristic_8
        self.fact_name_to_id = {}
        
        # Domain graphs and maps
        self.rover_graph = defaultdict(lambda: defaultdict(list)) # rover -> u -> v list
        self.dist = defaultdict(lambda: defaultdict(lambda: float('inf'))) # rover -> u -> v -> dist
        
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_cameras = defaultdict(list) # rover -> list of camera names
        self.camera_modes = defaultdict(set) # camera -> set of modes
        self.camera_target = {} # camera -> objective
        self.objective_visible_from = defaultdict(set) # objective -> set of waypoints
        self.lander_visible_points = set() # waypoints from which communication is possible
        self.rovers = set()
        self.stores = {} # rover -> store name
        self.rover_at_facts = defaultdict(list) # rover -> list of (wp, fact_id)
        
        self._analyze_domain()

    def _parse_fact_name(self, name: str):
        name = name.strip().lstrip('+').lstrip('-')
        if '[' in name and name.endswith(']'):
            pred = name[:name.find('[')]
            args = name[name.find('[')+1:-1].split(',')
            return pred, args
        return name, []

    def _analyze_domain(self):
        # 1. Map IDs and build basic indices
        visible_pairs = defaultdict(set)
        lander_locs = set()

        for fact in self.model.facts:
            self.fact_name_to_id[fact.name] = fact.global_id
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == 'can_traverse' and len(args) == 3:
                r, u, v = args
                self.rover_graph[r][u].append(v)
                self.rovers.add(r)
            elif pred == 'equipped_for_soil_analysis':
                self.rover_equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.rover_equipped_rock.add(args[0])
            elif pred == 'on_board':
                cam, rov = args
                self.rover_cameras[rov].append(cam)
            elif pred == 'supports':
                cam, mode = args
                self.camera_modes[cam].add(mode)
            elif pred == 'calibration_target':
                cam, obj = args
                self.camera_target[cam] = obj
            elif pred == 'visible_from':
                obj, wp = args
                self.objective_visible_from[obj].add(wp)
            elif pred == 'at_lander':
                lander_locs.add(args[1])
            elif pred == 'visible':
                visible_pairs[args[0]].add(args[1])
                visible_pairs[args[1]].add(args[0])
            elif pred == 'store_of':
                store, rov = args
                self.stores[rov] = store
            elif pred == 'at':
                # Cache potential location facts for fast lookup
                rov, wp = args
                self.rover_at_facts[rov].append((wp, fact.global_id))

        # 2. Determine communication points (visible from any lander location)
        for l_loc in lander_locs:
            self.lander_visible_points.add(l_loc)
            for neighbor in visible_pairs[l_loc]:
                self.lander_visible_points.add(neighbor)

        # 3. Compute APSP for each rover using BFS
        for r in self.rovers:
            self._bfs_apsp(r)

    def _bfs_apsp(self, rover):
        adj = self.rover_graph[rover]
        nodes = list(adj.keys())
        all_nodes = set(nodes)
        for u in nodes:
            all_nodes.update(adj[u])
            
        for start_node in all_nodes:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.dist[rover][start_node][start_node] = 0
            
            while q:
                curr, d = q.popleft()
                self.dist[rover][start_node][curr] = d
                
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        self.dist[rover][start_node][nxt] = d + 1
                        q.append((nxt, d+1))

    def _get_dist(self, rover, u, v):
        if u == v: return 0
        return self.dist[rover][u].get(v, float('inf'))

    def is_true(self, state, fact_name):
        fid = self.fact_name_to_id.get(fact_name)
        if fid is not None:
            return (state >> fid) & 1
        return False
    
    def is_true_id(self, state, fid):
        return (state >> fid) & 1

    def _find_rover_loc(self, state, rover):
        for wp, fid in self.rover_at_facts[rover]:
            if (state >> fid) & 1:
                return wp
        return None

    def _cost_to_comm(self, rover, from_loc):
        if from_loc in self.lander_visible_points:
            return 1 # 1 action to communicate
        
        min_dist = float('inf')
        for comm_wp in self.lander_visible_points:
            d = self._get_dist(rover, from_loc, comm_wp)
            if d < min_dist:
                min_dist = d
        
        if min_dist == float('inf'):
            return float('inf')
        return min_dist + 1 # dist + communicate action

    def __call__(self, task_network, state):
        total_cost = 0
        
        for task in task_network:
            # Parse task arguments from name
            # Expected format: name(arg1,arg2) or name arg1 arg2
            name_clean = task.name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = name_clean.split()
            t_type = parts[0]
            
            # Is it a primitive operator?
            if hasattr(task, 'cost'): 
                # It's an Operator
                total_cost += task.cost
                continue

            # It's an AbstractTask
            if 'navigate' in t_type:
                # do_navigate1 ?x ?to or do_navigate2 ?x ?from ?to
                # Heuristic: distance from current rover loc to dest
                if len(parts) >= 3:
                    rover = parts[1]
                    dest = parts[-1]
                    r_loc = self._find_rover_loc(state, rover)
                    if r_loc:
                        total_cost += self._get_dist(rover, r_loc, dest)
            
            elif 'get_soil_data' in t_type:
                # get_soil_data ?from
                wp = parts[-1]
                
                # Check if already done
                if self.is_true(state, f"+communicated_soil_data[{wp}]"):
                    continue
                
                # Find best rover
                best_cost = float('inf')
                for r in self.rover_equipped_soil:
                    # If analysis held, just comm
                    if self.is_true(state, f"+have_soil_analysis[{r},{wp}]"):
                        r_loc = self._find_rover_loc(state, r)
                        if r_loc:
                            c = self._cost_to_comm(r, r_loc)
                            if c < best_cost: best_cost = c
                    else:
                        # Need to go sample
                        r_loc = self._find_rover_loc(state, r)
                        if r_loc:
                            move = self._get_dist(r, r_loc, wp)
                            if move == float('inf'): continue
                            
                            # Check store
                            store = self.stores.get(r)
                            drop = 1 if self.is_true(state, f"+full[{store}]") else 0
                            
                            # Sample + Comm
                            # After sampling, rover is at wp
                            comm = self._cost_to_comm(r, wp)
                            
                            c = move + drop + 1 + comm # +1 for sample
                            if c < best_cost: best_cost = c
                
                if best_cost != float('inf'):
                    total_cost += best_cost

            elif 'get_rock_data' in t_type:
                # get_rock_data ?from
                wp = parts[-1]
                if self.is_true(state, f"+communicated_rock_data[{wp}]"):
                    continue
                
                best_cost = float('inf')
                for r in self.rover_equipped_rock:
                    if self.is_true(state, f"+have_rock_analysis[{r},{wp}]"):
                        r_loc = self._find_rover_loc(state, r)
                        if r_loc:
                            c = self._cost_to_comm(r, r_loc)
                            if c < best_cost: best_cost = c
                    else:
                        r_loc = self._find_rover_loc(state, r)
                        if r_loc:
                            move = self._get_dist(r, r_loc, wp)
                            if move == float('inf'): continue
                            
                            store = self.stores.get(r)
                            drop = 1 if self.is_true(state, f"+full[{store}]") else 0
                            comm = self._cost_to_comm(r, wp)
                            
                            c = move + drop + 1 + comm
                            if c < best_cost: best_cost = c
                
                if best_cost != float('inf'):
                    total_cost += best_cost

            elif 'get_image_data' in t_type:
                # get_image_data ?o ?m
                if len(parts) >= 3:
                    obj = parts[1]
                    mode = parts[2]
                    
                    if self.is_true(state, f"+communicated_image_data[{obj},{mode}]"):
                        continue
                        
                    # Check if any rover has image
                    has_image = False
                    for r in self.rovers:
                        if self.is_true(state, f"+have_image[{r},{obj},{mode}]"):
                            has_image = True
                            r_loc = self._find_rover_loc(state, r)
                            if r_loc:
                                c = self._cost_to_comm(r, r_loc)
                                total_cost += c
                            break
                    if has_image: continue

                    # Estimate capture cost
                    best_cost = float('inf')
                    for r, cams in self.rover_cameras.items():
                        r_loc = self._find_rover_loc(state, r)
                        if not r_loc: continue
                        
                        for cam in cams:
                            if mode in self.camera_modes[cam]:
                                # Calc cost for this rover/camera
                                current_cost = 0
                                current_r_loc = r_loc
                                
                                # Calibration
                                if not self.is_true(state, f"+calibrated[{cam},{r}]"):
                                    target = self.camera_target.get(cam)
                                    if target:
                                        # Dist to nearest visible point of target
                                        calib_wps = self.objective_visible_from[target]
                                        d_calib = float('inf')
                                        best_calib_wp = None
                                        for w in calib_wps:
                                            d = self._get_dist(r, current_r_loc, w)
                                            if d < d_calib:
                                                d_calib = d
                                                best_calib_wp = w
                                        
                                        if d_calib != float('inf'):
                                            current_cost += d_calib + 1 # +1 calibrate
                                            current_r_loc = best_calib_wp
                                        else:
                                            current_cost = float('inf') # Cannot calibrate
                                
                                if current_cost == float('inf'): continue

                                # Take Image
                                img_wps = self.objective_visible_from[obj]
                                d_img = float('inf')
                                best_img_wp = None
                                for w in img_wps:
                                    d = self._get_dist(r, current_r_loc, w)
                                    if d < d_img:
                                        d_img = d
                                        best_img_wp = w
                                
                                if d_img != float('inf'):
                                    current_cost += d_img + 1 # +1 take_image
                                    # Communicate
                                    comm = self._cost_to_comm(r, best_img_wp)
                                    current_cost += comm
                                    
                                    if current_cost < best_cost:
                                        best_cost = current_cost
                    
                    if best_cost != float('inf'):
                        total_cost += best_cost
                        
            elif 'send_' in t_type:
                # Abstract task to send data
                # Usually param 1 is rover, param 2 is source/obj
                # Approximate as move to comm + 1
                if len(parts) >= 2:
                    rover = parts[1]
                    r_loc = self._find_rover_loc(state, rover)
                    if r_loc:
                        total_cost += self._cost_to_comm(rover, r_loc)

        return total_cost
