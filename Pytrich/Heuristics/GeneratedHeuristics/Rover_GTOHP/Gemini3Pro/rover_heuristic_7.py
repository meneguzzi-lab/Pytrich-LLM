import sys
import re
from collections import defaultdict
from Pytrich.core.heuristic import Heuristic

class RoverGemini3ProHeuristic7(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_6=None):
        super().__init__(model)
        self.INF = 99999
        
        # Data structures for static domain info
        self.waypoints = set()
        self.rovers = set()
        self.objectives = set()
        self.modes = set()
        self.cameras = set()
        self.stores = set()
        self.landers = set()
        
        # Graph and Maps
        self.adj = defaultdict(list)
        self.dist = defaultdict(lambda: defaultdict(lambda: self.INF))
        self.visible_from = defaultdict(set) # objective -> {waypoints}
        self.calibration_target = {} # camera -> objective
        self.on_board = {} # camera -> rover
        self.supports = defaultdict(set) # camera -> {modes}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_of = {} # store -> rover
        self.at_lander = {} # lander -> waypoint
        
        # Dynamic fact lookups (predicate -> list of (args, global_id))
        self.dynamic_fact_indices = defaultdict(list)
        
        self._parse_domain()
        self._compute_apsp()
        self._precompute_comm_points()

    def _parse_domain(self):
        # Regex for parsing grounded facts: +predicate[arg1,arg2]
        fact_pattern = re.compile(r'^[+-]?([a-zA-Z0-9_]+)\[(.*)\]$')
        
        for fact in self.model.facts:
            name = fact.name.strip()
            match = fact_pattern.match(name)
            if not match:
                continue
                
            pred = match.group(1)
            args_str = match.group(2)
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            
            # Static Predicates
            if pred == 'can_traverse':
                # Args: rover, from, to. 
                # We build a union graph for heuristic admissibility.
                if len(args) == 3:
                    u, v = args[1], args[2]
                    self.waypoints.add(u)
                    self.waypoints.add(v)
                    self.adj[u].append(v) 
                    
            elif pred == 'visible_from':
                obj, wp = args[0], args[1]
                self.visible_from[obj].add(wp)
                
            elif pred == 'calibration_target':
                cam, obj = args[0], args[1]
                self.calibration_target[cam] = obj
                
            elif pred == 'on_board':
                cam, rov = args[0], args[1]
                self.on_board[cam] = rov
                self.rovers.add(rov)
                self.cameras.add(cam)
                
            elif pred == 'supports':
                cam, mode = args[0], args[1]
                self.supports[cam].add(mode)
                
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(args[0])
                
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(args[0])
                
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(args[0])
                
            elif pred == 'store_of':
                store, rov = args[0], args[1]
                self.store_of[store] = rov
                
            elif pred == 'at_lander':
                lander, wp = args[0], args[1]
                self.at_lander[lander] = wp
                self.landers.add(lander)
                
            elif pred == 'visible':
                # Used for comms calculation later
                if not hasattr(self, 'vis_graph'): self.vis_graph = defaultdict(set)
                self.vis_graph[args[0]].add(args[1])

            # Dynamic Predicates (track indices for bitwise check)
            if pred in ['at', 'empty', 'full', 'calibrated']:
                self.dynamic_fact_indices[pred].append((args, fact.global_id))

    def _compute_apsp(self):
        # Floyd-Warshall Algorithm for All-Pairs Shortest Paths
        # This runs once during initialization.
        nodes = list(self.waypoints)
        for w in nodes:
            self.dist[w][w] = 0
        
        for u in self.adj:
            for v in self.adj[u]:
                self.dist[u][v] = 1 # Unit cost per traverse step
                
        for k in nodes:
            for i in nodes:
                if self.dist[i][k] == self.INF: continue
                for j in nodes:
                    if self.dist[k][j] == self.INF: continue
                    if self.dist[i][j] > self.dist[i][k] + self.dist[k][j]:
                        self.dist[i][j] = self.dist[i][k] + self.dist[k][j]

    def _precompute_comm_points(self):
        # Identify set of waypoints from which communication to ANY lander is possible
        self.comm_points = set()
        if not hasattr(self, 'vis_graph'): return 
        
        lander_locs = set(self.at_lander.values())
        
        for p in self.waypoints:
            for l_loc in lander_locs:
                # Rover domain requires: visible(p, lander_loc)
                # Sometimes p == l_loc is sufficient if domain defines visible(x,x)
                if p in self.vis_graph and l_loc in self.vis_graph[p]:
                    self.comm_points.add(p)
                    break
                elif p == l_loc: 
                    # Safety check if reflexivity not explicit in visible graph
                    self.comm_points.add(p)
                    break

    def get_current_state_info(self, state_int):
        # Extract current world state from bitwise integer
        rover_locs = {}
        # Iterate stored indices for 'at' predicate
        for args, gid in self.dynamic_fact_indices['at']:
            if (state_int >> gid) & 1:
                rover_locs[args[0]] = args[1]
                
        empty_stores = set()
        for args, gid in self.dynamic_fact_indices['empty']:
            if (state_int >> gid) & 1:
                empty_stores.add(args[0])
                
        calibrated_cams = set() # Set of (camera, rover)
        for args, gid in self.dynamic_fact_indices['calibrated']:
            if (state_int >> gid) & 1:
                calibrated_cams.add((args[0], args[1]))
                
        return rover_locs, empty_stores, calibrated_cams

    def h(self, node):
        state = node.state
        rover_locs, empty_stores, calibrated_cams = self.get_current_state_info(state)
        
        total_cost = 0
        
        for task in node.task_network:
            # Handle primitive actions (Operators)
            if hasattr(task, 'pos_precons'): # It is an Operator
                total_cost += getattr(task, 'cost', 1)
                continue
                
            # Handle Abstract Tasks
            name = task.name
            # Normalize task name string
            clean_name = name.replace('[', ' ').replace(']', '').replace(',', ' ')
            parts = clean_name.split()
            if not parts: continue
            t_type = parts[0]
            t_args = parts[1:]
            
            if "navigate" in t_type:
                # Task: do_navigate1 ?rover ?to
                if len(t_args) >= 2:
                    r, to = t_args[0], t_args[1]
                    if r in rover_locs:
                        total_cost += self.dist[rover_locs[r]][to]
            
            elif "get_soil_data" in t_type:
                # Task: get_soil_data ?waypoint
                if len(t_args) >= 1:
                    wp = t_args[0]
                    cost = self.estimate_sample_cost(wp, rover_locs, empty_stores, self.equipped_soil)
                    total_cost += cost
                    
            elif "get_rock_data" in t_type:
                # Task: get_rock_data ?waypoint
                if len(t_args) >= 1:
                    wp = t_args[0]
                    cost = self.estimate_sample_cost(wp, rover_locs, empty_stores, self.equipped_rock)
                    total_cost += cost
                    
            elif "get_image_data" in t_type:
                # Task: get_image_data ?objective ?mode
                if len(t_args) >= 2:
                    obj, mode = t_args[0], t_args[1]
                    cost = self.estimate_image_cost(obj, mode, rover_locs, calibrated_cams)
                    total_cost += cost
            
            elif "empty_store" in t_type:
                # Task: empty_store ?store ?rover
                if len(t_args) >= 1:
                    store = t_args[0]
                    if store not in empty_stores:
                        total_cost += 1 # Drop action
            
            elif "calibrate" in t_type:
                # Task: do_calibrate ?rover ?camera
                if len(t_args) >= 2:
                    r, c = t_args[0], t_args[1]
                    if (c, r) not in calibrated_cams:
                        target = self.calibration_target.get(c)
                        if target:
                            # dist to view target + calibrate action
                            cost = self.min_dist_to_view(rover_locs.get(r), target) + 1
                            total_cost += cost
                            
            elif "send_" in t_type:
                # Task: send_..._data. Subtask involving comms.
                # Usually handled in get_X_data, but if independent:
                # Needs nav to comm point + comm action
                if len(t_args) >= 1:
                    r = t_args[0]
                    if r in rover_locs:
                        total_cost += self.dist_to_comm(rover_locs[r]) + 1

        return total_cost

    def estimate_sample_cost(self, wp, rover_locs, empty_stores, equipped_rovers):
        # Estimate minimum cost for any capable rover to fetch sample at wp
        best = self.INF
        for r in equipped_rovers:
            if r not in rover_locs: continue
            r_pos = rover_locs[r]
            
            # 1. Navigate to sample point
            c = self.dist[r_pos][wp]
            
            # 2. Empty store if full
            store = None
            for s, rov in self.store_of.items():
                if rov == r:
                    store = s
                    break
            if store and store not in empty_stores:
                c += 1 # Drop
            
            c += 1 # Sample action
            
            # 3. Send Data (Nav to nearest comm + Comm action)
            # Rover is now at wp
            c += self.dist_to_comm(wp) + 1
            
            if c < best:
                best = c
        return best if best < self.INF else 0

    def estimate_image_cost(self, obj, mode, rover_locs, calibrated_cams):
        # Estimate cost to get image. 
        # Sequence: [Calibrate] -> Navigate to view obj -> Take Image -> Send
        best = self.INF
        
        for cam, supported_modes in self.supports.items():
            if mode in supported_modes:
                r = self.on_board.get(cam)
                if not r or r not in rover_locs: continue
                r_pos = rover_locs[r]
                
                c = 0
                curr_pos = r_pos
                
                # 1. Calibrate if needed
                if (cam, r) not in calibrated_cams:
                    target = self.calibration_target.get(cam)
                    if target:
                        # Move to where target is visible
                        d, view_pt = self.min_dist_to_view_pt(curr_pos, target)
                        c += d + 1 # Nav + Calibrate
                        curr_pos = view_pt 
                
                # 2. Go to visibility of objective
                d, view_pt = self.min_dist_to_view_pt(curr_pos, obj)
                c += d + 1 # Nav + Take Image
                curr_pos = view_pt
                
                # 3. Send Data
                c += self.dist_to_comm(curr_pos) + 1
                
                if c < best:
                    best = c
                    
        return best if best < self.INF else 0

    def min_dist_to_view(self, from_wp, obj):
        d, _ = self.min_dist_to_view_pt(from_wp, obj)
        return d

    def min_dist_to_view_pt(self, from_wp, obj):
        # Returns (min_dist, waypoint_to_view_from)
        best_d = self.INF
        best_wp = from_wp
        
        targets = self.visible_from.get(obj, set())
        if not targets: return 0, from_wp # Should not happen if soluble

        for t in targets:
            d = self.dist[from_wp][t]
            if d < best_d:
                best_d = d
                best_wp = t
        
        return best_d, best_wp

    def dist_to_comm(self, from_wp):
        # Distance to nearest communication point
        if from_wp in self.comm_points:
            return 0
        best = self.INF
        for cp in self.comm_points:
            d = self.dist[from_wp][cp]
            if d < best:
                best = d
        return best
