import sys
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic15(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_14=None):
        super().__init__(model)
        self.INF = 99999
        self._initialize_static_knowledge()

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

    def _initialize_static_knowledge(self):
        # Maps for domain structure
        self.waypoints = set()
        self.rovers = set()
        self.edges = {} # wp -> set(wp)
        self.visibility = {} # wp -> set(wp)
        self.lander_visible_points = set()
        self.objective_locs = {} # obj -> set(wp)
        
        # Rover capabilities
        self.rover_caps = {} # rover -> set('soil', 'rock', 'image')
        self.rover_cameras = {} # rover -> list(camera)
        self.camera_modes = {} # camera -> set(mode)
        
        # Fact helpers for state decoding
        self.fact_at_rover = [] # list of (rover, global_id)
        self.fact_empty_store = [] # list of (rover, global_id)
        
        # 1. Parse static facts to build the graph and maps
        store_map = {} # store -> rover
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            gid = fact.global_id
            
            if pred == 'can_traverse':
                r, f, t = args
                self.rovers.add(r)
                self.waypoints.add(f)
                self.waypoints.add(t)
                if f not in self.edges: self.edges[f] = set()
                if t not in self.edges: self.edges[t] = set()
                self.edges[f].add(t)
                self.edges[t].add(f)
                
            elif pred == 'visible':
                p1, p2 = args
                if p1 not in self.visibility: self.visibility[p1] = set()
                self.visibility[p1].add(p2)
                
            elif pred == 'at_lander':
                l, p = args
                self.lander_visible_points.add(p)
                
            elif pred == 'visible_from':
                o, p = args
                if o not in self.objective_locs: self.objective_locs[o] = set()
                self.objective_locs[o].add(p)
                
            elif pred == 'equipped_for_soil_analysis':
                r = args[0]
                if r not in self.rover_caps: self.rover_caps[r] = set()
                self.rover_caps[r].add('soil')
                
            elif pred == 'equipped_for_rock_analysis':
                r = args[0]
                if r not in self.rover_caps: self.rover_caps[r] = set()
                self.rover_caps[r].add('rock')
                
            elif pred == 'equipped_for_imaging':
                r = args[0]
                if r not in self.rover_caps: self.rover_caps[r] = set()
                self.rover_caps[r].add('image')
                
            elif pred == 'on_board':
                c, r = args
                if r not in self.rover_cameras: self.rover_cameras[r] = []
                self.rover_cameras[r].append(c)
                
            elif pred == 'supports':
                c, m = args
                if c not in self.camera_modes: self.camera_modes[c] = set()
                self.camera_modes[c].add(m)
                
            elif pred == 'store_of':
                s, r = args
                store_map[s] = r

            elif pred == 'at':
                # Store mapping for dynamic checking later
                if args[0].startswith('rover'):
                    self.fact_at_rover.append((args[0], args[1], gid))

        # 2. Second pass for store facts now that we have store_map
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            gid = fact.global_id
            if pred == 'empty':
                s = args[0]
                if s in store_map:
                    self.fact_empty_store.append((store_map[s], gid))

        # 3. Identify Communication Points (Waypoints visible to a lander)
        self.comm_points = set()
        for p1 in self.visibility:
            for p2 in self.visibility[p1]:
                if p2 in self.lander_visible_points:
                    self.comm_points.add(p1)
        # Include lander points themselves if they are reflexive (often implied)
        self.comm_points.update(self.lander_visible_points)

        # 4. Precompute All-Pairs Shortest Paths (Floyd-Warshall)
        # Map waypoints to indices for matrix
        self.wp_list = list(self.waypoints)
        self.wp_to_idx = {w: i for i, w in enumerate(self.wp_list)}
        n = len(self.wp_list)
        
        # Initialize distance matrix
        self.d = [[self.INF]*n for _ in range(n)]
        for i in range(n): self.d[i][i] = 0
        
        for u, neighbors in self.edges.items():
            ui = self.wp_to_idx.get(u)
            if ui is not None:
                for v in neighbors:
                    vi = self.wp_to_idx.get(v)
                    if vi is not None:
                        self.d[ui][vi] = 1
                        
        # Run FW algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.d[i][k] + self.d[k][j] < self.d[i][j]:
                        self.d[i][j] = self.d[i][k] + self.d[k][j]

    def get_dist(self, w1, w2):
        if w1 not in self.wp_to_idx or w2 not in self.wp_to_idx: return self.INF
        return self.d[self.wp_to_idx[w1]][self.wp_to_idx[w2]]

    def compute(self, node):
        state = node.state
        tn = node.task_network
        
        # 1. Parse dynamic state efficiently
        rover_positions = {}
        for r, wp, gid in self.fact_at_rover:
            if (state >> gid) & 1:
                rover_positions[r] = wp
                
        rover_stores_empty = set()
        for r, gid in self.fact_empty_store:
            if (state >> gid) & 1:
                rover_stores_empty.add(r)
                
        # 2. Estimate cost for each task in the network
        total_cost = 0
        
        for task in tn:
            # task.name format example: "get_soil_data waypoint0" or "do_navigate rover0 waypoint1"
            parts = task.name.split()
            if not parts: continue
            tname = parts[0]
            args = parts[1:]
            
            if tname == 'get_soil_data' or tname == 'get_rock_data':
                # args: [waypoint]
                wp = args[0]
                type_ = 'soil' if 'soil' in tname else 'rock'
                
                # Find minimal cost among capable rovers
                min_task_cost = self.INF
                for r, pos in rover_positions.items():
                    if r not in self.rover_caps or type_ not in self.rover_caps[r]: continue
                    
                    # Cost: Navigate + Sample + NavigateToComm + Communicate
                    dist_to_sample = self.get_dist(pos, wp)
                    if dist_to_sample == self.INF: continue
                    
                    current_cost = dist_to_sample + 1 # +1 for sample action
                    
                    # If store full, need drop action
                    if r not in rover_stores_empty:
                        current_cost += 1
                        
                    # Navigate to nearest comm point from sample location
                    min_comm_dist = self.INF
                    for cp in self.comm_points:
                        d = self.get_dist(wp, cp)
                        if d < min_comm_dist:
                            min_comm_dist = d
                    
                    if min_comm_dist != self.INF:
                        current_cost += min_comm_dist + 1 # +1 for communicate action
                    else:
                        current_cost += 100 # Penalty if unreachable
                        
                    if current_cost < min_task_cost:
                        min_task_cost = current_cost
                
                total_cost += min_task_cost if min_task_cost != self.INF else 50

            elif tname == 'get_image_data':
                # args: [objective, mode]
                obj = args[0]
                mode = args[1]
                
                min_task_cost = self.INF
                
                for r, pos in rover_positions.items():
                    if r not in self.rover_caps or 'image' not in self.rover_caps[r]: continue
                    
                    # Check if rover has camera with required mode
                    valid_cam = False
                    cameras = self.rover_cameras.get(r, [])
                    for cam in cameras:
                        if mode in self.camera_modes.get(cam, set()):
                            valid_cam = True
                            break
                    if not valid_cam: continue
                    
                    # Navigate to a viewing point for the objective
                    view_points = self.objective_locs.get(obj, set())
                    if not view_points: continue
                    
                    # Find best viewing point (closest to rover + closest to comm)
                    best_vp_cost = self.INF
                    
                    for vp in view_points:
                        d_to_vp = self.get_dist(pos, vp)
                        if d_to_vp == self.INF: continue
                        
                        # From vp, need to communicate
                        min_comm_dist = self.INF
                        for cp in self.comm_points:
                            d_comm = self.get_dist(vp, cp)
                            if d_comm < min_comm_dist:
                                min_comm_dist = d_comm
                        
                        if min_comm_dist != self.INF:
                            # Path: r->vp (nav) + calibrate (1) + take_image (1) + vp->comm (nav) + comm (1)
                            # Simplified: d_to_vp + 1 + 1 + min_comm_dist + 1
                            cost = d_to_vp + min_comm_dist + 3
                            if cost < best_vp_cost:
                                best_vp_cost = cost
                                
                    if best_vp_cost < min_task_cost:
                        min_task_cost = best_vp_cost
                        
                total_cost += min_task_cost if min_task_cost != self.INF else 50

            elif tname in ['send_soil_data', 'send_rock_data', 'send_image_data']:
                # args: ?rover ?waypoint/obj ...
                # If this is in TN, sample is likely already taken. Just nav to comm point.
                if len(args) >= 1:
                    r = args[0]
                    if r in rover_positions:
                        r_pos = rover_positions[r]
                        min_d = self.INF
                        for cp in self.comm_points:
                            d = self.get_dist(r_pos, cp)
                            if d < min_d: min_d = d
                        if min_d != self.INF:
                            total_cost += min_d + 1
                        else:
                            total_cost += 50

            elif tname == 'do_navigate1' or tname == 'do_navigate2' or tname == 'navigate':
                # do_navigate1 ?x ?to
                # do_navigate2 ?x ?from ?to
                r = args[0]
                to = args[1] if tname == 'do_navigate1' else (args[2] if len(args) > 2 else None)
                
                if r in rover_positions and to:
                    dist = self.get_dist(rover_positions[r], to)
                    if dist != self.INF:
                        total_cost += dist
                    else:
                        total_cost += 50
            
            elif tname == 'empty_store':
                total_cost += 1
            elif tname == 'do_calibrate':
                total_cost += 1
            else:
                # Base cost for other actions or tasks
                total_cost += 1

        return total_cost