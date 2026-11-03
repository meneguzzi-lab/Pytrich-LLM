import sys
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic19(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_18=None):
        super().__init__(model)
        self.model = model
        self.max_int = 99999
        
        # Data structures for domain objects
        self.rovers = []
        self.waypoints = []
        self.objectives = []
        self.cameras = []
        self.modes = []
        self.stores = []
        self.landers = []
        
        # Fact lookup tables (mapping object tuples to global_id)
        self.at_facts = {} # (rover, waypoint) -> global_id
        self.empty_facts = {} # store -> global_id
        self.full_facts = {} # store -> global_id
        self.calibrated_facts = {} # (camera, rover) -> global_id
        self.communicated_soil = {} # waypoint -> global_id
        self.communicated_rock = {} # waypoint -> global_id
        self.communicated_image = {} # (objective, mode) -> global_id
        
        # Static relations extracted from initial state
        self.can_traverse = {} # rover -> set of (from, to) tuples
        self.visible = set() # (p1, p2)
        self.equipped_for_soil = set() # rovers
        self.equipped_for_rock = set() # rovers
        self.equipped_for_imaging = set() # rovers
        self.store_of = {} # rover -> store
        self.on_board = {} # rover -> list of cameras
        self.calibration_target = {} # camera -> objective
        self.supports = {} # camera -> set of modes
        self.visible_from = {} # objective -> set of waypoints
        self.at_lander = {} # lander -> waypoint
        
        # Initialization
        self._parse_domain_objects()
        self._compute_distances()
        self._compute_comm_points()

    def _parse_domain_objects(self):
        """
        Scans the grounded facts to identify objects and static relations.
        This builds the connectivity graph and capability maps.
        """
        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            if not pred: continue
            
            if pred == "at":
                # Dynamic fact, store mapping for state lookup
                if "rover" in args[0]: 
                    if args[0] not in self.rovers: self.rovers.append(args[0])
                if "waypoint" in args[1]:
                    if args[1] not in self.waypoints: self.waypoints.append(args[1])
                self.at_facts[(args[0], args[1])] = fact.global_id
                
            elif pred == "can_traverse":
                r, f, t = args
                if r not in self.can_traverse: self.can_traverse[r] = set()
                self.can_traverse[r].add((f, t))
                
            elif pred == "visible":
                self.visible.add((args[0], args[1]))
                
            elif pred == "equipped_for_soil_analysis":
                self.equipped_for_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_for_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_for_imaging.add(args[0])
                
            elif pred == "store_of":
                self.store_of[args[1]] = args[0] # Note: predicate is store_of ?s ?x
                if args[0] not in self.stores: self.stores.append(args[0])
                
            elif pred == "empty":
                self.empty_facts[args[0]] = fact.global_id
            elif pred == "full":
                self.full_facts[args[0]] = fact.global_id
                
            elif pred == "on_board":
                c, r = args
                if r not in self.on_board: self.on_board[r] = []
                self.on_board[r].append(c)
                if c not in self.cameras: self.cameras.append(c)
                
            elif pred == "calibration_target":
                self.calibration_target[args[0]] = args[1]
                if args[1] not in self.objectives: self.objectives.append(args[1])
                
            elif pred == "supports":
                c, m = args
                if c not in self.supports: self.supports[c] = set()
                self.supports[c].add(m)
                if m not in self.modes: self.modes.append(m)
                
            elif pred == "visible_from":
                o, w = args
                if o not in self.visible_from: self.visible_from[o] = set()
                self.visible_from[o].add(w)
                
            elif pred == "calibrated":
                self.calibrated_facts[(args[0], args[1])] = fact.global_id
                
            elif pred == "communicated_soil_data":
                self.communicated_soil[args[0]] = fact.global_id
            elif pred == "communicated_rock_data":
                self.communicated_rock[args[0]] = fact.global_id
            elif pred == "communicated_image_data":
                self.communicated_image[(args[0], args[1])] = fact.global_id
            
            elif pred == "at_lander":
                self.at_lander[args[0]] = args[1]
                if args[0] not in self.landers: self.landers.append(args[0])

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact string like '+at[rover0,waypoint1]'.
        """
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
        
    def _compute_distances(self):
        """
        Computes All-Pairs Shortest Paths (APSP) for each rover using BFS.
        Since `can_traverse` is rover-specific, distances depend on the rover.
        """
        self.distances = {}
        
        for r in self.rovers:
            self.distances[r] = {}
            # Build adjacency list for this rover
            adj = {}
            if r in self.can_traverse:
                for f, t in self.can_traverse[r]:
                    if f not in adj: adj[f] = []
                    adj[f].append(t)
            
            # BFS from each waypoint
            for start_node in self.waypoints:
                dists = {wp: self.max_int for wp in self.waypoints}
                dists[start_node] = 0
                q = deque([start_node])
                
                while q:
                    u = q.popleft()
                    if u in adj:
                        for v in adj[u]:
                            if dists[v] == self.max_int:
                                dists[v] = dists[u] + 1
                                q.append(v)
                self.distances[r][start_node] = dists

    def _compute_comm_points(self):
        """
        Identifies waypoints from which communication with a lander is possible.
        A waypoint is a comm point if it is at a lander or visible from a lander's location.
        """
        self.comm_points = set()
        lander_locs = set(self.at_lander.values())
        
        for wp in self.waypoints:
            can_comm = False
            if wp in lander_locs:
                can_comm = True
            else:
                for l_loc in lander_locs:
                    if (wp, l_loc) in self.visible or (l_loc, wp) in self.visible:
                        can_comm = True
                        break
            if can_comm:
                self.comm_points.add(wp)

    def _get_rover_location(self, rover, state):
        """Helper to find rover location from state bitmask."""
        for wp in self.waypoints:
            key = (rover, wp)
            if key in self.at_facts:
                gid = self.at_facts[key]
                if (state >> gid) & 1:
                    return wp
        return None

    def _is_store_full(self, store, state):
        if store in self.full_facts:
            gid = self.full_facts[store]
            return (state >> gid) & 1
        return False

    def _is_calibrated(self, camera, rover, state):
        key = (camera, rover)
        if key in self.calibrated_facts:
            gid = self.calibrated_facts[key]
            return (state >> gid) & 1
        return False

    def h(self, state, task_network):
        """
        Estimates the cost to complete the task network from the current state.
        Sum of independent estimated costs for each task.
        """
        total_cost = 0
        
        # Cache rover locations for this state to avoid repeated lookups
        rover_locations = {}
        for r in self.rovers:
            loc = self._get_rover_location(r, state)
            if loc: rover_locations[r] = loc
            
        for task in task_network:
            name = task.name
            pred, args = self._parse_fact_name(name)
            if not pred: 
                # Primitive actions roughly cost 1
                total_cost += 1
                continue

            if pred == "do_navigate1" or pred == "do_navigate2" or pred == "navigate":
                # Task: Move rover to target
                # Args: [rover, to] or [rover, from, to] or [rover, to, from] (m1)
                # We rely on arg naming convention or position. 
                # In domain: do_navigate1 ?x ?to. do_navigate2 ?x ?from ?to.
                rover = args[0]
                # For do_navigate2, the destination is last if args > 2, but let's assume 2nd arg is destination for nav1
                # However, navigate action is ?x ?p1 ?p2. Target is ?p2 (args[2])
                # Robust check:
                to_wp = args[-1]
                
                if rover in rover_locations:
                    dist = self.distances.get(rover, {}).get(rover_locations[rover], {}).get(to_wp, self.max_int)
                    if dist < self.max_int:
                        total_cost += dist
                    else:
                        total_cost += 5 # Unreachable penalty
                else:
                    total_cost += 1

            elif pred == "get_soil_data":
                wp = args[0]
                # If goal achieved, cost 0
                if wp in self.communicated_soil:
                    gid = self.communicated_soil[wp]
                    if (state >> gid) & 1:
                        continue
                
                # Best rover strategy: min cost across equipped rovers
                min_c = self.max_int
                for r in self.equipped_for_soil:
                    if r not in rover_locations: continue
                    r_loc = rover_locations[r]
                    
                    # 1. Travel to sample location
                    d_sample = self.distances[r][r_loc].get(wp, self.max_int)
                    if d_sample == self.max_int: continue
                    
                    # 2. Drop if full
                    store = self.store_of.get(r)
                    cost_drop = 1 if self._is_store_full(store, state) else 0
                    
                    # 3. Travel to communication point
                    d_comm = self.max_int
                    for cp in self.comm_points:
                        d = self.distances[r][wp].get(cp, self.max_int)
                        if d < d_comm: d_comm = d
                    
                    if d_comm == self.max_int: continue
                    
                    # Total: navigate + drop + sample(1) + navigate + comm(1)
                    c = d_sample + cost_drop + 1 + d_comm + 1
                    if c < min_c: min_c = c
                
                total_cost += min_c if min_c < self.max_int else 10

            elif pred == "get_rock_data":
                wp = args[0]
                if wp in self.communicated_rock:
                    gid = self.communicated_rock[wp]
                    if (state >> gid) & 1: continue
                
                min_c = self.max_int
                for r in self.equipped_for_rock:
                    if r not in rover_locations: continue
                    r_loc = rover_locations[r]
                    
                    d_sample = self.distances[r][r_loc].get(wp, self.max_int)
                    if d_sample == self.max_int: continue
                    
                    store = self.store_of.get(r)
                    cost_drop = 1 if self._is_store_full(store, state) else 0
                    
                    d_comm = self.max_int
                    for cp in self.comm_points:
                        d = self.distances[r][wp].get(cp, self.max_int)
                        if d < d_comm: d_comm = d
                    
                    if d_comm == self.max_int: continue
                    
                    c = d_sample + cost_drop + 1 + d_comm + 1
                    if c < min_c: min_c = c
                total_cost += min_c if min_c < self.max_int else 10

            elif pred == "get_image_data":
                obj, mode = args[0], args[1]
                if (obj, mode) in self.communicated_image:
                    gid = self.communicated_image[(obj, mode)]
                    if (state >> gid) & 1: continue
                
                min_c = self.max_int
                for r in self.equipped_for_imaging:
                    if r not in rover_locations: continue
                    r_loc = rover_locations[r]
                    
                    # Find compatible cameras on this rover
                    cameras = [c for c in self.on_board.get(r, []) if mode in self.supports.get(c, set())]
                    for cam in cameras:
                        # Calibration cost
                        cost_calib = 0
                        current_loc = r_loc
                        
                        if not self._is_calibrated(cam, r, state):
                            target = self.calibration_target.get(cam)
                            if not target: continue # Should not happen if well-defined
                            vis_pts = self.visible_from.get(target, set())
                            d_calib = self.max_int
                            best_calib_pt = None
                            
                            for p in vis_pts:
                                d = self.distances[r][r_loc].get(p, self.max_int)
                                if d < d_calib:
                                    d_calib = d
                                    best_calib_pt = p
                            
                            if d_calib == self.max_int: continue
                            cost_calib = d_calib + 1 # navigate + calibrate
                            current_loc = best_calib_pt
                        
                        # Imaging cost
                        vis_pts_obj = self.visible_from.get(obj, set())
                        d_img = self.max_int
                        best_img_pt = None
                        for p in vis_pts_obj:
                            d = self.distances[r][current_loc].get(p, self.max_int)
                            if d < d_img:
                                d_img = d
                                best_img_pt = p
                        
                        if d_img == self.max_int: continue
                        cost_img = d_img + 1 # navigate + take_image
                        
                        # Communication cost
                        d_comm = self.max_int
                        for cp in self.comm_points:
                            d = self.distances[r][best_img_pt].get(cp, self.max_int)
                            if d < d_comm: d_comm = d
                        
                        if d_comm == self.max_int: continue
                        
                        total = cost_calib + cost_img + d_comm + 1 # + send
                        if total < min_c: min_c = total
                        
                total_cost += min_c if min_c < self.max_int else 10

            elif pred == "empty_store":
                store = args[0]
                if self._is_store_full(store, state):
                    total_cost += 1
            
            elif pred == "do_calibrate":
                rover, cam = args[0], args[1]
                if not self._is_calibrated(cam, rover, state):
                    if rover in rover_locations:
                        r_loc = rover_locations[rover]
                        target = self.calibration_target.get(cam)
                        vis_pts = self.visible_from.get(target, set())
                        d_calib = self.max_int
                        for p in vis_pts:
                            d = self.distances[rover][r_loc].get(p, self.max_int)
                            if d < d_calib: d_calib = d
                        if d_calib < self.max_int:
                            total_cost += d_calib + 1
                        else:
                            total_cost += 5
                    else:
                        total_cost += 5

            elif pred.startswith("send_"):
                # Generic handling for send_soil_data, send_image_data, etc.
                # These tasks involve navigating to a comm point and communicating.
                rover = args[0]
                if rover in rover_locations:
                    r_loc = rover_locations[rover]
                    d_comm = self.max_int
                    for cp in self.comm_points:
                        d = self.distances[rover][r_loc].get(cp, self.max_int)
                        if d < d_comm: d_comm = d
                    if d_comm < self.max_int:
                        total_cost += d_comm + 1
                    else:
                        total_cost += 5
                else:
                    total_cost += 5
            
            else:
                # Default fallback for other tasks
                total_cost += 1
                
        return total_cost
