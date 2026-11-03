import sys
from typing import List, Union, Set, Dict, Tuple
from collections import defaultdict, deque
import math

# Assuming the provided classes exist in the environment (imported by the planner)
# from pytrich_planner import Heuristic, Model, Node ... 

class RoverGemini3ProHeuristic20(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_19=None):
        super().__init__(model)
        self.infinity = float('inf')
        
        # --- Static Analysis Data Structures ---
        self.waypoints = set()
        self.rovers = set()
        self.store_map = {} # store -> rover
        self.rover_stores = {} # rover -> store
        self.adj = defaultdict(list)
        self.distances = {} # (w1, w2) -> int
        
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        
        self.cameras = set()
        self.camera_on_board = {} # camera -> rover
        self.rover_cameras = defaultdict(list)
        self.camera_supports = defaultdict(set) # camera -> set(modes)
        self.camera_target = {} # camera -> objective
        
        self.visible_from = defaultdict(list) # objective -> [waypoints]
        self.calibration_targets = {} # camera -> objective
        
        self.lander_visible_locs = set() # Locations visible to a lander
        
        # --- Mappings for Fast State Lookup (Dynamic Predicates) ---
        # Maps grounded arguments to global_id for O(1) state checks
        self.fact_id_at = defaultdict(dict) # rover -> {waypoint: global_id}
        self.fact_id_empty = {} # store -> global_id
        self.fact_id_calibrated = defaultdict(dict) # camera -> {rover: global_id}
        self.fact_id_have_soil = defaultdict(dict) # rover -> {waypoint: global_id}
        self.fact_id_have_rock = defaultdict(dict) # rover -> {waypoint: global_id}
        self.fact_id_have_image = defaultdict(lambda: defaultdict(dict)) # rover -> objective -> mode -> global_id
        
        self._analyze_static_facts()
        self._compute_distances()

    def _parse_atom(self, atom_name: str):
        """
        Parses a grounded atom string like '+at[rover0,waypoint1]' or 'get_soil_data[waypoint0]'.
        Returns (predicate, args_list, is_positive).
        """
        name = atom_name.strip()
        is_pos = True
        if name.startswith("+"):
            name = name[1:]
        elif name.startswith("-"):
            name = name[1:]
            is_pos = False
            
        if "[" in name and name.endswith("]"):
            pred = name[:name.index("[")]
            args = name[name.index("[")+1:-1].split(",")
            return pred, args, is_pos
        return name, [], is_pos

    def _analyze_static_facts(self):
        """
        One-time parsing of the problem's static facts to build connectivity graphs,
        capability maps, and visibility lookups.
        """
        landers = set()
        visible_pairs = defaultdict(list)
        at_lander_locs = set()

        for fact in self.model.facts:
            pred, args, is_pos = self._parse_atom(fact.name)
            if not is_pos: continue
            
            gid = fact.global_id
            
            if pred == "can_traverse":
                # can_traverse ?r ?from ?to
                # In Rover, traverse depends on the rover, but usually the map is consistent.
                # We build a general adjacency graph assuming connectivity is structural.
                if len(args) == 3:
                    r, f, t = args
                    self.waypoints.add(f)
                    self.waypoints.add(t)
                    self.rovers.add(r)
                    self.adj[f].append(t)
                
            elif pred == "store_of":
                s, r = args
                self.store_map[s] = r
                self.rover_stores[r] = s
                
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])
                
            elif pred == "on_board":
                c, r = args
                self.cameras.add(c)
                self.camera_on_board[c] = r
                self.rover_cameras[r].append(c)
                
            elif pred == "supports":
                c, m = args
                self.camera_supports[c].add(m)
                
            elif pred == "calibration_target":
                c, o = args
                self.calibration_targets[c] = o
                
            elif pred == "visible_from":
                o, w = args
                self.visible_from[o].append(w)
                
            elif pred == "at_lander":
                l, w = args
                landers.add(l)
                at_lander_locs.add(w)
                
            elif pred == "visible":
                p1, p2 = args
                visible_pairs[p1].append(p2)
                visible_pairs[p2].append(p1)
                
            # --- Indexing Dynamic Facts for State Checking ---
            elif pred == "at":
                r, w = args
                self.fact_id_at[r][w] = gid
            elif pred == "empty":
                s = args[0]
                self.fact_id_empty[s] = gid
            elif pred == "calibrated":
                c, r = args
                self.fact_id_calibrated[c][r] = gid
            elif pred == "have_soil_analysis":
                r, w = args
                self.fact_id_have_soil[r][w] = gid
            elif pred == "have_rock_analysis":
                r, w = args
                self.fact_id_have_rock[r][w] = gid
            elif pred == "have_image":
                r, o, m = args
                self.fact_id_have_image[r][o][m] = gid

        # Identify all locations from which we can communicate (either at lander or visible to it)
        for w in self.waypoints:
            if w in at_lander_locs:
                self.lander_visible_locs.add(w)
            else:
                for l_loc in at_lander_locs:
                    if l_loc in visible_pairs[w]:
                        self.lander_visible_locs.add(w)
                        break
        # Add any lander locations not in waypoints (static definition idiosyncrasy check)
        for l_loc in at_lander_locs:
            self.lander_visible_locs.add(l_loc)

    def _compute_distances(self):
        """
        Computes All-Pairs Shortest Paths using BFS since edge weights are uniform (1).
        Used for navigation heuristics.
        """
        all_nodes = list(self.waypoints)
        for start_node in all_nodes:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0
            while q:
                curr, d = q.popleft()
                for nxt in self.adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        self.distances[(start_node, nxt)] = d + 1
                        q.append((nxt, d + 1))
        
    # --- Helper Methods for State Queries ---
    
    def _get_rover_pos(self, rover, state_bitwise):
        """Returns the current waypoint of the rover from the bitwise state."""
        for wp, gid in self.fact_id_at[rover].items():
            if (state_bitwise >> gid) & 1:
                return wp
        return None

    def _is_store_empty(self, rover, state_bitwise):
        s = self.rover_stores.get(rover)
        if s and s in self.fact_id_empty:
            gid = self.fact_id_empty[s]
            return (state_bitwise >> gid) & 1
        return False

    def _is_calibrated(self, camera, rover, state_bitwise):
        if camera in self.fact_id_calibrated and rover in self.fact_id_calibrated[camera]:
            gid = self.fact_id_calibrated[camera][rover]
            return (state_bitwise >> gid) & 1
        return False

    def _dist(self, w1, w2):
        if w1 == w2: return 0
        return self.distances.get((w1, w2), self.infinity)

    def _nearest_dist(self, w_from, targets):
        """Distance from w_from to the closest target waypoint."""
        if not targets: return self.infinity
        dists = [self._dist(w_from, t) for t in targets]
        return min(dists) if dists else self.infinity

    # --- Task Estimation Logic ---

    def _estimate_navigate(self, rover, target, state_bitwise):
        pos = self._get_rover_pos(rover, state_bitwise)
        if not pos: return self.infinity
        return self._dist(pos, target)

    def _estimate_get_soil(self, loc, state_bitwise):
        """
        Estimates cost to get soil data from 'loc'.
        Logic: Min over all equipped rovers: Navigate -> (Drop) -> Sample -> Navigate Comm -> Comm.
        """
        candidates = []
        for r in self.equipped_soil:
            pos = self._get_rover_pos(r, state_bitwise)
            if not pos: continue
            
            # Check if analysis already held
            if loc in self.fact_id_have_soil[r]:
                gid = self.fact_id_have_soil[r][loc]
                if (state_bitwise >> gid) & 1:
                    # Only need to communicate
                    d_comm = self._nearest_dist(pos, self.lander_visible_locs)
                    candidates.append(d_comm + 1)
                    continue

            cost = 0
            # 1. Navigate to sample location
            cost += self._dist(pos, loc)
            # 2. Empty store if not empty (heuristic assumption: assumes drop is needed)
            if not self._is_store_empty(r, state_bitwise):
                cost += 1
            # 3. Sample action
            cost += 1
            # 4. Navigate to communication point
            cost += self._nearest_dist(loc, self.lander_visible_locs)
            # 5. Communicate action
            cost += 1
            candidates.append(cost)
            
        return min(candidates) if candidates else self.infinity

    def _estimate_get_rock(self, loc, state_bitwise):
        """Similar to soil but for rock analysis."""
        candidates = []
        for r in self.equipped_rock:
            pos = self._get_rover_pos(r, state_bitwise)
            if not pos: continue
            
            if loc in self.fact_id_have_rock[r]:
                gid = self.fact_id_have_rock[r][loc]
                if (state_bitwise >> gid) & 1:
                    d_comm = self._nearest_dist(pos, self.lander_visible_locs)
                    candidates.append(d_comm + 1)
                    continue

            cost = 0
            cost += self._dist(pos, loc)
            if not self._is_store_empty(r, state_bitwise):
                cost += 1
            cost += 1 # sample
            cost += self._nearest_dist(loc, self.lander_visible_locs)
            cost += 1 # comm
            candidates.append(cost)
        return min(candidates) if candidates else self.infinity

    def _estimate_get_image(self, obj, mode, state_bitwise):
        """
        Estimates cost to get image data.
        Logic: Min over equipped rovers: Calibrate (if needed) -> Navigate View -> Take Image -> Navigate Comm -> Comm.
        """
        candidates = []
        for r in self.equipped_imaging:
            pos = self._get_rover_pos(r, state_bitwise)
            if not pos: continue
            
            # Does rover have a camera that supports this mode?
            valid_cams = [c for c in self.rover_cameras[r] if mode in self.camera_supports[c]]
            if not valid_cams: continue
            
            # Check if image already held
            if self.fact_id_have_image[r][obj].get(mode) is not None:
                gid = self.fact_id_have_image[r][obj][mode]
                if (state_bitwise >> gid) & 1:
                    d_comm = self._nearest_dist(pos, self.lander_visible_locs)
                    candidates.append(d_comm + 1)
                    continue

            # Calculate cost for each valid camera
            cam_costs = []
            for c in valid_cams:
                c_cost = 0
                curr_pos = pos
                
                # Calibration Step
                if not self._is_calibrated(c, r, state_bitwise):
                    calib_obj = self.calibration_targets.get(c)
                    if calib_obj:
                        calib_locs = self.visible_from[calib_obj]
                        if not calib_locs:
                            c_cost += self.infinity
                        else:
                            # For heuristic lower bound, assume we can take the path:
                            # Pos -> Best Calibration Point -> Best Image Point
                            img_locs = self.visible_from[obj]
                            if not img_locs:
                                c_cost += self.infinity
                            else:
                                # Find best (calib_pt, img_pt) pair minimizing total distance
                                best_path = self.infinity
                                for cl in calib_locs:
                                    d1 = self._dist(pos, cl)
                                    for il in img_locs:
                                        d2 = self._dist(cl, il)
                                        if d1 + d2 < best_path:
                                            best_path = d1 + d2
                                            final_img_loc = il
                                
                                c_cost += best_path + 2 # +1 calibrate, +1 take_image
                                # Move to communication
                                c_cost += self._nearest_dist(final_img_loc, self.lander_visible_locs) + 1
                    else:
                        c_cost += self.infinity # No calibration target defined
                else:
                    # Already calibrated
                    img_locs = self.visible_from[obj]
                    if not img_locs:
                        c_cost += self.infinity
                    else:
                        # Path: Pos -> Best Image Point -> Best Comm Point
                        best_path = self.infinity
                        for il in img_locs:
                            d1 = self._dist(pos, il)
                            d2 = self._nearest_dist(il, self.lander_visible_locs)
                            if d1 + d2 < best_path:
                                best_path = d1 + d2
                        c_cost += best_path + 2 # +1 take, +1 comm
                
                cam_costs.append(c_cost)
            
            if cam_costs:
                candidates.append(min(cam_costs))
                
        return min(candidates) if candidates else self.infinity

    def __call__(self, node):
        """
        Calculates the heuristic value for the given node (Task Network).
        Sum of estimated costs for all tasks in the network.
        """
        state = node.state
        tn = node.task_network
        
        total_cost = 0
        
        for task in tn:
            # If it's a primitive operator (leaf), add its cost
            if hasattr(task, 'cost'):
                total_cost += task.cost
                continue
                
            # Parse Abstract Task
            pred, args, _ = self._parse_atom(task.name)
            
            if pred == "do_navigate1" or pred == "do_navigate2":
                # do_navigate1 ?r ?to / do_navigate2 ?r ?from ?to
                r = args[0]
                to = args[-1]
                total_cost += self._estimate_navigate(r, to, state)
                
            elif pred == "get_soil_data":
                # get_soil_data ?loc
                loc = args[0]
                total_cost += self._estimate_get_soil(loc, state)
                
            elif pred == "get_rock_data":
                loc = args[0]
                total_cost += self._estimate_get_rock(loc, state)
                
            elif pred == "get_image_data":
                # get_image_data ?obj ?mode
                obj, mode = args
                total_cost += self._estimate_get_image(obj, mode, state)
            
            elif pred in ["send_soil_data", "send_rock_data", "send_image_data"]:
                # Tasks representing just the communication phase
                r = args[0]
                pos = self._get_rover_pos(r, state)
                if pos:
                    d = self._nearest_dist(pos, self.lander_visible_locs)
                    total_cost += d + 1
                else:
                    total_cost += self.infinity
            
            elif pred == "empty_store":
                s, r = args
                if not self._is_store_empty(r, state):
                    total_cost += 1
            
            elif pred == "do_calibrate":
                r, c = args
                if not self._is_calibrated(c, r, state):
                    calib_obj = self.calibration_targets.get(c)
                    if calib_obj:
                        targets = self.visible_from[calib_obj]
                        pos = self._get_rover_pos(r, state)
                        if pos:
                            total_cost += self._nearest_dist(pos, targets) + 1
                        else:
                            total_cost += self.infinity
            
            else:
                # Fallback for unknown tasks or primitive actions wrapped as tasks
                # Minimal cost assumption
                pass
                
        return total_cost
