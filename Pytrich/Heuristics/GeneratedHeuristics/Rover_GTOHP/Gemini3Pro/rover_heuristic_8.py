import sys
import math
from collections import defaultdict, deque
from typing import List, Set, Tuple, Dict, Optional
from pytrich.heuristics.heuristic import Heuristic
from pytrich.pddl.operator import Operator
from pytrich.hddl.abstract_task import AbstractTask

class RoverGemini3ProHeuristic8(Heuristic):
    """
    A domain-dependent heuristic for the Rover HDDL domain.
    
    This heuristic leverages the hierarchical structure of the Rover domain to estimate
    the remaining cost to achieve the goal tasks. It precomputes All-Pairs Shortest Paths (APSP)
    for each rover's navigation graph and analyzes equipment capabilities.
    
    Key strategies:
    1. **Precomputation**: Builds connectivity graphs per rover and identifies valid comms/imaging locations.
    2. **State Abstraction**: Efficiently extracts dynamic state info (locations, calibration, storage) using bitwise masks.
    3. **Task Decomposition Costing**: Estimates costs for high-level tasks (get_soil_data, get_image_data)
       by finding the optimal rover/resource allocation (min-cost rover selection).
    4. **Goal Awareness**: Checks if high-level objectives (communicated_data) are already met in the state to avoid redundant costing.
    """

    def __init__(self, model, rover_gemini3pro_heuristic_7=None):
        super().__init__(model)
        self.model = model
        self.INF = 99999

        # --- 1. Domain Object Identification ---
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()

        # --- 2. Static Relation Maps ---
        # Graph: rover -> u -> [v]
        self.rover_graph = defaultdict(lambda: defaultdict(list))
        self.visible = defaultdict(set)
        self.visible_from = defaultdict(set)  # objective -> {waypoints}
        self.calibration_targets = {}         # camera -> objective
        self.supports = defaultdict(set)      # camera -> {modes}
        self.on_board = defaultdict(set)      # rover -> {cameras}
        self.store_of = {}                    # store -> rover
        self.rover_store = {}                 # rover -> store
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        self.at_lander_locs = set()           # waypoints where landers are
        
        # --- 3. Index Maps for Fast State Lookup ---
        # Maps for dynamic predicates: pred_name -> arg_tuple -> global_id
        self.fact_ids = defaultdict(dict)
        # Reverse maps for efficient iteration: pred_name -> list of (global_id, args...)
        self.fact_list = defaultdict(list)

        self._parse_static_facts()
        self._compute_apsp()
        self._identify_comm_points()

    def _parse_args(self, name: str) -> Tuple[str, List[str]]:
        """Parses a grounded fact/task name like '+at[rover0,waypoint1]'"""
        name = name.strip()
        is_neg = False
        if name.startswith("+"):
            name = name[1:]
        elif name.startswith("-"):
            name = name[1:]
            is_neg = True
            
        if "[" not in name:
            return name, []
            
        pred = name[:name.find("[")]
        args_str = name[name.find("[")+1 : name.find("]")]
        args = [a.strip() for a in args_str.split(",")] if args_str else []
        return pred, args

    def _parse_static_facts(self):
        """Iterates through all model facts to build static knowledge."""
        # We iterate all facts to find static relations and build ID maps for dynamic ones
        for f in self.model.facts:
            pred, args = self._parse_args(f.name)
            
            # Store ID lookup
            if args:
                self.fact_ids[pred][tuple(args)] = f.global_id
                self.fact_list[pred].append((f.global_id, args))
            else:
                self.fact_ids[pred][()] = f.global_id

            # Static Knowledge Extraction (using initial state logic for rigid relations)
            # Note: In Pytrich, model.facts includes all possible ground facts.
            # We check initial_state for rigid relations that don't change.
            is_true_initially = ((self.model.initial_state >> f.global_id) & 1)
            
            if is_true_initially:
                if pred == "can_traverse":
                    r, p1, p2 = args
                    self.rover_graph[r][p1].append(p2)
                    self.rovers.add(r)
                    self.waypoints.add(p1)
                    self.waypoints.add(p2)
                elif pred == "visible":
                    p1, p2 = args
                    self.visible[p1].add(p2)
                elif pred == "visible_from":
                    o, p = args
                    self.visible_from[o].add(p)
                    self.objectives.add(o)
                elif pred == "calibration_target":
                    c, o = args
                    self.calibration_targets[c] = o
                    self.cameras.add(c)
                elif pred == "supports":
                    c, m = args
                    self.supports[c].add(m)
                    self.modes.add(m)
                elif pred == "on_board":
                    c, r = args
                    self.on_board[r].add(c)
                elif pred == "store_of":
                    s, r = args
                    self.store_of[s] = r
                    self.rover_store[r] = s
                    self.stores.add(s)
                elif pred == "equipped_for_soil_analysis":
                    self.equipped_for_soil.add(args[0])
                elif pred == "equipped_for_rock_analysis":
                    self.equipped_for_rock.add(args[0])
                elif pred == "equipped_for_imaging":
                    self.equipped_for_imaging.add(args[0])
                elif pred == "at_lander":
                    l, p = args
                    self.at_lander_locs.add(p)
                    self.landers.add(l)

    def _compute_apsp(self):
        """Computes All-Pairs Shortest Paths for each rover's navigation graph."""
        # self.dist[rover][start][end] = steps
        self.dist = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: self.INF)))
        
        for r in self.rovers:
            nodes = list(self.waypoints)
            # Initialize distance 0 to self
            for w in nodes:
                self.dist[r][w][w] = 0
            
            # Run BFS from each node to handle unweighted graphs efficiently
            # Since edges are directed (can_traverse), BFS respects direction
            for start_node in nodes:
                queue = deque([(start_node, 0)])
                visited = {start_node}
                
                while queue:
                    curr, d = queue.popleft()
                    # Neighbors
                    for nxt in self.rover_graph[r][curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            self.dist[r][start_node][nxt] = d + 1
                            self.dist[r][nxt][start_node] = d + 1 # undirected assumption usually holds for rover, but logic uses can_traverse
                            queue.append((nxt, d + 1))
            
            # Fix for directed graphs if domain defines one-way paths (rare in rover but possible)
            # The above BFS assumes if A->B then dist is found. We iterate all starts, so it covers directed.
            pass

    def _identify_comm_points(self):
        """Identifies all waypoints from which communication with a lander is possible."""
        self.comm_waypoints = set()
        # Direct visibility to lander location
        # Comm possible if at p1, lander at p2, and visible p1 p2
        for w1 in self.waypoints:
            for l_loc in self.at_lander_locs:
                # Check visibility (often symmetric, but check both directions or visible predicate)
                # predicate is (visible ?w1 ?w2)
                if l_loc in self.visible[w1] or w1 == l_loc:
                    self.comm_waypoints.add(w1)

    def _get_rover_loc(self, state_bitwise, rover):
        """Finds the current location of a rover from the state."""
        # This is a bit slow if we iterate all waypoints. 
        # Optimization: cache the list of 'at' facts for this rover.
        # Only iterate waypoints relevant to 'at' facts.
        # Using pre-parsed list
        
        # Iterate all 'at' facts for this rover (filter from self.fact_list['at'])
        # Since looping all 'at' facts might be slow, we rely on the fact that rovers are few.
        # Better: iterate self.fact_list['at'] and check if arg0 is rover.
        
        # Optimization: In init, build self.rover_at_facts[rover] = list of (gid, waypoint)
        if not hasattr(self, 'rover_at_facts_cache'):
            self.rover_at_facts_cache = defaultdict(list)
            for gid, args in self.fact_list['at']:
                self.rover_at_facts_cache[args[0]].append((gid, args[1]))
        
        for gid, wp in self.rover_at_facts_cache[rover]:
            if (state_bitwise >> gid) & 1:
                return wp
        return None

    def compute(self, state, task_network):
        total_cost = 0
        
        # --- 1. Snapshot Dynamic State ---
        # Only fetch what is needed on demand to save time, or cache essential
        # Since we might need locations often, let's cache them lazily or inside the loop
        rover_locs = {} 
        
        def get_r_loc(r):
            if r not in rover_locs:
                rover_locs[r] = self._get_rover_loc(state, r)
            return rover_locs[r]

        # --- 2. Iterate Task Network ---
        for task in task_network:
            pred, args = self._parse_args(task.name)
            
            # --- A. Navigation ---
            if pred in ["do_navigate1", "do_navigate2"]:
                # args: [rover, to] or [rover, from, to]
                r = args[0]
                to_wp = args[-1]
                curr = get_r_loc(r)
                if curr:
                    dist = self.dist[r][curr][to_wp]
                    if dist != self.INF:
                        total_cost += dist
                    else:
                        total_cost += 100 # Penalty for unreachable
            
            elif pred == "navigate": # Primitive
                total_cost += 1

            # --- B. Soil / Rock Data ---
            elif pred in ["get_soil_data", "get_rock_data"]:
                # args: [waypoint]
                w = args[0]
                is_soil = (pred == "get_soil_data")
                goal_pred = "communicated_soil_data" if is_soil else "communicated_rock_data"
                
                # 1. Check if already communicated
                gid_comm = self.fact_ids[goal_pred].get(tuple([w]))
                if gid_comm is not None and ((state >> gid_comm) & 1):
                    continue # Done
                
                # 2. Check if held
                have_pred = "have_soil_analysis" if is_soil else "have_rock_analysis"
                # Check if any rover has it
                held_by = None
                for gid, f_args in self.fact_list[have_pred]:
                    # f_args = [rover, waypoint]
                    if f_args[1] == w:
                        if (state >> gid) & 1:
                            held_by = f_args[0]
                            break
                
                if held_by:
                    # Just need to send: navigate to comm + communicate
                    curr = get_r_loc(held_by)
                    if curr:
                        min_comm_dist = self.INF
                        for comm_wp in self.comm_waypoints:
                            d = self.dist[held_by][curr][comm_wp]
                            if d < min_comm_dist:
                                min_comm_dist = d
                        if min_comm_dist != self.INF:
                            total_cost += (min_comm_dist + 1)
                    continue

                # 3. Not held, need to acquire
                # Find best rover
                candidates = self.equipped_for_soil if is_soil else self.equipped_for_rock
                best_acq_cost = self.INF
                
                for r in candidates:
                    curr = get_r_loc(r)
                    if not curr: continue
                    
                    # Nav to sample
                    d_to_sample = self.dist[r][curr][w]
                    
                    # Empty store check
                    store = self.rover_store.get(r)
                    drop_cost = 0
                    if store:
                        # Check if full
                        full_gid = self.fact_ids['full'].get(tuple([store]))
                        if full_gid is not None and ((state >> full_gid) & 1):
                            drop_cost = 1
                            
                    # Nav to comm after sampling (sampling happens at w)
                    d_to_comm = self.INF
                    for comm_wp in self.comm_waypoints:
                        d = self.dist[r][w][comm_wp]
                        if d < d_to_comm:
                            d_to_comm = d
                    
                    cost = d_to_sample + drop_cost + 1 + d_to_comm + 1 # +1 sample, +1 communicate
                    if cost < best_acq_cost:
                        best_acq_cost = cost
                        
                if best_acq_cost != self.INF:
                    total_cost += best_acq_cost
                else:
                    total_cost += 50 # Penalty

            # --- C. Image Data ---
            elif pred == "get_image_data":
                # args: [objective, mode]
                obj = args[0]
                mode = args[1]
                
                # 1. Check goal
                gid_comm = self.fact_ids['communicated_image_data'].get((obj, mode))
                if gid_comm is not None and ((state >> gid_comm) & 1):
                    continue
                
                # 2. Check held
                held_by = None
                for gid, f_args in self.fact_list['have_image']:
                    if f_args[1] == obj and f_args[2] == mode:
                        if (state >> gid) & 1:
                            held_by = f_args[0]
                            break
                
                if held_by:
                    curr = get_r_loc(held_by)
                    if curr:
                        min_comm_dist = self.INF
                        for comm_wp in self.comm_waypoints:
                            d = self.dist[held_by][curr][comm_wp]
                            if d < min_comm_dist:
                                min_comm_dist = d
                        if min_comm_dist != self.INF:
                            total_cost += (min_comm_dist + 1)
                    continue
                
                # 3. Acquire
                best_img_cost = self.INF
                # Candidates: equipped for imaging AND have camera supporting mode
                candidates = [r for r in self.equipped_for_imaging if 
                              any(mode in self.supports[c] for c in self.on_board[r])]
                
                valid_shoot_wps = list(self.visible_from[obj])
                if not valid_shoot_wps: 
                    total_cost += 100
                    continue

                for r in candidates:
                    curr = get_r_loc(r)
                    if not curr: continue
                    
                    # Find best camera on this rover
                    valid_cams = [c for c in self.on_board[r] if mode in self.supports[c]]
                    
                    rover_best = self.INF
                    
                    for cam in valid_cams:
                        # Calibration check
                        calibrated_gid = self.fact_ids['calibrated'].get((cam, r))
                        is_calibrated = (calibrated_gid is not None and ((state >> calibrated_gid) & 1))
                        
                        calib_cost = 0
                        start_node_for_shoot = curr
                        
                        if not is_calibrated:
                            calib_cost = 1 # action
                            # Must go to calib target visibility
                            target = self.calibration_targets.get(cam)
                            if target:
                                calib_wps = self.visible_from[target]
                                # Dist curr -> calib_wp
                                min_d_calib = self.INF
                                best_calib_wp = None
                                for cw in calib_wps:
                                    d = self.dist[r][curr][cw]
                                    if d < min_d_calib:
                                        min_d_calib = d
                                        best_calib_wp = cw
                                
                                if min_d_calib != self.INF:
                                    calib_cost += min_d_calib
                                    start_node_for_shoot = best_calib_wp # Now we are here
                                else:
                                    calib_cost = 100 # Can't calibrate
                        
                        # Shoot cost: nav from start_node_for_shoot to visible_from(obj)
                        min_d_shoot = self.INF
                        best_shoot_wp = None
                        for sw in valid_shoot_wps:
                            d = self.dist[r][start_node_for_shoot][sw]
                            if d < min_d_shoot:
                                min_d_shoot = d
                                best_shoot_wp = sw
                        
                        # Send cost: from best_shoot_wp to comm
                        min_d_comm = self.INF
                        if best_shoot_wp:
                            for cwp in self.comm_waypoints:
                                d = self.dist[r][best_shoot_wp][cwp]
                                if d < min_d_comm:
                                    min_d_comm = d
                        
                        if min_d_shoot != self.INF and min_d_comm != self.INF:
                            # Total = calib_nav + calib_act + shoot_nav + shoot_act + comm_nav + comm_act
                            cost = calib_cost + min_d_shoot + 1 + min_d_comm + 1
                            if cost < rover_best:
                                rover_best = cost
                    
                    if rover_best < best_img_cost:
                        best_img_cost = rover_best
                
                if best_img_cost != self.INF:
                    total_cost += best_img_cost
                else:
                    total_cost += 50

            # --- D. Send Data Tasks ---
            elif pred.startswith("send_"):
                # Abstract tasks: send_soil_data, send_rock_data, send_image_data
                # Assume data is held (precondition of send methods usually implies data is ready or being fetched)
                # But if called standalone, we just assume rover needs to go to comms.
                # However, the rover might not be specified in args depending on decomposition level.
                # args: [rover, ...]
                if len(args) > 0:
                    r = args[0]
                    curr = get_r_loc(r)
                    if curr:
                        min_d = self.INF
                        for cwp in self.comm_waypoints:
                            d = self.dist[r][curr][cwp]
                            if d < min_d:
                                min_d = d
                        if min_d != self.INF:
                            total_cost += (min_d + 1)
                else:
                    # Generic send
                    total_cost += 1
            
            # --- E. Primitives --- 
            # (navigate handled above)
            elif pred in ["sample_soil", "sample_rock", "take_image", "communicate_soil_data", 
                          "communicate_rock_data", "communicate_image_data", "calibrate", "drop"]:
                total_cost += 1
            
            elif pred == "nop":
                pass
                
        return total_cost
