import sys
import re
import collections
from typing import List, Set, Dict, Tuple, Optional

# Ensure these classes are available in the context of the planner
# from heuristic import Heuristic

class RoverGemini3ProHeuristic13(Heuristic):
    def __init__(self, model, rover_gemini3pro_heuristic_12=None):
        super().__init__(model)
        self.model = model
        self.INF = 99999

        # --- Precomputation Storage ---
        self.rovers: List[str] = []
        self.waypoints: List[str] = []
        self.objectives: List[str] = []
        self.cameras: List[str] = []
        self.stores: List[str] = []
        self.modes: List[str] = []
        self.landers: List[str] = []

        # Maps for static domain info
        self.rover_connectivity: Dict[str, Dict[str, List[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
        self.distances: Dict[str, Dict[str, Dict[str, int]]] = {}  # rover -> u -> v -> dist
        
        self.equipped_soil: Set[str] = set()
        self.equipped_rock: Set[str] = set()
        self.equipped_imaging: Set[str] = set()
        
        # rover -> list of stores
        self.rover_stores: Dict[str, List[str]] = collections.defaultdict(list)
        
        # camera -> obj
        self.calibration_targets: Dict[str, str] = {}
        # camera -> set of modes
        self.camera_supports: Dict[str, Set[str]] = collections.defaultdict(set)
        # rover -> list of cameras
        self.rover_cameras: Dict[str, List[str]] = collections.defaultdict(list)
        
        # objective -> set of visible waypoints
        self.visible_from: Dict[str, Set[str]] = collections.defaultdict(set)
        
        # Waypoints from which communication is possible (visible to a lander)
        self.comm_waypoints: Set[str] = set()

        # Global IDs for fast state checking
        self.fact_ids: Dict[str, int] = {}
        self.at_ids: Dict[Tuple[str, str], int] = {} # (rover, wp) -> id
        self.empty_ids: Dict[str, int] = {} # store -> id
        self.full_ids: Dict[str, int] = {} # store -> id
        self.calibrated_ids: Dict[Tuple[str, str], int] = {} # (cam, rover) -> id
        self.have_soil_ids: Dict[Tuple[str, str], int] = {} # (rover, wp) -> id
        self.have_rock_ids: Dict[Tuple[str, str], int] = {} # (rover, wp) -> id
        self.have_image_ids: Dict[Tuple[str, str, str], int] = {} # (rover, obj, mode) -> id
        self.communicated_soil_ids: Dict[str, int] = {} # wp -> id
        self.communicated_rock_ids: Dict[str, int] = {} # wp -> id
        self.communicated_image_ids: Dict[Tuple[str, str], int] = {} # (obj, mode) -> id

        self._initialize()

    def _parse_fact_name(self, name: str) -> Tuple[str, List[str]]:
        """Parses +predicate[arg1,arg2] into (predicate, [arg1, arg2])."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return name, []

    def _initialize(self):
        """Parses static facts and precomputes distances."""
        # 1. Parse all facts
        lander_locs = set()
        visible_pairs = set() # (p1, p2) means p1 visible to p2

        for fact in self.model.facts:
            self.fact_ids[fact.name] = fact.global_id
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == "can_traverse" and len(args) == 3:
                r, f, t = args
                self.rover_connectivity[r][f].append(t)
                if r not in self.rovers: self.rovers.append(r)
                if f not in self.waypoints: self.waypoints.append(f)
                if t not in self.waypoints: self.waypoints.append(t)
            
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])
            
            elif pred == "store_of":
                self.rover_stores[args[1]].append(args[0])
            
            elif pred == "on_board":
                self.rover_cameras[args[1]].append(args[0])
            elif pred == "calibration_target":
                self.calibration_targets[args[0]] = args[1]
            elif pred == "supports":
                self.camera_supports[args[0]].add(args[1])
            
            elif pred == "visible_from":
                self.visible_from[args[0]].add(args[1])
            
            elif pred == "at_lander":
                self.landers.append(args[0])
                lander_locs.add(args[1])
            elif pred == "visible":
                visible_pairs.add((args[0], args[1]))
                
            # Map dynamic IDs
            elif pred == "at": self.at_ids[tuple(args)] = fact.global_id
            elif pred == "empty": self.empty_ids[args[0]] = fact.global_id
            elif pred == "full": self.full_ids[args[0]] = fact.global_id
            elif pred == "calibrated": self.calibrated_ids[tuple(args)] = fact.global_id
            elif pred == "have_soil_analysis": self.have_soil_ids[tuple(args)] = fact.global_id
            elif pred == "have_rock_analysis": self.have_rock_ids[tuple(args)] = fact.global_id
            elif pred == "have_image": self.have_image_ids[tuple(args)] = fact.global_id
            elif pred == "communicated_soil_data": self.communicated_soil_ids[args[0]] = fact.global_id
            elif pred == "communicated_rock_data": self.communicated_rock_ids[args[0]] = fact.global_id
            elif pred == "communicated_image_data": self.communicated_image_ids[tuple(args)] = fact.global_id

        # 2. Determine Communication Waypoints
        # A waypoint is a comm point if it is visible to a lander location, or is the lander location
        self.comm_waypoints.update(lander_locs)
        for p1, p2 in visible_pairs:
            if p2 in lander_locs:
                self.comm_waypoints.add(p1)

        # 3. Precompute Distances (APSP) for each rover
        # Rovers might have distinct traversal graphs
        for rover in self.rovers:
            self.distances[rover] = {}
            adj = self.rover_connectivity[rover]
            
            # Collect all known waypoints for this rover (graph nodes)
            nodes = set(adj.keys())
            for neighbors in adj.values():
                nodes.update(neighbors)
            
            # Floyd-Warshall or BFS. Since graph is unweighted, BFS from each node is fine.
            # Given N ~ 100, N*BFS is OK.
            for start_node in nodes:
                dists = {node: self.INF for node in nodes}
                dists[start_node] = 0
                queue = collections.deque([start_node])
                
                while queue:
                    u = queue.popleft()
                    d_u = dists[u]
                    for v in adj[u]:
                        if dists[v] == self.INF:
                            dists[v] = d_u + 1
                            queue.append(v)
                self.distances[rover][start_node] = dists

    def _get_rover_loc(self, rover: str, state_bitwise: int) -> Optional[str]:
        """Finds current location of a rover from state bitmask."""
        # Optimization: We could cache which waypoints are relevant for each rover
        # But iterating at_ids for specific rover is reasonably fast if parsed well
        # Actually, iterating self.at_ids keys is slow. 
        # Better: check known waypoints for this rover.
        
        # Note: In the domain, rovers move. We iterate all waypoints? 
        # Heuristic optimization: check only valid waypoints for this rover
        if rover not in self.distances:
            return None # Should not happen if rover in static analysis
            
        known_wps = self.distances[rover].keys()
        for wp in known_wps:
            fid = self.at_ids.get((rover, wp))
            if fid is not None and ((state_bitwise >> fid) & 1):
                return wp
        return None

    def _is_true(self, global_id: int, state_bitwise: int) -> bool:
        return (state_bitwise >> global_id) & 1

    def compute(self, state_bitwise: int, task_network: List) -> int:
        total_cost = 0
        
        # We iterate over the task network.
        # Since HDDL is hierarchical, we define costs for high-level tasks
        # based on a relaxation (shortest path + essential actions).
        
        for task in task_network:
            # task is Operator or AbstractTask. Both have 'name'.
            pred, args = self._parse_fact_name(task.name)
            
            if pred.startswith("do_navigate") or pred == "navigate":
                # args: [rover, to] or [rover, from, to] ...
                # navigate primitive is [rover, from, to]
                # do_navigate1 is [rover, to]
                rover = args[0]
                target = args[-1] # Target is usually last
                
                current_loc = self._get_rover_loc(rover, state_bitwise)
                if current_loc and rover in self.distances and current_loc in self.distances[rover]:
                    dist = self.distances[rover][current_loc].get(target, self.INF)
                    if dist < self.INF:
                        total_cost += dist
                    else:
                        total_cost += 100 # Penalty for unreachable
                else:
                    total_cost += 1 # minimal assumption

            elif pred == "get_soil_data":
                wp = args[0]
                # Check if done
                comm_id = self.communicated_soil_ids.get(wp)
                if comm_id is not None and self._is_true(comm_id, state_bitwise):
                    continue

                # Relaxation: Find Best Rover
                # Cost = Move(r, wp) + Sample(1) + Move(r, Comm) + Comm(1)
                best_cost = self.INF
                
                # Check if any rover already has it
                has_data = False
                for r in self.rovers:
                    hid = self.have_soil_ids.get((r, wp))
                    if hid is not None and self._is_true(hid, state_bitwise):
                        # Just need to communicate
                        loc = self._get_rover_loc(r, state_bitwise)
                        dist_comm = self._min_dist_to_comm(r, loc)
                        best_cost = min(best_cost, dist_comm + 1)
                        has_data = True
                
                if not has_data:
                    # Need to fetch and sample
                    for r in self.equipped_soil:
                        loc = self._get_rover_loc(r, state_bitwise)
                        if not loc: continue
                        
                        # 1. Travel to sample
                        dist_to_sample = self._get_dist(r, loc, wp)
                        
                        # 2. Store check (add drop cost if full)
                        # Note: This is loose because we don't know which store.
                        # Just check if *any* store of this rover is empty. 
                        # If all full, add 1.
                        store_cost = 1 # Assume need to drop or minimal interaction
                        any_empty = False
                        for s in self.rover_stores[r]:
                            eid = self.empty_ids.get(s)
                            if eid is not None and self._is_true(eid, state_bitwise):
                                any_empty = True
                                break
                        if any_empty: store_cost = 0
                        else: store_cost = 1 # drop action

                        # 3. Travel to comm
                        # We need to go from wp (sample loc) to comm node
                        dist_to_comm = self._min_dist_to_comm(r, wp)
                        
                        cost = dist_to_sample + store_cost + 1 + dist_to_comm + 1
                        if cost < best_cost:
                            best_cost = cost
                
                total_cost += (best_cost if best_cost < self.INF else 100)

            elif pred == "get_rock_data":
                wp = args[0]
                comm_id = self.communicated_rock_ids.get(wp)
                if comm_id is not None and self._is_true(comm_id, state_bitwise):
                    continue

                best_cost = self.INF
                has_data = False
                for r in self.rovers:
                    hid = self.have_rock_ids.get((r, wp))
                    if hid is not None and self._is_true(hid, state_bitwise):
                        loc = self._get_rover_loc(r, state_bitwise)
                        best_cost = min(best_cost, self._min_dist_to_comm(r, loc) + 1)
                        has_data = True
                
                if not has_data:
                    for r in self.equipped_rock:
                        loc = self._get_rover_loc(r, state_bitwise)
                        if not loc: continue
                        
                        dist_to_sample = self._get_dist(r, loc, wp)
                        
                        store_cost = 1
                        any_empty = False
                        for s in self.rover_stores[r]:
                            eid = self.empty_ids.get(s)
                            if eid is not None and self._is_true(eid, state_bitwise):
                                any_empty = True
                                break
                        if any_empty: store_cost = 0
                        
                        dist_to_comm = self._min_dist_to_comm(r, wp)
                        cost = dist_to_sample + store_cost + 1 + dist_to_comm + 1
                        if cost < best_cost:
                            best_cost = cost

                total_cost += (best_cost if best_cost < self.INF else 100)

            elif pred == "get_image_data":
                obj = args[0]
                mode = args[1]
                
                comm_id = self.communicated_image_ids.get((obj, mode))
                if comm_id is not None and self._is_true(comm_id, state_bitwise):
                    continue

                best_cost = self.INF
                has_data = False
                
                # Check existing data
                for r in self.rovers:
                    hid = self.have_image_ids.get((r, obj, mode))
                    if hid is not None and self._is_true(hid, state_bitwise):
                        loc = self._get_rover_loc(r, state_bitwise)
                        best_cost = min(best_cost, self._min_dist_to_comm(r, loc) + 1)
                        has_data = True
                
                if not has_data:
                    # Find rovers with appropriate camera
                    # Need camera c on rover r, supports(c, mode)
                    # Steps: Calibrate (move to calib target) -> Move to Vis -> Take -> Move to Comm -> Send
                    
                    potential_view_points = self.visible_from[obj]
                    if not potential_view_points:
                        total_cost += 100
                        continue

                    for r in self.equipped_imaging:
                        loc = self._get_rover_loc(r, state_bitwise)
                        if not loc: continue
                        
                        valid_cams = []
                        for c in self.rover_cameras[r]:
                            if mode in self.camera_supports[c]:
                                valid_cams.append(c)
                        
                        if not valid_cams: continue
                        
                        for c in valid_cams:
                            calib_target = self.calibration_targets.get(c)
                            is_calibrated = False
                            cid = self.calibrated_ids.get((c, r))
                            if cid is not None and self._is_true(cid, state_bitwise):
                                is_calibrated = True
                            
                            # Cost to calibrate
                            calib_cost = 0
                            curr_r_virtual = loc
                            
                            if not is_calibrated and calib_target:
                                # Move to visibility of calib target
                                calib_views = self.visible_from.get(calib_target, set())
                                if not calib_views: 
                                    calib_cost = self.INF
                                else:
                                    # Dist to nearest calib view
                                    d_calib, nearest_calib_wp = self._min_dist_to_set(r, loc, calib_views)
                                    calib_cost = d_calib + 1 # +1 for calibrate action
                                    curr_r_virtual = nearest_calib_wp # Rover is now here

                            # Cost to take image (move from virtual pos to obj view)
                            d_img, nearest_img_wp = self._min_dist_to_set(r, curr_r_virtual, potential_view_points)
                            img_cost = d_img + 1 # take image
                            
                            # Cost to send
                            d_comm = self._min_dist_to_comm(r, nearest_img_wp)
                            send_cost = d_comm + 1
                            
                            total_seq = calib_cost + img_cost + send_cost
                            if total_seq < best_cost:
                                best_cost = total_seq
                
                total_cost += (best_cost if best_cost < self.INF else 100)

            elif pred == "send_soil_data" or pred == "send_rock_data":
                # This usually follows acquisition. args: [rover, from_wp]
                # If isolated, estimate comm cost.
                r = args[0]
                loc = self._get_rover_loc(r, state_bitwise)
                if loc:
                    d = self._min_dist_to_comm(r, loc)
                    total_cost += (d + 1 if d < self.INF else 50)
                else:
                    total_cost += 1

            elif pred == "send_image_data":
                # [rover, obj, mode]
                r = args[0]
                loc = self._get_rover_loc(r, state_bitwise)
                if loc:
                    d = self._min_dist_to_comm(r, loc)
                    total_cost += (d + 1 if d < self.INF else 50)
                else:
                    total_cost += 1
            
            elif pred == "empty_store":
                # [store, rover]
                s = args[0]
                eid = self.empty_ids.get(s)
                if eid is not None and not self._is_true(eid, state_bitwise):
                    total_cost += 1 # drop
            
            elif pred == "do_calibrate":
                # [rover, camera]
                r, c = args[0], args[1]
                cid = self.calibrated_ids.get((c, r))
                if cid is None or not self._is_true(cid, state_bitwise):
                    # Need to go to calib target
                    targ = self.calibration_targets.get(c)
                    if targ:
                        views = self.visible_from.get(targ, set())
                        loc = self._get_rover_loc(r, state_bitwise)
                        if loc:
                            d, _ = self._min_dist_to_set(r, loc, views)
                            total_cost += d + 1
                        else:
                            total_cost += 1
                    else:
                        total_cost += 1
            
            else:
                # Default for unknown/primitive ops
                total_cost += 1
                
        return total_cost

    def _get_dist(self, rover, u, v):
        if rover in self.distances and u in self.distances[rover]:
            return self.distances[rover][u].get(v, self.INF)
        return self.INF

    def _min_dist_to_comm(self, rover, start_wp):
        """Distance to nearest communication point."""
        if not self.comm_waypoints:
            return self.INF
        d, _ = self._min_dist_to_set(rover, start_wp, self.comm_waypoints)
        return d

    def _min_dist_to_set(self, rover, start_wp, targets: Set[str]) -> Tuple[int, str]:
        """Returns (min_dist, closest_target_wp)."""
        if rover not in self.distances or start_wp not in self.distances[rover]:
            return self.INF, None
        
        min_d = self.INF
        best_wp = None
        
        # Optimization: iterate targets if smaller than graph, or use precomputed structure
        # Here just direct check
        dists = self.distances[rover][start_wp]
        for t in targets:
            d = dists.get(t, self.INF)
            if d < min_d:
                min_d = d
                best_wp = t
        
        return min_d, best_wp
