import sys
from typing import List, Dict, Set, Tuple, Optional
from collections import deque, defaultdict
import heapq

# Pytrich classes are usually available in the environment when the heuristic is loaded.
# We inherit from the base Heuristic class provided by the framework.
# If not explicitly imported, we assume Heuristic is in the global namespace or imported by the loader.
# For safety in this standalone snippet, we can try to import or define a dummy if testing offline.

from Pytrich.Heuristics.heuristic import Heuristic
class RoverGemini3ProHeuristic16(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Rover domain.
    
    This heuristic estimates the cost to complete the current task network by:
    1. Parsing the domain connectivity graph (waypoints and traversal).
    2. Mapping rover capabilities (soil, rock, imaging, calibration).
    3. Pre-computing All-Pairs Shortest Paths (APSP) for waypoints.
    4. Estimating the cost of high-level abstract tasks (get_soil_data, get_image_data, etc.)
       based on the closest capable rover's distance to targets and required sub-actions.
    
    It handles the hierarchical nature by summing estimates for tasks in the task network,
    and handles state-dependency by checking current rover locations and calibration status.
    """

    def __init__(self, model, rover_gemini3pro_heuristic_15=None):
        super().__init__(model)
        self.rover_gemini3pro_heuristic_15 = rover_gemini3pro_heuristic_15
        
        # Domain knowledge structures
        self.waypoints = set()
        self.graph = defaultdict(set)  # u -> {v}
        self.distances = defaultdict(lambda: float('inf'))  # (u, v) -> int
        
        self.rovers = set()
        self.rover_capabilities = defaultdict(set) # rover -> {'soil', 'rock', 'imaging'}
        self.rover_cameras = defaultdict(list)     # rover -> [camera_name]
        self.rover_stores = {}                     # rover -> store_name
        
        self.camera_modes = defaultdict(set)       # camera -> {modes}
        self.camera_targets = {}                   # camera -> objective
        
        self.visible_from = defaultdict(set)       # objective -> {waypoints}
        self.visible_points = defaultdict(set)     # wp -> {wp} (for communication)
        
        # Fact ID lookups for fast state checking
        self.at_facts = defaultdict(dict)          # rover -> {wp: global_id}
        self.calibrated_facts = {}                 # (cam, rover) -> global_id
        self.empty_store_facts = {}                # store -> global_id
        self.full_store_facts = {}                 # store -> global_id
        
        # Initialization
        self._parse_domain_facts()
        self._compute_all_pairs_shortest_paths()

    def _parse_fact_name(self, name: str):
        """Parses facts like +at[rover0,waypoint1] or -can_traverse[r,x,y]."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        if "[" in name and name.endswith("]"):
            pred_end = name.find("[")
            predicate = name[:pred_end]
            args_str = name[pred_end+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return name, []

    def _parse_task_name(self, name: str):
        """Parses tasks like get_soil_data(waypoint0)."""
        name = name.strip()
        if "(" in name and name.endswith(")"):
            pred_end = name.find("(")
            task_name = name[:pred_end]
            args_str = name[pred_end+1:-1]
            args = args_str.split(",") if args_str else []
            return task_name, [a.strip() for a in args]
        return name, []

    def _parse_domain_facts(self):
        """Iterates over static facts to build domain maps."""
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            fid = fact.global_id
            
            if pred == "can_traverse":
                # can_traverse[rover, from, to]
                # Assuming map is rover-independent or union of all
                if len(args) == 3:
                    u, v = args[1], args[2]
                    self.waypoints.add(u)
                    self.waypoints.add(v)
                    self.graph[u].add(v)
                    self.graph[v].add(u)
            
            elif pred == "visible":
                if len(args) == 2:
                    self.visible_points[args[0]].add(args[1])
                    self.visible_points[args[1]].add(args[0])

            elif pred == "at":
                if len(args) == 2:
                    rover, wp = args
                    self.rovers.add(rover)
                    self.at_facts[rover][wp] = fid
                    self.waypoints.add(wp)

            elif pred == "equipped_for_soil_analysis":
                self.rover_capabilities[args[0]].add("soil")
            elif pred == "equipped_for_rock_analysis":
                self.rover_capabilities[args[0]].add("rock")
            elif pred == "equipped_for_imaging":
                self.rover_capabilities[args[0]].add("imaging")
            
            elif pred == "store_of":
                self.rover_stores[args[1]] = args[0]
            
            elif pred == "empty":
                self.empty_store_facts[args[0]] = fid
            elif pred == "full":
                self.full_store_facts[args[0]] = fid
                
            elif pred == "on_board":
                # on_board[camera, rover]
                self.rover_cameras[args[1]].append(args[0])
                
            elif pred == "calibration_target":
                self.camera_targets[args[0]] = args[1]
                
            elif pred == "supports":
                self.camera_modes[args[0]].add(args[1])
                
            elif pred == "visible_from":
                # visible_from[objective, waypoint]
                self.visible_from[args[0]].add(args[1])
                
            elif pred == "calibrated":
                # calibrated[camera, rover]
                self.calibrated_facts[(args[0], args[1])] = fid

    def _compute_all_pairs_shortest_paths(self):
        """Floyd-Warshall or BFS for unweighted graph to get distances."""
        # Initialize with 0 for self, 1 for neighbors
        for w in self.waypoints:
            self.distances[(w, w)] = 0
            for neighbor in self.graph[w]:
                self.distances[(w, neighbor)] = 1
        
        # Initialize others to inf (defaultdict does this)
        
        # Floyd-Warshall is O(V^3). For p30 V=130, V^3 ~ 2e6, feasible.
        # BFS from each node is O(V*(V+E)). Since graph is sparse, likely faster.
        for start_node in self.waypoints:
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                curr, dist = q.popleft()
                self.distances[(start_node, curr)] = dist
                for neighbor in self.graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

    def _get_current_rover_locations(self, state: int) -> Dict[str, str]:
        """Returns map of rover -> current_waypoint based on state bitmask."""
        locs = {}
        for rover, wp_map in self.at_facts.items():
            for wp, fid in wp_map.items():
                if (state >> fid) & 1:
                    locs[rover] = wp
                    break
        return locs

    def compute(self, node) -> int:
        """Estimates the cost for the current task network."""
        total_cost = 0
        state = node.state
        current_locs = self._get_current_rover_locations(state)
        
        # Process each task in the network
        for task in node.task_network:
            if hasattr(task, 'cost'): # Operator
                total_cost += task.cost
            else: # AbstractTask
                # The task might be a decomposition or abstract task
                # We parse its name to identify what it is
                t_name, t_args = self._parse_task_name(task.name)
                total_cost += self._estimate_abstract_task(t_name, t_args, state, current_locs)
        
        return total_cost

    def _estimate_abstract_task(self, name: str, args: List[str], state: int, current_locs: Dict[str, str]) -> int:
        """Estimates cost for specific high-level tasks."""
        
        # --- Navigation ---
        if name == "do_navigate1":
            # do_navigate1(rover, to)
            if len(args) >= 2:
                rover, to = args[0], args[1]
                start = current_locs.get(rover)
                if start:
                    return self.distances[(start, to)]
            return 1 # Fallback
            
        elif name == "do_navigate2":
            # do_navigate2(rover, from, to)
            if len(args) >= 3:
                _, start, to = args[0], args[1], args[2]
                return self.distances[(start, to)]
            return 1

        # --- Data Gathering ---
        elif name == "get_soil_data" or name == "get_rock_data":
            # get_soil_data(waypoint)
            if len(args) < 1:
                return 0
            target_wp = args[0]
            req_type = "soil" if "soil" in name else "rock"
            
            # Find minimum cost among capable rovers
            min_cost = float('inf')
            for rover in self.rovers:
                if req_type in self.rover_capabilities[rover]:
                    start = current_locs.get(rover)
                    if start:
                        dist = self.distances[(start, target_wp)]
                        if dist == float('inf'):
                            continue
                        
                        cost = dist + 1 # +1 for sample
                        
                        # Check store status if possible
                        store = self.rover_stores.get(rover)
                        if store:
                            # If full, add drop cost. 
                            # Not strictly necessary for admissibility but good for accuracy.
                            full_id = self.full_store_facts.get(store)
                            if full_id is not None and (state >> full_id) & 1:
                                cost += 1 # Drop action
                        
                        # Communication cost (simplified: assume 1 action to send)
                        # Optimistic: maybe already visible to lander? Let's add 1 for 'send' action.
                        cost += 1
                        
                        if cost < min_cost:
                            min_cost = cost
                            
            return min_cost if min_cost != float('inf') else 100

        elif name == "get_image_data":
            # get_image_data(objective, mode)
            if len(args) < 2:
                return 0
            objective, mode = args[0], args[1]
            
            min_cost = float('inf')
            
            # Iterate all rovers that handle imaging
            for rover in self.rovers:
                if "imaging" not in self.rover_capabilities[rover]:
                    continue
                
                start = current_locs.get(rover)
                if not start:
                    continue
                
                # Check cameras on this rover
                for cam in self.rover_cameras[rover]:
                    if mode in self.camera_modes[cam]:
                        # Found a capable camera/rover pair
                        # 1. Calibration
                        calib_cost = 0
                        # Check if already calibrated
                        is_calibrated = False
                        calib_fid = self.calibrated_facts.get((cam, rover))
                        if calib_fid is not None and (state >> calib_fid) & 1:
                            is_calibrated = True
                        
                        if not is_calibrated:
                            target = self.camera_targets.get(cam)
                            if target:
                                # Find closest visibility point for calibration target
                                min_dist_calib = float('inf')
                                # If target is visible from current loc? (Need visible_from map)
                                # We have visible_from[obj] -> {wps}
                                potential_calib_wps = self.visible_from[target]
                                
                                for c_wp in potential_calib_wps:
                                    d = self.distances[(start, c_wp)]
                                    if d < min_dist_calib:
                                        min_dist_calib = d
                                
                                if min_dist_calib != float('inf'):
                                    calib_cost = min_dist_calib + 1 # navigate + calibrate
                                else:
                                    calib_cost = 100 # Unreachable
                            else:
                                calib_cost = 1 # Fallback
                        
                        # 2. Imaging
                        # Cost from calibration point (or current if calibrated) to objective visibility
                        imaging_cost = float('inf')
                        potential_img_wps = self.visible_from[objective]
                        
                        # If we had to calibrate, we are at one of the potential_calib_wps (the closest one)
                        # But to keep it simple and admissible, assume we start from 'start' + calib_dist
                        # Actually, better: if calibrated, start from 'start'. If not, we move to 'calib_wp' then to 'img_wp'.
                        
                        if not potential_img_wps:
                            continue
                            
                        if is_calibrated:
                            # Just move to viewing point
                            for i_wp in potential_img_wps:
                                d = self.distances[(start, i_wp)]
                                if d < imaging_cost:
                                    imaging_cost = d
                        else:
                            # Move start -> calib -> img
                            # This is O(N^2) on waypoints in worst case, might be slow.
                            # Simplified: Cost = calib_cost (which includes move to calib) + min_dist(calib_locs, img_locs)
                            # We already found min_dist_calib. Let's assume we are at that optimal calibration point.
                            # This is an approximation.
                            pass
                            # To be safe and fast: 
                            # Cost = calib_cost + min(dist(any_calib_wp, any_img_wp))
                            # This requires iteration. Let's do a slightly looser bound:
                            # Cost >= calib_cost + 0 (if overlap) + 1 (take image)
                            # Let's just calculate: start -> img_wp + 1 (take) + 1 (calibrate_overhead)
                            # The overhead is the deviation. 
                            # Let's stick to: cost = calib_cost + distance(best_calib_wp, best_img_wp) + 1
                            # Too expensive to calculate pairwise. 
                            # Fallback: calib_cost + 1 (take image) + 1 (send)
                            imaging_cost = 0 # Placeholder if we used complex logic
                            
                            # Faster admissible estimate:
                            # If not calibrated: must visit calib_wp, then img_wp.
                            # If calibrated: just visit img_wp.
                            
                            best_path = float('inf')
                            # We iterate image waypoints
                            for i_wp in potential_img_wps:
                                if is_calibrated:
                                    d = self.distances[(start, i_wp)]
                                else:
                                    # start -> closest_calib -> i_wp
                                    # We approximate closest_calib as the one found earlier
                                    # or just dist(start, i_wp) + 1 (minimal detour) ? No, unsafe.
                                    # Let's assume we go start->i_wp and add minimal calib cost separately.
                                    d = self.distances[(start, i_wp)]
                                
                                if d < best_path:
                                    best_path = d
                            
                            imaging_cost = best_path

                        total_rover_cost = calib_cost + imaging_cost + 1 + 1 # +1 take, +1 send
                        if total_rover_cost < min_cost:
                            min_cost = total_rover_cost
                            
            return min_cost if min_cost != float('inf') else 100

        # --- Support Tasks ---
        elif name == "do_calibrate":
            if len(args) >= 2:
                rover, cam = args[0], args[1]
                target = self.camera_targets.get(cam)
                start = current_locs.get(rover)
                if target and start:
                    wps = self.visible_from[target]
                    min_dist = float('inf')
                    for w in wps:
                        d = self.distances[(start, w)]
                        if d < min_dist:
                            min_dist = d
                    if min_dist != float('inf'):
                        return min_dist + 1
            return 1
            
        elif name == "empty_store":
            return 1 # drop action
            
        elif name.startswith("send_"):
            return 1 # communicate action

        # Default for unknown tasks
        return 0
