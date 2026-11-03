import sys
import math
from collections import deque
from typing import List, Dict, Set, Tuple, Union

# Try importing the base class
from Pytrich.Heuristics.heuristic import Heuristic
class RoverGemini3ProHeuristic14(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Rover domain.
    
    Rationale:
    1. **Static Analysis**: The connectivity graph ('can_traverse') and visibility graph ('visible') 
       are static. We precompute All-Pairs Shortest Paths (APSP) to instantly retrieve navigation costs.
    2. **Task Estimation**: 
       - For navigation tasks, we use the precomputed distances.
       - For data gathering (soil/rock/image), we approximate the cost as:
         Travel to target + (Drop if full) + Sample/Calibrate/Image + Travel to Comm Point + Communicate.
       - Since 'get_data' tasks don't specify the rover, we compute the minimum cost across all 
         capable rovers (equipped and available).
    3. **State Decoding**: 
       - We efficiently decode the current state (rover locations, store status) using bitwise checks 
         against pre-mapped global IDs, avoiding slow string parsing during search.
    """

    def __init__(self, rover_gemini3pro_heuristic_13):
        super().__init__(rover_gemini3pro_heuristic_13)
        self.model = rover_gemini3pro_heuristic_13
        
        # --- Domain Objects ---
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.cameras = set()
        self.stores = set()
        self.modes = set()
        
        # --- Static Graphs & Maps ---
        # traverse_graph: rover -> u -> [v]
        self.traverse_graph: Dict[str, Dict[str, List[str]]] = {}
        # visible_graph: u -> [v]
        self.visible_graph: Dict[str, List[str]] = {}
        # visible_from: objective -> [waypoints]
        self.visible_from: Dict[str, List[str]] = {}
        # calibration_target: camera -> objective
        self.calibration_target: Dict[str, str] = {}
        
        # --- Capabilities ---
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.store_of = {} # rover -> store
        self.camera_on = {} # rover -> list[camera]
        self.camera_supports = {} # camera -> set(modes)
        
        # --- Precomputed Data ---
        self.dist_matrix = {} # rover -> u -> v -> int
        self.comm_points = set() # waypoints visible to a lander
        
        # --- Dynamic Fact Lookup ---
        # rover -> list of (waypoint, global_id)
        self.rover_at_facts = {}
        # store -> (global_id_empty, global_id_full)
        self.store_facts = {}
        # (rover, camera) -> global_id_calibrated
        self.calibrated_facts = {}
        
        self._initialize()

    def _parse_fact_name(self, name: str) -> Tuple[str, List[str]]:
        """Parses +predicate[arg1,arg2] into predicate, [args]."""
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

    def _parse_task_name(self, name: str) -> Tuple[str, List[str]]:
        """Parses taskname[arg1,arg2] into taskname, [args]."""
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            taskname = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return taskname, args
        return name, []

    def _initialize(self):
        """Parses static facts and precomputes distances."""
        
        # 1. Parse Facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == "at":
                # Format: at[rover, waypoint]
                # We collect these for dynamic lookup later
                r, w = args
                self.rovers.add(r)
                self.waypoints.add(w)
                if r not in self.rover_at_facts:
                    self.rover_at_facts[r] = []
                self.rover_at_facts[r].append((w, fact.global_id))
                
            elif pred == "can_traverse":
                # can_traverse[rover, from, to]
                r, f, t = args
                if r not in self.traverse_graph:
                    self.traverse_graph[r] = {}
                if f not in self.traverse_graph[r]:
                    self.traverse_graph[r][f] = []
                self.traverse_graph[r][f].append(t)
                
            elif pred == "visible":
                # visible[p1, p2]
                p1, p2 = args
                if p1 not in self.visible_graph:
                    self.visible_graph[p1] = []
                self.visible_graph[p1].append(p2)
                
            elif pred == "visible_from":
                # visible_from[objective, waypoint]
                o, w = args
                self.objectives.add(o)
                if o not in self.visible_from:
                    self.visible_from[o] = []
                self.visible_from[o].append(w)
                
            elif pred == "at_lander":
                # at_lander[lander, waypoint]
                l, w = args
                # A point is a comm point if it is visible to a lander location
                # We will process this after reading all 'visible' facts
                self.comm_points.add(w) # Usually visible to self, but handled later
                
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])
                
            elif pred == "store_of":
                s, r = args
                self.stores.add(s)
                self.store_of[r] = s
            
            elif pred == "empty":
                s = args[0]
                if s not in self.store_facts:
                    self.store_facts[s] = [-1, -1]
                self.store_facts[s][0] = fact.global_id
                
            elif pred == "full":
                s = args[0]
                if s not in self.store_facts:
                    self.store_facts[s] = [-1, -1]
                self.store_facts[s][1] = fact.global_id
                
            elif pred == "on_board":
                c, r = args
                self.cameras.add(c)
                if r not in self.camera_on:
                    self.camera_on[r] = []
                self.camera_on[r].append(c)
                
            elif pred == "calibration_target":
                c, o = args
                self.calibration_target[c] = o
                
            elif pred == "supports":
                c, m = args
                self.modes.add(m)
                if c not in self.camera_supports:
                    self.camera_supports[c] = set()
                self.camera_supports[c].add(m)
                
            elif pred == "calibrated":
                c, r = args
                self.calibrated_facts[(r, c)] = fact.global_id

        # 2. Expand Comm Points
        # comm_points currently contains lander locations.
        # Add all points visible to these locations.
        lander_locs = list(self.comm_points)
        for l_loc in lander_locs:
            # If visible graph is symmetric, this works.
            # Check edges starting from l_loc
            if l_loc in self.visible_graph:
                for v in self.visible_graph[l_loc]:
                    self.comm_points.add(v)
            # Also check edges ending at l_loc (in case visible is parsed only one way)
            # But undirected usually generated symmetric facts.
            # Assuming visible(p1, p2) means p1 can see p2.
            
        # 3. Compute All-Pairs Shortest Paths (BFS) for each rover
        # Since connectivity depends on the rover (can_traverse ?x ...)
        for r in self.rovers:
            self.dist_matrix[r] = {}
            if r not in self.traverse_graph:
                continue
            
            graph = self.traverse_graph[r]
            all_nodes = set(graph.keys())
            for u in graph:
                all_nodes.update(graph[u])
            
            for start_node in all_nodes:
                self.dist_matrix[r][start_node] = {}
                q = deque([(start_node, 0)])
                visited = {start_node}
                self.dist_matrix[r][start_node][start_node] = 0
                
                while q:
                    curr, dist = q.popleft()
                    if curr in graph:
                        for neighbor in graph[curr]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                self.dist_matrix[r][start_node][neighbor] = dist + 1
                                q.append((neighbor, dist + 1))

    def get_dist(self, rover, u, v):
        """Get precomputed distance."""
        if u == v: return 0
        if rover in self.dist_matrix and u in self.dist_matrix[rover] and v in self.dist_matrix[rover][u]:
            return self.dist_matrix[rover][u][v]
        return 9999 # Unreachable

    def get_nearest_comm_cost(self, rover, loc):
        """Cost to reach nearest communication point."""
        min_dist = 9999
        for comm in self.comm_points:
            d = self.get_dist(rover, loc, comm)
            if d < min_dist:
                min_dist = d
        return min_dist

    def h(self, state, task_network):
        if not task_network:
            return 0
            
        # 1. Decode Dynamic State
        rover_locs = {}
        for r in self.rovers:
            if r in self.rover_at_facts:
                for wp, fid in self.rover_at_facts[r]:
                    if (state >> fid) & 1:
                        rover_locs[r] = wp
                        break
        
        # We assume if a rover location isn't found, it's not in the state (unlikely) 
        # or we treat it as unknown.
        
        total_cost = 0
        
        for task in task_network:
            # If it's an operator (primitive task), cost is 1
            # Pytrich passes AbstractTask or Operator objects
            # We check if it has a 'cost' attribute or check type name
            
            # Assuming standard structure where Operators have 'cost' and AbstractTasks don't (or we decompose)
            # In HTN heuristics, we usually estimate abstract tasks.
            
            if hasattr(task, 'cost') and hasattr(task, 'add_effects'): 
                # It is an operator
                total_cost += getattr(task, 'cost', 1)
                continue
                
            # It is an abstract task
            # Parse name: e.g. "get_soil_data[waypoint0]"
            t_name, t_args = self._parse_task_name(task.name)
            
            if t_name == "do_navigate1":
                # do_navigate1[rover, to]
                if len(t_args) >= 2:
                    r, to = t_args[0], t_args[1]
                    if r in rover_locs:
                        total_cost += self.get_dist(r, rover_locs[r], to)
                    else:
                        total_cost += 1 # Unknown loc penalty
                        
            elif t_name == "do_navigate2":
                # do_navigate2[rover, from, to]
                if len(t_args) >= 3:
                    r, frm, to = t_args[0], t_args[1], t_args[2]
                    total_cost += self.get_dist(r, frm, to)
            
            elif t_name == "empty_store":
                # empty_store[store, rover]
                # Check if full. If full, cost 1 (drop). If empty, cost 0.
                if len(t_args) >= 2:
                    s = t_args[0]
                    if s in self.store_facts:
                        is_full_id = self.store_facts[s][1]
                        if is_full_id != -1 and ((state >> is_full_id) & 1):
                            total_cost += 1
                            
            elif t_name == "get_soil_data" or t_name == "get_rock_data":
                # get_soil_data[waypoint] / get_rock_data[waypoint]
                # Needs: Nav to wp + (Drop) + Sample + Nav to comm + Comm
                target_wp = t_args[0] if t_args else None
                is_soil = (t_name == "get_soil_data")
                candidates = self.equipped_soil if is_soil else self.equipped_rock
                
                best_r_cost = 9999
                
                for r in candidates:
                    if r not in rover_locs: continue
                    
                    # Travel to sample
                    d_to_target = self.get_dist(r, rover_locs[r], target_wp)
                    
                    # Check store full
                    drop_cost = 0
                    if r in self.store_of:
                        s = self.store_of[r]
                        if s in self.store_facts:
                            is_full_id = self.store_facts[s][1]
                            if is_full_id != -1 and ((state >> is_full_id) & 1):
                                drop_cost = 1
                    
                    # Travel to comm
                    # Assuming rover samples at target_wp, it is now at target_wp
                    d_to_comm = self.get_nearest_comm_cost(r, target_wp)
                    
                    # Total: Travel + Drop + Sample(1) + TravelComm + Comm(1)
                    c = d_to_target + drop_cost + 1 + d_to_comm + 1
                    if c < best_r_cost:
                        best_r_cost = c
                
                total_cost += (best_r_cost if best_r_cost < 9999 else 5)

            elif t_name == "get_image_data":
                # get_image_data[objective, mode]
                if len(t_args) >= 2:
                    obj, mode = t_args[0], t_args[1]
                    best_r_cost = 9999
                    
                    # Find rovers with camera supporting mode
                    for r in self.equipped_imaging:
                        if r not in rover_locs: continue
                        if r not in self.camera_on: continue
                        
                        # Find valid camera on this rover
                        valid_cam = None
                        for cam in self.camera_on[r]:
                            if cam in self.camera_supports and mode in self.camera_supports[cam]:
                                valid_cam = cam
                                break
                        if not valid_cam: continue
                        
                        # 1. Calibration
                        calib_cost = 0
                        calib_target = self.calibration_target.get(valid_cam)
                        
                        # Check if already calibrated
                        is_calibrated = False
                        if (r, valid_cam) in self.calibrated_facts:
                            fid = self.calibrated_facts[(r, valid_cam)]
                            if (state >> fid) & 1:
                                is_calibrated = True
                        
                        curr_loc = rover_locs[r]
                        
                        if not is_calibrated and calib_target:
                            # Need to go see calibration target
                            # Find closest point to see calib_target
                            view_pts = self.visible_from.get(calib_target, [])
                            if not view_pts: 
                                calib_cost = 9999
                            else:
                                d_calib = min([self.get_dist(r, curr_loc, v) for v in view_pts])
                                calib_cost = d_calib + 1 # navigate + calibrate
                                # Approximating that rover ends up at a view point
                                # We pick the "closest" view point as new location? 
                                # For heuristic, assume we pay the travel cost but don't strictly update location for next steps
                                # or assume worst case / best case.
                        
                        # 2. Imaging
                        # Need to see objective
                        img_cost = 9999
                        obj_view_pts = self.visible_from.get(obj, [])
                        if obj_view_pts:
                            # Distance from current loc (or calib loc) to obj view
                            # If calibrated, we travelled. Let's simplistically add distances.
                            # d(start, calib_view) + d(calib_view, obj_view) is better but complex.
                            # We use: if needs calib, d_calib + d_img. 
                            # Simpler: d(start, obj_view) + 1. If calib needed, add penalty.
                            
                            d_img = min([self.get_dist(r, curr_loc, v) for v in obj_view_pts])
                            img_cost = d_img + 1 # navigate + take_image
                        
                        # 3. Comm
                        # From image location (unknown exact one), travel to comm.
                        # Approximation: min_dist(obj_view_pt, comm_pt)
                        comm_cost = 9999
                        if obj_view_pts:
                             # optimistic: picked the view point closest to a comm point
                             comm_cost = min([self.get_nearest_comm_cost(r, v) for v in obj_view_pts])
                             comm_cost += 1 # communicate
                        
                        total_r = calib_cost + img_cost + comm_cost
                        if total_r < best_r_cost:
                            best_r_cost = total_r
                            
                    total_cost += (best_r_cost if best_r_cost < 9999 else 5)

            elif t_name.startswith("send_") and t_name.endswith("_data"):
                # send_soil_data[rover, from] etc.
                # Just nav to comm + comm
                if len(t_args) >= 2:
                    r, loc = t_args[0], t_args[1]
                    if r in rover_locs:
                        total_cost += self.get_nearest_comm_cost(r, rover_locs[r]) + 1
                    else:
                        total_cost += 2
                        
            elif t_name == "do_calibrate":
                # do_calibrate[rover, camera]
                if len(t_args) >= 2:
                    r, cam = t_args[0], t_args[1]
                    calib_target = self.calibration_target.get(cam)
                    view_pts = self.visible_from.get(calib_target, [])
                    if r in rover_locs and view_pts:
                        d = min([self.get_dist(r, rover_locs[r], v) for v in view_pts])
                        total_cost += d + 1
                    else:
                        total_cost += 1
            
            else:
                # Default for unknown tasks or nop
                if t_name != "nop":
                    total_cost += 1

        return total_cost
