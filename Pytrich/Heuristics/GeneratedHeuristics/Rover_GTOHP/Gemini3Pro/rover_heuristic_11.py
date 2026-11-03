from typing import List, Set, Dict, Tuple, Optional
import sys
import math
from collections import defaultdict, deque

from pytrich.heuristics.heuristic import Heuristic
from pytrich.model import Model, Fact, Operator, AbstractTask

class RoverGemini3ProHeuristic11(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain.
    
    Rationale:
    1. **Pre-computation**: We parse the static domain structure (connectivity graph, rover capabilities, 
       camera configurations, visibility constraints) once during initialization to avoid overhead during search.
       We compute All-Pairs Shortest Paths (APSP) for each rover's traverse capability.
    
    2. **Hierarchical Estimation**: The heuristic iterates through the current task network. For each task,
       it estimates the minimum remaining cost to complete it. 
       - For primitive operators, cost is 1.
       - For high-level tasks like `get_soil_data`, `get_rock_data`, and `get_image_data`, it calculates 
         a relaxed plan cost: (navigation to target + interaction cost + navigation to transmission spot + communication).
         It considers dynamic state (current rover position, calibration status, store status) to give a precise lower bound.
    
    3. **Resource Awareness**: It selects the "best" rover for a task (min cost) among those capable. 
       This acts as a relaxation (assuming rovers don't block each other) which is admissible for sum-cost metrics.
    
    4. **State Checking**: Uses efficient bitwise operations to check predicates in the current state.
    """

    def __init__(self, model: Model, rover_gemini3pro_heuristic_10=None):
        super().__init__(model)
        self.model = model
        self.inf = float('inf')
        
        # -- 1. Parse Static Domain Knowledge --
        self.rovers = set()
        self.waypoints = set()
        self.cameras = set()
        self.objectives = set()
        self.landers = set()
        self.modes = set()
        
        # Maps
        self.can_traverse = defaultdict(lambda: defaultdict(set)) # rover -> from -> set(to)
        self.visible = defaultdict(set) # wp -> set(visible_wp)
        self.lander_at = {} # lander -> wp
        self.at_soil = set() # set(wp)
        self.at_rock = set() # set(wp)
        self.visible_from = defaultdict(set) # objective -> set(wp)
        self.calibration_target = {} # camera -> objective
        self.on_board = {} # camera -> rover
        self.supports = defaultdict(set) # camera -> set(mode)
        self.equipped_soil = set() # set(rover)
        self.equipped_rock = set() # set(rover)
        self.equipped_imaging = set() # set(rover)
        
        # ID Mapping for dynamic state checks
        self.fact_name_to_id = {}
        
        self._parse_static_facts()
        
        # -- 2. Precompute Distances (APSP) --
        # self.dist[rover][u][v] = min_steps
        self.dist = defaultdict(lambda: defaultdict(lambda: self.inf))
        self._compute_distances()
        
        # Cache lander visibility for quick lookup
        # Set of waypoints from which *any* lander can be communicated with
        self.comm_waypoints = set()
        self._compute_comm_waypoints()

    def _parse_fact_name(self, name: str):
        """Parses +predicate[arg1,arg2] into (predicate, [arg1, arg2])"""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        if "[" in name and name.endswith("]"):
            idx = name.find("[")
            pred = name[:idx]
            args = name[idx+1:-1].split(",")
            return pred, args
        return name, []

    def _parse_static_facts(self):
        for fact in self.model.facts:
            self.fact_name_to_id[fact.name] = fact.global_id
            
            # We focus on positive facts for static structure
            if not fact.name.startswith("+"):
                continue
                
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == "can_traverse":
                r, f, t = args
                self.can_traverse[r][f].add(t)
                self.rovers.add(r)
                self.waypoints.add(f)
                self.waypoints.add(t)
            elif pred == "visible":
                p1, p2 = args
                self.visible[p1].add(p2)
                self.visible[p2].add(p1)
            elif pred == "at_lander":
                l, p = args
                self.lander_at[l] = p
                self.landers.add(l)
                self.waypoints.add(p)
            elif pred == "at_soil_sample":
                self.at_soil.add(args[0])
            elif pred == "at_rock_sample":
                self.at_rock.add(args[0])
            elif pred == "visible_from":
                o, p = args
                self.visible_from[o].add(p)
                self.objectives.add(o)
                self.waypoints.add(p)
            elif pred == "calibration_target":
                cam, obj = args
                self.calibration_target[cam] = obj
                self.cameras.add(cam)
            elif pred == "on_board":
                cam, r = args
                self.on_board[cam] = r
            elif pred == "supports":
                cam, m = args
                self.supports[cam].add(m)
                self.modes.add(m)
            elif pred == "equipped_for_soil_analysis":
                self.equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging":
                self.equipped_imaging.add(args[0])

    def _compute_distances(self):
        """Floyd-Warshall or BFS for each rover map"""
        for r in self.rovers:
            # Initialize graph for this rover
            adj = self.can_traverse[r]
            all_wps = list(self.waypoints)
            
            # BFS from each node to all others
            for start_node in all_wps:
                self.dist[r][start_node][start_node] = 0
                queue = deque([(start_node, 0)])
                visited = {start_node}
                
                while queue:
                    curr, d = queue.popleft()
                    self.dist[r][start_node][curr] = d
                    
                    for nxt in adj[curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, d + 1))

    def _compute_comm_waypoints(self):
        """Identify all waypoints that can communicate with a lander"""
        # A waypoint is a comm waypoint if it is at a lander OR visible to a lander's location
        for l, l_pos in self.lander_at.items():
            self.comm_waypoints.add(l_pos)
            for v in self.visible[l_pos]:
                self.comm_waypoints.add(v)

    def _is_true(self, state_bitwise, pred, args):
        """Helper to check if a grounded fact is true in the bitwise state"""
        fact_str = f"+{pred}[" + ",".join(args) + "]"
        fid = self.fact_name_to_id.get(fact_str)
        if fid is not None:
            return (state_bitwise >> fid) & 1
        return False

    def _get_rover_pos(self, state_bitwise, rover):
        # Scan possible locations. Usually efficient enough if waypoints aren't huge,
        # or we could optimize by caching if needed. 
        # Given bitwise state, iteration is necessary unless we cache indices.
        # Optimization: We only check `at[rover, ?]`
        # Since we don't have direct reverse map, we iterate waypoints.
        for wp in self.waypoints:
            if self._is_true(state_bitwise, "at", [rover, wp]):
                return wp
        return None

    def compute(self, node) -> int:
        state = node.state
        tn = node.task_network
        
        total_cost = 0
        
        # Pre-fetch rover positions to avoid repeated scanning
        rover_positions = {}
        for r in self.rovers:
            pos = self._get_rover_pos(state, r)
            if pos:
                rover_positions[r] = pos

        for task in tn:
            # Parse task name
            # Format assumption: name[arg1,arg2] similar to facts
            t_name = task.name
            if "[" in t_name:
                base_name = t_name.split("[")[0]
                args_str = t_name.split("[")[1][:-1]
                args = args_str.split(",") if args_str else []
            else:
                base_name = t_name
                args = []

            # --- Estimate Cost Based on Task Type ---
            
            if base_name in ["navigate", "sample_soil", "sample_rock", "drop", "calibrate", "take_image", 
                             "communicate_soil_data", "communicate_rock_data", "communicate_image_data"]:
                # Primitive actions cost 1
                total_cost += 1
                
            elif base_name == "do_navigate1" or base_name == "do_navigate2":
                # args: rover, to (or rover, from, to)
                if len(args) >= 2:
                    r = args[0]
                    dest = args[-1]
                    if r in rover_positions:
                        curr = rover_positions[r]
                        d = self.dist[r][curr][dest]
                        if d != self.inf:
                            total_cost += d
                        else:
                            return self.inf # Unreachable
                    else:
                        total_cost += 1 # Fallback if pos unknown

            elif base_name == "get_soil_data":
                wp = args[0]
                # Check if already communicated
                if self._is_true(state, "communicated_soil_data", [wp]):
                    continue
                
                # Find best rover
                min_task_cost = self.inf
                
                for r in self.equipped_soil:
                    if r not in rover_positions: continue
                    curr = rover_positions[r]
                    
                    # 1. Have analysis?
                    if self._is_true(state, "have_soil_analysis", [r, wp]):
                        # Just need to send
                        comm_dist = min([self.dist[r][curr][cwp] for cwp in self.comm_waypoints], default=self.inf)
                        min_task_cost = min(min_task_cost, comm_dist + 1)
                    else:
                        # Need to go there, sample, then send
                        to_sample_dist = self.dist[r][curr][wp]
                        
                        # Store management
                        # Find store of rover. 
                        # Assuming 1 store per rover named 'roverXstore'
                        store = f"{r}store"
                        store_full = self._is_true(state, "full", [store])
                        drop_cost = 1 if store_full else 0
                        
                        # Dist from sample to comm
                        sample_to_comm = min([self.dist[r][wp][cwp] for cwp in self.comm_waypoints], default=self.inf)
                        
                        cost = to_sample_dist + drop_cost + 1 + sample_to_comm + 1
                        min_task_cost = min(min_task_cost, cost)
                
                if min_task_cost != self.inf:
                    total_cost += min_task_cost

            elif base_name == "get_rock_data":
                wp = args[0]
                if self._is_true(state, "communicated_rock_data", [wp]):
                    continue
                
                min_task_cost = self.inf
                for r in self.equipped_rock:
                    if r not in rover_positions: continue
                    curr = rover_positions[r]
                    
                    if self._is_true(state, "have_rock_analysis", [r, wp]):
                        comm_dist = min([self.dist[r][curr][cwp] for cwp in self.comm_waypoints], default=self.inf)
                        min_task_cost = min(min_task_cost, comm_dist + 1)
                    else:
                        to_sample_dist = self.dist[r][curr][wp]
                        store = f"{r}store"
                        store_full = self._is_true(state, "full", [store])
                        drop_cost = 1 if store_full else 0
                        sample_to_comm = min([self.dist[r][wp][cwp] for cwp in self.comm_waypoints], default=self.inf)
                        cost = to_sample_dist + drop_cost + 1 + sample_to_comm + 1
                        min_task_cost = min(min_task_cost, cost)
                
                if min_task_cost != self.inf:
                    total_cost += min_task_cost

            elif base_name == "get_image_data":
                obj, mode = args
                if self._is_true(state, "communicated_image_data", [obj, mode]):
                    continue
                
                min_task_cost = self.inf
                
                # Identify cameras that support this objective and mode
                # And the rovers that carry them
                valid_rovers = set()
                for cam, r in self.on_board.items():
                    if mode in self.supports[cam]:
                        valid_rovers.add((r, cam))
                
                for r, cam in valid_rovers:
                    if r not in self.equipped_imaging: continue
                    if r not in rover_positions: continue
                    curr = rover_positions[r]
                    
                    if self._is_true(state, "have_image", [r, obj, mode]):
                        comm_dist = min([self.dist[r][curr][cwp] for cwp in self.comm_waypoints], default=self.inf)
                        min_task_cost = min(min_task_cost, comm_dist + 1)
                        continue
                    
                    # Cost to acquire image
                    calib_cost = 0
                    current_loc_for_nav = curr
                    
                    # Check calibration
                    if not self._is_true(state, "calibrated", [cam, r]):
                        # Need to go to calibration target
                        cal_target_obj = self.calibration_target.get(cam)
                        if cal_target_obj:
                            # Find closest visibility point for cal target
                            cal_wps = self.visible_from[cal_target_obj]
                            if not cal_wps: 
                                continue # Cannot calibrate
                            dist_to_cal = min([self.dist[r][curr][cw] for cw in cal_wps], default=self.inf)
                            calib_cost = dist_to_cal + 1
                            # After cal, rover is at one of cal_wps. For heuristic, assume closest.
                            # To be precise, we'd calculate subsequent cost from that specific wp.
                            # Approximation: add dist from cal target to objective view
                            current_loc_for_nav = None # Signal to restart dist calc from cal spots
                    
                    # Navigate to objective visibility
                    obj_wps = self.visible_from[obj]
                    if not obj_wps: continue
                    
                    if current_loc_for_nav is not None:
                        # Calibrated or didn't move
                        dist_to_obj = min([self.dist[r][current_loc_for_nav][ow] for ow in obj_wps], default=self.inf)
                    else:
                        # Moved to calibrate. Find min path: Curr -> Cal -> Obj
                        # Optimization: We already added `dist_to_cal`. Now we need min(dist(cal_wp, obj_wp))
                        # This is roughly dist from 'somewhere valid' to obj.
                        # Let's take min over all cal_wps to all obj_wps
                        cal_wps = self.visible_from[self.calibration_target[cam]]
                        dist_to_obj = self.inf
                        for cw in cal_wps:
                            for ow in obj_wps:
                                d = self.dist[r][cw][ow]
                                if d < dist_to_obj: dist_to_obj = d
                    
                    # From Obj wp to Comm wp
                    # Approximation: min dist from any obj_wp to any comm_wp
                    # A tighter bound would be: min_{ow in obj_wps} ( dist(curr, ow) + dist(ow, comm) )
                    
                    # Let's refine the whole chain for R
                    # Path: Curr -> [Cal_WP] -> Obj_WP -> Comm_WP
                    
                    best_r_cost = self.inf
                    
                    # Case A: Needs Calibration
                    if calib_cost > 0:
                        cal_wps = self.visible_from[self.calibration_target[cam]]
                        for cw in cal_wps:
                            d1 = self.dist[r][curr][cw]
                            if d1 == self.inf: continue
                            for ow in obj_wps:
                                d2 = self.dist[r][cw][ow]
                                if d2 == self.inf: continue
                                for km in self.comm_waypoints:
                                    d3 = self.dist[r][ow][km]
                                    if d3 == self.inf: continue
                                    # Path: Nav(cw)+Cal(1)+Nav(ow)+Take(1)+Nav(km)+Comm(1)
                                    c = d1 + 1 + d2 + 1 + d3 + 1
                                    if c < best_r_cost: best_r_cost = c
                    
                    # Case B: No Calibration needed
                    else:
                        for ow in obj_wps:
                            d1 = self.dist[r][curr][ow]
                            if d1 == self.inf: continue
                            for km in self.comm_waypoints:
                                d2 = self.dist[r][ow][km]
                                if d2 == self.inf: continue
                                # Path: Nav(ow)+Take(1)+Nav(km)+Comm(1)
                                c = d1 + 1 + d2 + 1
                                if c < best_r_cost: best_r_cost = c
                    
                    min_task_cost = min(min_task_cost, best_r_cost)

                if min_task_cost != self.inf:
                    total_cost += min_task_cost
            
            # Handle other send tasks similarly or assume handled by parent high-level tasks
            # If 'send_soil_data' is in the network (e.g. from partial decomposition), add cost
            elif base_name.startswith("send_"):
                # args: rover, data_source
                if len(args) >= 2:
                    r = args[0]
                    if r in rover_positions:
                        curr = rover_positions[r]
                        d = min([self.dist[r][curr][cwp] for cwp in self.comm_waypoints], default=self.inf)
                        total_cost += d + 1

        return total_cost
