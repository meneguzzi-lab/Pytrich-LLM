import sys
from typing import List, Dict, Tuple, Set, Union
from collections import deque
import re

from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Rover HDDL domain.
    
    Design Rationale:
    1. Static Analysis: The heuristic precomputes connectivity graphs (can_traverse), visibility graphs,
       and equipment maps (which rover has which tool) to avoid re-parsing static facts during search.
    2. Hierarchical Estimation: It looks at high-level abstract tasks (get_soil_data, get_image_data)
       and estimates the cost to complete them by simulating the necessary primitive actions: 
       Navigation -> Sampling/Imaging -> Communication.
    3. Relaxed Distance: It uses precomputed All-Pairs Shortest Paths (BFS) on the traverse graph 
       to estimate navigation costs instantly.
    4. Resource Selection: For abstract tasks that don't specify a rover (or imply a choice), 
       it greedily estimates the minimum cost among all capable rovers.
    5. State Awareness: It utilizes bitwise operations on the efficient Pytrich state representation
       to check dynamic conditions like 'at', 'full', 'calibrated', preventing redundant actions.
    """

    def __init__(self, model, rover_gemini3pro_heuristic_1=None):
        super().__init__(model)
        self.model = model
        self.inf = float('inf')
        
        # --- Static Knowledge Structures ---
        self.rovers = []
        self.waypoints = []
        self.objectives = []
        self.landers = []
        
        # Maps for connectivity and capability
        self.can_traverse = {} # rover -> {u -> set(v)}
        self.visible = {} # p1 -> set(p2)
        self.visible_from = {} # objective -> set(waypoint)
        self.at_lander = {} # lander -> waypoint
        self.store_of = {} # rover -> store
        self.on_board = {} # rover -> list(camera)
        self.supports = {} # camera -> set(mode)
        
        # Sets for quick capability lookup
        self.equipped_for_soil = set()
        self.equipped_for_rock = set()
        self.equipped_for_imaging = set()
        
        # Map: predicate -> tuple(args) -> global_id
        # Used for fast O(1) state checks on dynamic predicates
        self.fact_id_map = {}
        
        # Precomputed distances: rover -> start_wp -> end_wp -> cost
        self.distances = {}
        
        self._initialize_static_knowledge()
        self._precompute_distances()

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact string like '+predicate[arg1,arg2]' into ('predicate', ['arg1', 'arg2']).
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

    def _initialize_static_knowledge(self):
        """Parses model facts to build static graph and lookup tables."""
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            # Map dynamic predicates for state checking
            if pred in ['at', 'empty', 'full', 'have_soil_analysis', 'have_rock_analysis', 
                        'have_image', 'calibrated', 'communicated_soil_data', 
                        'communicated_rock_data', 'communicated_image_data']:
                if pred not in self.fact_id_map: self.fact_id_map[pred] = {}
                self.fact_id_map[pred][tuple(args)] = fact.global_id

            # Build static database
            if pred == 'can_traverse':
                r, f, t = args
                if r not in self.can_traverse: self.can_traverse[r] = {}
                if f not in self.can_traverse[r]: self.can_traverse[r][f] = set()
                self.can_traverse[r][f].add(t)
                if r not in self.rovers: self.rovers.append(r)
                if f not in self.waypoints: self.waypoints.append(f)
                if t not in self.waypoints: self.waypoints.append(t)

            elif pred == 'visible':
                p1, p2 = args
                if p1 not in self.visible: self.visible[p1] = set()
                self.visible[p1].add(p2)
                
            elif pred == 'visible_from':
                o, p = args
                if o not in self.visible_from: self.visible_from[o] = set()
                self.visible_from[o].add(p)
                if o not in self.objectives: self.objectives.append(o)
                
            elif pred == 'at_lander':
                l, p = args
                self.at_lander[l] = p
                self.landers.append(l)
                
            elif pred == 'store_of':
                s, r = args
                self.store_of[r] = s
                
            elif pred == 'on_board':
                c, r = args
                if r not in self.on_board: self.on_board[r] = []
                self.on_board[r].append(c)
                
            elif pred == 'supports':
                c, m = args
                if c not in self.supports: self.supports[c] = set()
                self.supports[c].add(m)
                
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_for_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_for_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_for_imaging.add(args[0])

        # Ensure unique lists
        self.rovers = list(set(self.rovers))
        self.waypoints = list(set(self.waypoints))

    def _precompute_distances(self):
        """Computes All-Pairs Shortest Paths for each rover using BFS."""
        for r in self.rovers:
            self.distances[r] = {}
            # If rover has no traverse rules, it implies it can't move or map is incomplete for it
            if r not in self.can_traverse: continue
            
            # BFS from every waypoint
            for start_node in self.waypoints:
                self.distances[r][start_node] = {}
                q = deque([(start_node, 0)])
                visited = {start_node}
                self.distances[r][start_node][start_node] = 0
                
                while q:
                    curr, dist = q.popleft()
                    # Expand neighbors if defined in map
                    if curr in self.can_traverse[r]:
                        for nxt in self.can_traverse[r][curr]:
                            if nxt not in visited:
                                visited.add(nxt)
                                self.distances[r][start_node][nxt] = dist + 1
                                q.append((nxt, dist + 1))

    def _get_distance(self, rover, start, end):
        """Returns precomputed distance or infinity if unreachable."""
        if start == end: return 0
        try:
            return self.distances.get(rover, {}).get(start, {}).get(end, self.inf)
        except: # Safety catch
            return self.inf

    def _is_true(self, state, predicate, args):
        """Checks if a dynamic fact is true in the current bitmask state."""
        key = tuple(args)
        if predicate in self.fact_id_map and key in self.fact_id_map[predicate]:
            gid = self.fact_id_map[predicate][key]
            return (state >> gid) & 1
        return False

    def _get_rover_location(self, state, rover):
        """Finds the current location of a rover from the state bitmask."""
        # Iterate possible locations for this rover. 
        # Optimization: iterate entries in fact_id_map['at'] that start with this rover
        if 'at' not in self.fact_id_map: return None
        # We iterate over the map keys. A reverse index could be faster if this is a bottleneck.
        for (r, w), gid in self.fact_id_map['at'].items():
            if r == rover and ((state >> gid) & 1):
                return w
        return None

    def __call__(self, state, task_network):
        total_cost = 0
        
        for task in task_network:
            # Parse task name and arguments. Pytrich format is typically "name arg1 arg2"
            # Normalize by replacing common separators
            cleaned_name = task.name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = cleaned_name.split()
            if not parts: continue
            
            task_type = parts[0].lower()
            args = parts[1:]

            # --- Heuristic for Abstract Tasks ---

            if 'get_soil_data' in task_type:
                # Abstract task: acquire soil data from a waypoint and communicate it
                waypoint = args[0]
                if self._is_true(state, 'communicated_soil_data', [waypoint]):
                    continue
                
                # Find best capable rover
                min_cost = self.inf
                for r in self.equipped_for_soil:
                    # Check if already sampled
                    if self._is_true(state, 'have_soil_analysis', [r, waypoint]):
                        cost = 0
                    else:
                        r_pos = self._get_rover_location(state, r)
                        if not r_pos: continue
                        
                        # Navigation + Sample cost
                        d_to_target = self._get_distance(r, r_pos, waypoint)
                        cost = d_to_target + 1 # +1 for sample action
                        
                        # Penalty if store is full
                        store = self.store_of.get(r)
                        if store and self._is_true(state, 'full', [store]):
                            cost += 1 # Drop action
                        
                    # Communication Cost
                    # Need to be at a waypoint visible from a lander
                    # We estimate distance from target (or current pos if done) to nearest comm point
                    # Start point for comms is 'waypoint' because we just sampled there
                    start_comm_node = waypoint
                    
                    min_comm_dist = self.inf
                    for l in self.landers:
                        l_pos = self.at_lander[l]
                        # Find waypoints visible to lander
                        visible_ws = [w for w, vis_set in self.visible.items() if l_pos in vis_set]
                        # Also check lander loc itself (often visible to itself)
                        if l_pos in self.visible.get(l_pos, set()): 
                            visible_ws.append(l_pos)
                        
                        for comm_wp in visible_ws:
                            d = self._get_distance(r, start_comm_node, comm_wp)
                            if d < min_comm_dist:
                                min_comm_dist = d
                    
                    if min_comm_dist != self.inf:
                        cost += min_comm_dist + 1 # +1 communicate action
                        if cost < min_cost:
                            min_cost = cost
                
                total_cost += min_cost if min_cost != self.inf else 10

            elif 'get_rock_data' in task_type:
                # Abstract task: acquire rock data
                waypoint = args[0]
                if self._is_true(state, 'communicated_rock_data', [waypoint]):
                    continue
                
                min_cost = self.inf
                for r in self.equipped_for_rock:
                    if self._is_true(state, 'have_rock_analysis', [r, waypoint]):
                        cost = 0
                    else:
                        r_pos = self._get_rover_location(state, r)
                        if not r_pos: continue
                        
                        d_to_target = self._get_distance(r, r_pos, waypoint)
                        cost = d_to_target + 1
                        
                        store = self.store_of.get(r)
                        if store and self._is_true(state, 'full', [store]):
                            cost += 1
                    
                    start_comm_node = waypoint
                    min_comm_dist = self.inf
                    for l in self.landers:
                        l_pos = self.at_lander[l]
                        visible_ws = [w for w, vis_set in self.visible.items() if l_pos in vis_set]
                        for comm_wp in visible_ws:
                            d = self._get_distance(r, start_comm_node, comm_wp)
                            if d < min_comm_dist:
                                min_comm_dist = d
                    
                    if min_comm_dist != self.inf:
                        cost += min_comm_dist + 1
                        if cost < min_cost:
                            min_cost = cost
                
                total_cost += min_cost if min_cost != self.inf else 10

            elif 'get_image_data' in task_type:
                # Abstract task: get image data
                objective = args[0]
                mode = args[1]
                if self._is_true(state, 'communicated_image_data', [objective, mode]):
                    continue
                
                min_cost = self.inf
                for r in self.equipped_for_imaging:
                    # Check if rover supports the mode
                    valid_cam = False
                    cameras = self.on_board.get(r, [])
                    for c in cameras:
                        if mode in self.supports.get(c, set()):
                            valid_cam = True
                            break
                    if not valid_cam: continue
                    
                    if self._is_true(state, 'have_image', [r, objective, mode]):
                        cost = 0
                        start_comm_node = self._get_rover_location(state, r)
                    else:
                        r_pos = self._get_rover_location(state, r)
                        if not r_pos: continue
                        
                        # Must navigate to a waypoint visible from objective
                        visible_spots = self.visible_from.get(objective, set())
                        if not visible_spots: continue
                        
                        min_nav_dist = self.inf
                        best_spot = None
                        for spot in visible_spots:
                            d = self._get_distance(r, r_pos, spot)
                            if d < min_nav_dist:
                                min_nav_dist = d
                                best_spot = spot
                        
                        if min_nav_dist == self.inf: continue
                        
                        cost = min_nav_dist + 1 # take image
                        cost += 1 # Estimation for calibration cost
                        start_comm_node = best_spot

                    # Communication
                    if not start_comm_node: continue

                    min_comm_dist = self.inf
                    for l in self.landers:
                        l_pos = self.at_lander[l]
                        visible_ws = [w for w, vis_set in self.visible.items() if l_pos in vis_set]
                        for comm_wp in visible_ws:
                            d = self._get_distance(r, start_comm_node, comm_wp)
                            if d < min_comm_dist:
                                min_comm_dist = d
                    
                    if min_comm_dist != self.inf:
                        cost += min_comm_dist + 1
                        if cost < min_cost:
                            min_cost = cost
                            
                total_cost += min_cost if min_cost != self.inf else 10

            elif 'navigate' in task_type: 
                # Handles 'navigate', 'do_navigate1', 'do_navigate2'
                # Args: rover, [from], to. We find 'to' as last arg, 'rover' as first.
                if len(args) >= 2:
                    r = args[0]
                    to = args[-1]
                    r_pos = self._get_rover_location(state, r)
                    if r_pos:
                        dist = self._get_distance(r, r_pos, to)
                        if dist != self.inf:
                            total_cost += dist
                        else:
                            total_cost += 5 # Penalty for unreachable path in this state
            
            # --- Base cost for other primitive actions ---
            elif not task.name.startswith("nop") and not task.name.startswith("m") and not task_type.startswith("visit"):
                 # Standard primitive actions cost 1 (sample, drop, calibrate, communicate, etc.)
                 total_cost += 1

        return total_cost
