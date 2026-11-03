from typing import List, Set, Dict, Tuple, Optional
import math
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini3ProHeuristic18(Heuristic):
    def __init__(self, rover_gemini3pro_heuristic_17):
        super().__init__(rover_gemini3pro_heuristic_17)
        self.model = rover_gemini3pro_heuristic_17
        
        # Static Domain Info
        self.rovers = set()
        self.waypoints = set()
        self.objectives = set()
        self.cameras = set()
        self.modes = set()
        self.stores = set()
        self.landers = set()
        
        # Static Predicate Maps
        self.can_traverse = {}  # rover -> {u -> set(v)}
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_imaging = set()
        self.on_board = {}  # camera -> rover
        self.calibration_target = {}  # camera -> objective
        self.supports = {}  # camera -> set(modes)
        self.visible_from = {}  # objective -> set(waypoints)
        self.lander_at = {}  # lander -> waypoint
        self.visible = {}  # wp -> set(visible_wps)
        self.store_of = {} # store -> rover
        
        # Precomputations
        self.comm_points = set()  # waypoints from which one can communicate
        self.rover_dist = {}  # rover -> {source -> {dest -> cost}}
        
        # Optimization for state parsing
        self.parsed_facts = [] # Index matches global_id
        self.relevant_fact_indices = {
            'at': [], 'empty': [], 'full': [], 
            'have_soil_analysis': [], 'have_rock_analysis': [], 
            'have_image': [], 'calibrated': [],
            'communicated_soil_data': [], 'communicated_rock_data': [], 
            'communicated_image_data': []
        }

        self._initialize_static_data()

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
        return name, []

    def _initialize_static_data(self):
        # 1. Parse all facts and extract static structure
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            self.parsed_facts.append((pred, args))
            
            # Index dynamic predicates for fast lookups in h()
            if pred in self.relevant_fact_indices:
                self.relevant_fact_indices[pred].append((fact.global_id, args))
            
            # Extract static predicates
            if pred == 'can_traverse':
                r, f, t = args
                if r not in self.can_traverse: self.can_traverse[r] = {}
                if f not in self.can_traverse[r]: self.can_traverse[r][f] = set()
                self.can_traverse[r][f].add(t)
                self.rovers.add(r)
                self.waypoints.add(f); self.waypoints.add(t)
            elif pred == 'equipped_for_soil_analysis':
                self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis':
                self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging':
                self.equipped_imaging.add(args[0])
            elif pred == 'on_board':
                self.on_board[args[0]] = args[1]
                self.cameras.add(args[0])
            elif pred == 'calibration_target':
                self.calibration_target[args[0]] = args[1]
            elif pred == 'supports':
                if args[0] not in self.supports: self.supports[args[0]] = set()
                self.supports[args[0]].add(args[1])
            elif pred == 'visible_from':
                o, w = args
                if o not in self.visible_from: self.visible_from[o] = set()
                self.visible_from[o].add(w)
                self.objectives.add(o)
            elif pred == 'at_lander':
                self.lander_at[args[0]] = args[1]
                self.landers.add(args[0])
            elif pred == 'visible':
                w1, w2 = args
                if w1 not in self.visible: self.visible[w1] = set()
                self.visible[w1].add(w2)
            elif pred == 'store_of':
                self.store_of[args[0]] = args[1]
                self.stores.add(args[0])

        # 2. Compute Comm Points (waypoints visible to a lander location)
        # Note: Landers are static. We check which waypoints can see a lander location.
        for l, w_lander in self.lander_at.items():
            # Waypoint w1 is a comm point if visible(w1, w_lander)
            # We iterate all 'visible' facts to find w1
            for w1, neighbors in self.visible.items():
                if w_lander in neighbors:
                    self.comm_points.add(w1)
            # Also check reflexivity implicitly: if rover at lander loc, can it comm? 
            # Usually visible(p, p) is true in Rover domain facts. If so, it's covered above.

        # 3. Compute All-Pairs Shortest Paths for each Rover (BFS)
        for r in self.rovers:
            self.rover_dist[r] = self._compute_bfs_distances(r)

    def _compute_bfs_distances(self, rover):
        distances = {}
        graph = self.can_traverse.get(rover, {})
        all_nodes = list(self.waypoints)
        
        for start_node in all_nodes:
            dists = {node: float('inf') for node in all_nodes}
            dists[start_node] = 0
            queue = [start_node]
            while queue:
                u = queue.pop(0)
                d = dists[u]
                if u in graph:
                    for v in graph[u]:
                        if dists[v] == float('inf'):
                            dists[v] = d + 1
                            queue.append(v)
            distances[start_node] = dists
        return distances

    def h(self, task_network, state_bitwise):
        cost = 0
        
        # 1. Efficient State Parsing
        # Only parse what we need by checking pre-indexed facts against state bits
        rover_at = {} 
        store_full = set() # rovers with full stores
        has_soil = set() 
        has_rock = set()
        has_image = set()
        calibrated = set()
        communicated_soil = set()
        communicated_rock = set()
        communicated_image = set()
        
        # Check 'at'
        for gid, args in self.relevant_fact_indices['at']:
            if (state_bitwise >> gid) & 1:
                rover_at[args[0]] = args[1]
        
        # Check 'full'
        for gid, args in self.relevant_fact_indices['full']:
            if (state_bitwise >> gid) & 1:
                # args[0] is store. Map to rover.
                if args[0] in self.store_of:
                    store_full.add(self.store_of[args[0]])
                    
        # Check analyses/images
        for gid, args in self.relevant_fact_indices['have_soil_analysis']:
            if (state_bitwise >> gid) & 1: has_soil.add((args[0], args[1]))
        for gid, args in self.relevant_fact_indices['have_rock_analysis']:
            if (state_bitwise >> gid) & 1: has_rock.add((args[0], args[1]))
        for gid, args in self.relevant_fact_indices['have_image']:
            if (state_bitwise >> gid) & 1: has_image.add((args[0], args[1], args[2]))
        for gid, args in self.relevant_fact_indices['calibrated']:
            if (state_bitwise >> gid) & 1: calibrated.add((args[0], args[1]))
            
        # Check goals achieved
        for gid, args in self.relevant_fact_indices['communicated_soil_data']:
            if (state_bitwise >> gid) & 1: communicated_soil.add(args[0])
        for gid, args in self.relevant_fact_indices['communicated_rock_data']:
            if (state_bitwise >> gid) & 1: communicated_rock.add(args[0])
        for gid, args in self.relevant_fact_indices['communicated_image_data']:
            if (state_bitwise >> gid) & 1: communicated_image.add((args[0], args[1]))

        # 2. Iterate Task Network
        for task in task_network:
            # Task name might be "taskname arg1 arg2" or "taskname[arg1,arg2]"
            # Normalize simply by replacing [,] with space
            norm_name = task.name.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            parts = norm_name.split()
            if not parts: continue
            task_type = parts[0]
            args = parts[1:]

            if task_type == 'get_soil_data':
                wp = args[0]
                if wp in communicated_soil: continue
                cost += self._estimate_soil_cost(wp, rover_at, has_soil, store_full)
                
            elif task_type == 'get_rock_data':
                wp = args[0]
                if wp in communicated_rock: continue
                cost += self._estimate_rock_cost(wp, rover_at, has_rock, store_full)
                
            elif task_type == 'get_image_data':
                obj, mode = args[0], args[1]
                if (obj, mode) in communicated_image: continue
                cost += self._estimate_image_cost(obj, mode, rover_at, has_image, calibrated)
                
            elif task_type == 'do_navigate1' or task_type == 'do_navigate2':
                # do_navigate1 ?r ?to
                if len(args) >= 2:
                    r, to = args[0], args[-1]
                    if r in rover_at:
                        dist = self.rover_dist.get(r, {}).get(rover_at[r], {}).get(to, 999)
                        if dist != float('inf'): cost += dist
                        
            elif task_type == 'send_soil_data':
                # send_soil_data ?r ?from
                r, wp = args[0], args[1]
                if r in rover_at: cost += self._cost_to_send(r, rover_at[r])
                
            elif task_type == 'send_rock_data':
                r, wp = args[0], args[1]
                if r in rover_at: cost += self._cost_to_send(r, rover_at[r])
                
            elif task_type == 'send_image_data':
                r = args[0]
                if r in rover_at: cost += self._cost_to_send(r, rover_at[r])
            
            # Primitive actions cost 1
            elif task_type in ['navigate', 'sample_soil', 'sample_rock', 'drop', 'calibrate', 'take_image', 
                               'communicate_soil_data1', 'communicate_soil_data2', 
                               'communicate_rock_data1', 'communicate_rock_data2', 'communicate_image_data']:
                cost += 1
                
        return cost

    def _estimate_soil_cost(self, wp, rover_at, has_soil, store_full):
        min_c = float('inf')
        for r in self.equipped_soil:
            if r not in rover_at: continue
            pos = rover_at[r]
            
            # Case 1: Already has sample
            if (r, wp) in has_soil:
                c = self._cost_to_send(r, pos)
            # Case 2: Needs to fetch
            else:
                dist = self.rover_dist.get(r, {}).get(pos, {}).get(wp, float('inf'))
                if dist == float('inf'): continue
                
                c = dist + 1 # Travel + Sample
                if r in store_full:
                    c += 1 # Drop
                
                # After sampling, at wp. Must send.
                c += self._cost_to_send(r, wp)
            
            if c < min_c: min_c = c
            
        return min_c if min_c != float('inf') else 999

    def _estimate_rock_cost(self, wp, rover_at, has_rock, store_full):
        min_c = float('inf')
        for r in self.equipped_rock:
            if r not in rover_at: continue
            pos = rover_at[r]
            
            if (r, wp) in has_rock:
                c = self._cost_to_send(r, pos)
            else:
                dist = self.rover_dist.get(r, {}).get(pos, {}).get(wp, float('inf'))
                if dist == float('inf'): continue
                c = dist + 1 
                if r in store_full: c += 1
                c += self._cost_to_send(r, wp)
            
            if c < min_c: min_c = c
        return min_c if min_c != float('inf') else 999

    def _estimate_image_cost(self, obj, mode, rover_at, has_image, calibrated):
        min_c = float('inf')
        
        # Identify rovers that can take this image
        # Must have camera, camera supports mode
        candidate_rovers = []
        for cam, r in self.on_board.items():
            if r in self.equipped_imaging and cam in self.supports and mode in self.supports[cam]:
                candidate_rovers.append((r, cam))
                
        for r, cam in candidate_rovers:
            if r not in rover_at: continue
            pos = rover_at[r]
            
            if (r, obj, mode) in has_image:
                c = self._cost_to_send(r, pos)
            else:
                # Needs to go to a waypoint visible from obj
                # Find closest visibility point
                visible_wps = self.visible_from.get(obj, set())
                if not visible_wps: continue
                
                min_travel = float('inf')
                target_wp = None
                for vwp in visible_wps:
                    d = self.rover_dist.get(r, {}).get(pos, {}).get(vwp, float('inf'))
                    if d < min_travel:
                        min_travel = d
                        target_wp = vwp
                
                if min_travel == float('inf'): continue
                
                c = min_travel + 1 # Travel + Take_Image
                
                # Calibration cost
                if (cam, r) not in calibrated:
                    # Needs to go to calibration target first? 
                    # If already at a place visible to calib target... simplified: add +2 (nav+calib)
                    # Accurate: dist(pos, calib_wp) + 1 + dist(calib_wp, target_wp)
                    # We'll use a constant penalty for admissibility/speed trade-off
                    c += 2 
                
                # Send cost from target_wp
                c += self._cost_to_send(r, target_wp)
            
            if c < min_c: min_c = c
            
        return min_c if min_c != float('inf') else 999

    def _cost_to_send(self, r, current_pos):
        # Dist to nearest comm point + 1
        min_dist = float('inf')
        dists = self.rover_dist.get(r, {}).get(current_pos, {})
        for cp in self.comm_points:
            d = dists.get(cp, float('inf'))
            if d < min_dist:
                min_dist = d
        return (min_dist + 1) if min_dist != float('inf') else 999
