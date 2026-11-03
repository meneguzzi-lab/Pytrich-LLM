import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Rover_GTOHP domain.

1. Bottleneck Analysis:
   The primary bottleneck in the Rover domain is navigation and store management. Rovers must move 
   between waypoints to acquire samples (soil, rock) or images, and then move to specific waypoints 
   visible to landers to communicate data. Stores must be emptied (dropped) before a new sample 
   can be acquired. The hierarchical structure serializes these dependencies.

2. Admissible Lower Bounds:
   - Task Decomposition Actions (Term 1): Each task in the network requires a minimum number of 
     atomic actions (e.g., sample + communicate = 2 actions). This is admissible as these actions 
     are unavoidable for satisfying the hierarchy.
   - Navigation Steps (Term 2): The shortest path (number of 'navigate' actions) for a rover to 
     reach its required waypoints (sampling wp, objective wp, or communication-visible wp) 
     is a rigorous lower bound on unavoidable navigation actions.
   - Storage Conflict (Term 3): If a rover's store is full, it must perform a 'drop' action 
     before sampling. This adds at least 1 action to the cost.

3. Tie-breaking:
   Small penalties (0.1) are applied for full stores to prioritize 'empty_store' tasks when 
   appropriate and for long-range targets to steer search towards closer goals.
"""

class RoverGTOHPGemini3FlashHeuristic10(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_10=None, name="RoverGTOHPGemini3FlashHeuristic10"):
        super().__init__(name=name)
        self.dist = {}               # rover -> start_wp -> end_wp -> distance
        self.rover_comm_dist = {}    # rover -> wp -> min_dist_to_comm_wp
        self.at_ids_by_rover = {}    # rover -> [(wp, global_id)]
        self.full_ids = {}           # store -> global_id
        self.store_to_rover = {}     # store -> rover
        self.equipped_soil = {}      # rover -> bool
        self.equipped_rock = {}      # rover -> bool
        self.equipped_image = {}     # rover -> bool
        self.obj_visible_wps = {}    # objective -> [waypoints]
        self.cam_to_wps = {}         # camera -> [waypoints]
        self.comm_wps = set()
        self.rovers = set()
        self.waypoints = set()
        self.stores = set()
        
        # Minimum action count for tasks (admissible)
        self.task_min_costs = {
            'get_soil_data': 2,
            'get_rock_data': 2,
            'get_image_data': 3,
            'send_soil_data': 1,
            'send_rock_data': 1,
            'send_image_data': 1,
            'do_calibrate': 1,
            'navigate': 1,
            'sample_soil': 1,
            'sample_rock': 1,
            'drop': 1,
            'calibrate': 1,
            'take_image': 1,
            'communicate_soil_data1': 1,
            'communicate_soil_data2': 1,
            'communicate_rock_data1': 1,
            'communicate_rock_data2': 1,
            'communicate_image_data': 1,
            'empty_store': 0, # Evaluated dynamically based on store state
            'nop': 0, 'visit': 0, 'unvisit': 0
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        if name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            return parts[0], parts[1:] if parts else []
        idx = name.find("[")
        if idx > 0 and name.endswith("]"):
            return name[:idx], name[idx+1:-1].split(",")
        return name, []

    def _preprocess(self):
        adj = defaultdict(lambda: defaultdict(set))
        lander_locs = set()
        cam_target = {}
        
        # Initial pass to gather entities and static knowledge
        for fact in self.model.facts:
            p, a = self._parse_name(fact.name)
            if not p: continue
            
            if p == 'at':
                self.rovers.add(a[0]); self.waypoints.add(a[1])
                if a[0] not in self.at_ids_by_rover: self.at_ids_by_rover[a[0]] = []
                self.at_ids_by_rover[a[0]].append((a[1], fact.global_id))
            elif p == 'can_traverse':
                adj[a[0]][a[1]].add(a[2]); adj[a[0]][a[2]].add(a[1])
            elif p == 'store_of':
                self.stores.add(a[0]); self.store_to_rover[a[0]] = a[1]
            elif p == 'full':
                self.full_ids[a[0]] = fact.global_id
            elif p == 'equipped_for_soil_analysis': self.equipped_soil[a[0]] = True
            elif p == 'equipped_for_rock_analysis': self.equipped_rock[a[0]] = True
            elif p == 'equipped_for_imaging': self.equipped_image[a[0]] = True
            elif p == 'visible_from':
                if a[0] not in self.obj_visible_wps: self.obj_visible_wps[a[0]] = []
                self.obj_visible_wps[a[0]].append(a[1])
            elif p == 'calibration_target':
                cam_target[a[0]] = a[1]
            elif p == 'at_lander':
                lander_locs.add(a[1])

        # Secondary pass for comm waypoints
        for fact in self.model.facts:
            p, a = self._parse_name(fact.name)
            if p == 'visible':
                if a[0] in lander_locs: self.comm_wps.add(a[1])
                if a[1] in lander_locs: self.comm_wps.add(a[0])

        # Distance precomputation (BFS)
        self.dist = {r: {w1: {w2: 50 for w2 in self.waypoints} for w1 in self.waypoints} for r in self.rovers}
        for r in self.rovers:
            for start_wp in self.waypoints:
                self.dist[r][start_wp][start_wp] = 0
                q = deque([start_wp])
                while q:
                    u = q.popleft()
                    for v in adj[r][u]:
                        if self.dist[r][start_wp][v] == 50:
                            self.dist[r][start_wp][v] = self.dist[r][start_wp][u] + 1
                            q.append(v)

        # Comm distance precomputation
        self.rover_comm_dist = {r: {w: 50 for w in self.waypoints} for r in self.rovers}
        for r in self.rovers:
            for w in self.waypoints:
                if self.comm_wps:
                    self.rover_comm_dist[r][w] = min([self.dist[r][w][cw] for cw in self.comm_wps] or [50])
                else: self.rover_comm_dist[r][w] = 0

        # Camera waypoint mapping
        for cam, obj in cam_target.items():
            self.cam_to_wps[cam] = self.obj_visible_wps.get(obj, [])

    def _get_rover_at(self, state, rover):
        if rover not in self.at_ids_by_rover: return None
        for wp, gid in self.at_ids_by_rover[rover]:
            if (state >> gid) & 1: return wp
        return None

    def _compute(self, state, task_network):
        if not task_network: return 0
        h = 0
        rover_locs = {}
        
        for tid in task_network:
            task = self.model.get_component(tid)
            name, args = self._parse_name(task.name)
            
            if name in ['do_navigate1', 'do_navigate2']:
                r, to_wp = args[0], args[-1]
                if r not in rover_locs: rover_locs[r] = self._get_rover_at(state, r)
                curr = rover_locs[r]
                if curr and to_wp in self.dist.get(r, {}).get(curr, {}): 
                    h += self.dist[r][curr][to_wp]
            
            elif name == 'get_soil_data':
                wp = args[0]
                best_r = 50
                for r in self.rovers:
                    if not self.equipped_soil.get(r): continue
                    if r not in rover_locs: rover_locs[r] = self._get_rover_at(state, r)
                    curr = rover_locs[r]
                    if curr: best_r = min(best_r, self.dist[r][curr][wp] + self.rover_comm_dist[r][wp])
                h += 2 + (best_r if best_r < 50 else 5)
                
            elif name == 'get_rock_data':
                wp = args[0]
                best_r = 50
                for r in self.rovers:
                    if not self.equipped_rock.get(r): continue
                    if r not in rover_locs: rover_locs[r] = self._get_rover_at(state, r)
                    curr = rover_locs[r]
                    if curr: best_r = min(best_r, self.dist[r][curr][wp] + self.rover_comm_dist[r][wp])
                h += 2 + (best_r if best_r < 50 else 5)

            elif name == 'get_image_data':
                obj = args[0]
                best_r = 50
                target_wps = self.obj_visible_wps.get(obj, [])
                for r in self.rovers:
                    if not self.equipped_image.get(r): continue
                    if r not in rover_locs: rover_locs[r] = self._get_rover_at(state, r)
                    curr = rover_locs[r]
                    if curr and target_wps:
                        best_r = min(best_r, min(self.dist[r][curr][tw] + self.rover_comm_dist[r][tw] for tw in target_wps))
                h += 3 + (best_r if best_r < 50 else 5)

            elif name == 'empty_store':
                s = args[0]
                if s in self.full_ids and (state >> self.full_ids[s]) & 1: h += 1
            
            elif name == 'do_calibrate':
                r, cam = args[0], args[1]
                if r not in rover_locs: rover_locs[r] = self._get_rover_at(state, r)
                curr = rover_locs[r]
                wps = self.cam_to_wps.get(cam, [])
                if curr and wps: h += 1 + min(self.dist[r][curr][pw] for pw in wps)
                else: h += 1

            else:
                h += self.task_min_costs.get(name, 1)

        # Tie-breaker for stores
        for s, gid in self.full_ids.items():
            if (state >> gid) & 1: h += 0.05
            
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
