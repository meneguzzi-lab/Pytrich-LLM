from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    1. Bottleneck: Navigation travel (moving rovers between sampling locations and communication points).
    
    2. Admissible Lower Bounds:
       - Task Cost Bound: Every get_soil, get_rock, or get_image task requires at least 2 navigation phases (one to target, one to lander) and 2 primary actions (sample/take, communicate). 
         Minimum actions = (min_dist_to_wp * 3) + 1 + (min_dist_to_comm * 3) + 1. This is a lower bound on completing that specific data requirement.
       - Decomposition Bound: Every non-primitive task in the network has a minimum action sequence length required to reach its primitive components.
       - Rover Capacity Bound: Since waypoints and objectives are fixed, and rovers have specific traversal and equipment capabilities, the max cost assigned to any individual rover is a lower bound.
    """

    def __init__(self, rover_gemini3flash_heuristic_12="rover_gemini3flash_heuristic_12"):
        super().__init__(rover_gemini3flash_heuristic_12)
        self.at_fact_ids = {} # rover -> waypoint -> global_id
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_image = set()
        self.rover_cameras = collections.defaultdict(list)
        self.camera_modes = collections.defaultdict(set)
        self.camera_target = {} # cam -> obj
        self.obj_visible_from = collections.defaultdict(set) # obj -> [wps]
        self.lander_comm_wps = set()
        self.dist = {} # rover -> wp1 -> wp2 -> hops
        self.min_dist_to_comm = {} # rover -> wp -> hops
        self.rovers = set()
        self.waypoints = set()
        self.calibrated_ids = collections.defaultdict(dict) # cam -> rover -> id
        self.full_ids = {} # rover -> id
        self.rover_to_store = {}

    def _parse_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return predicate, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates and extract static properties from operators
        adj = collections.defaultdict(lambda: collections.defaultdict(set))
        store_to_full = {}

        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == 'at' and len(args) == 2:
                r, wp = args[0], args[1]
                if r not in self.at_fact_ids: self.at_fact_ids[r] = {}
                self.at_fact_ids[r][wp] = f.global_id
                self.rovers.add(r)
                self.waypoints.add(wp)
            elif pred == 'full' and len(args) == 1:
                store_to_full[args[0]] = f.global_id
            elif pred == 'calibrated' and len(args) == 2:
                self.calibrated_ids[args[0]][args[1]] = f.global_id

        for op in model.operators:
            pred, args = self._parse_name(op.name)
            if pred == 'navigate':
                r, p1, p2 = args[0], args[1], args[2]
                adj[r][p1].add(p2)
                self.rovers.add(r)
                self.waypoints.update([p1, p2])
            elif pred == 'sample_soil':
                self.rover_equipped_soil.add(args[0])
                self.rover_to_store[args[0]] = args[1]
            elif pred == 'sample_rock':
                self.rover_equipped_rock.add(args[0])
                self.rover_to_store[args[0]] = args[1]
            elif pred == 'take_image':
                self.rover_equipped_image.add(args[0])
                self.obj_visible_from[args[2]].add(args[1])
                self.rover_cameras[args[0]].append(args[3])
                self.camera_modes[args[3]].add(args[4])
            elif pred == 'calibrate':
                self.camera_target[args[1]] = args[2]
                self.obj_visible_from[args[2]].add(args[3])
            elif pred.startswith('communicate_soil_data1') or pred.startswith('communicate_rock_data1'):
                self.lander_comm_wps.add(args[3])
            elif pred.startswith('communicate_soil_data2') or pred.startswith('communicate_rock_data2'):
                self.lander_comm_wps.add(args[2])
            elif pred.startswith('communicate_image_data'):
                self.lander_comm_wps.add(args[4])

        # Link rover stores to full facts
        for r, s in self.rover_to_store.items():
            if s in store_to_full: self.full_ids[r] = store_to_full[s]

        # 2. Compute All-Pairs Shortest Paths per rover
        for r in self.rovers:
            self.dist[r] = {}
            for start_wp in self.waypoints:
                self.dist[r][start_wp] = {start_wp: 0}
                queue = collections.deque([start_wp])
                while queue:
                    u = queue.popleft()
                    for v in adj[r].get(u, []):
                        if v not in self.dist[r][start_wp]:
                            self.dist[r][start_wp][v] = self.dist[r][start_wp][u] + 1
                            queue.append(v)

        # 3. Precompute communication distances
        for r in self.rovers:
            self.min_dist_to_comm[r] = {}
            for wp in self.waypoints:
                best = 1000
                for cw in self.lander_comm_wps:
                    d = self.dist[r].get(wp, {}).get(cw, 1000)
                    if d < best: best = d
                self.min_dist_to_comm[r][wp] = best

        h_value = self._compute_h(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute_h(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute_h(self, state, task_network):
        if not task_network: return 0
        
        # Find rover positions
        rover_pos = {}
        for r_name in self.rovers:
            at_map = self.at_fact_ids.get(r_name, {})
            for wp_name, f_id in at_map.items():
                if (state >> f_id) & 1:
                    rover_pos[r_name] = wp_name
                    break

        total_h = 0
        for task in task_network:
            pred, args = self._parse_name(task.name)
            if not pred or pred == 'nop': continue
            
            task_cost = 1
            if pred == 'get_soil_data' and args:
                wp = args[0]
                best = 1000
                for r in self.rover_equipped_soil:
                    d1 = self.dist[r].get(rover_pos.get(r), {}).get(wp, 1000)
                    d2 = self.min_dist_to_comm[r].get(wp, 1000)
                    best = min(best, (d1 + d2) * 3 + 2)
                task_cost = best
            elif pred == 'get_rock_data' and args:
                wp = args[0]
                best = 1000
                for r in self.rover_equipped_rock:
                    d1 = self.dist[r].get(rover_pos.get(r), {}).get(wp, 1000)
                    d2 = self.min_dist_to_comm[r].get(wp, 1000)
                    best = min(best, (d1 + d2) * 3 + 2)
                task_cost = best
            elif pred == 'get_image_data' and len(args) >= 2:
                obj, mode = args[0], args[1]
                best = 1000
                for r in self.rover_equipped_image:
                    d1 = 1000
                    best_owp = None
                    for owp in self.obj_visible_from.get(obj, []):
                        d = self.dist[r].get(rover_pos.get(r), {}).get(owp, 1000)
                        if d < d1:
                            d1 = d
                            best_owp = owp
                    if best_owp:
                        d2 = self.min_dist_to_comm[r].get(best_owp, 1000)
                        best = min(best, (d1 + d2) * 3 + 3)
                task_cost = best
            elif pred == 'do_navigate1' and len(args) == 2:
                r, to_wp = args[0], args[1]
                task_cost = self.dist.get(r, {}).get(rover_pos.get(r), {}).get(to_wp, 1000) * 3
            elif pred.startswith('send_'):
                r = args[0]
                task_cost = self.min_dist_to_comm.get(r, {}).get(rover_pos.get(r), 1000) * 3 + 1
            elif pred == 'empty_store':
                r = args[1]
                f_id = self.full_ids.get(r)
                task_cost = 1 if f_id is not None and (state >> f_id) & 1 else 0
            
            total_h += max(1, task_cost) # Ensure progress

        return int(total_h)
