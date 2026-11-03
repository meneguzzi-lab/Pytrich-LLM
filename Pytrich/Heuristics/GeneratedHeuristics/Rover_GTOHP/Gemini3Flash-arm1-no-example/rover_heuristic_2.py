from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic2(Heuristic):
    '''
    Bottleneck Analysis: The primary bottleneck in the Rover domain is navigation 
    constrained by the availability of traversal paths (can_traverse) and visibility 
    requirements for both calibration and data transmission to landers.

    Admissibility Sketches:
    1. Goal-Based Bound: Each unachieved communication goal requires at least one 
       communication action and one sampling/imaging action (cost >= 2 or 3).
    2. Task-Based Bound: Each task in the network has a minimum action requirement 
       including mandatory navigation distances, sampling, and transmission steps.
    '''

    def __init__(self, rover_gemini3flash_heuristic_2="rover_gemini3flash_heuristic_2"):
        super().__init__(name=rover_gemini3flash_heuristic_2)
        self.dist = {} # rover -> w1 -> w2 -> cost
        self.min_trans_dist = {} # rover -> w -> cost
        self.at_rover_map = {} # rover -> w -> global_id
        self.soil_goal_ids = {} # w -> global_id
        self.rock_goal_ids = {} # w -> global_id
        self.image_goal_ids = {} # obj -> mode -> global_id
        self.full_store_map = {} # store -> global_id
        self.rover_store = {} # rover -> store
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.rover_modes = {} # rover -> set(modes)
        self.obj_vis = {} # obj -> [waypoint]
        self.trans_points = set()
        self.rovers = []
        self.waypoints = []
        self.COSTS = {
            'sample_soil': 1, 'sample_rock': 1, 'take_image': 1, 'calibrate': 1,
            'drop': 1, 'navigate': 1, 'visit': 0, 'unvisit': 0, 'nop': 0,
            'communicate_soil_data1': 1, 'communicate_soil_data2': 1,
            'communicate_rock_data1': 1, 'communicate_rock_data2': 1,
            'communicate_image_data': 1
        }

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        if "[" in name:
            idx = name.find("[")
            pred = name[:idx]
            args = [a.strip() for a in name[idx+1:-1].split(",")]
            return pred, args
        return name, []

    def _parse_task_name(self, name: str):
        if "[" in name:
            idx = name.find("[")
            tname = name[:idx]
            args = [a.strip() for a in name[idx+1:-1].split(",")]
            return tname, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        super().initialize(model, 0)
        self.rovers = [f.name for f in model.facts if 'at[' in f.name and ',' in f.name]
        rover_set = set()
        waypoint_set = set()
        lander_waypoints = set()
        visibility = collections.defaultdict(list)
        traversal = collections.defaultdict(list)
        
        rover_cams = collections.defaultdict(list)
        cam_modes = collections.defaultdict(list)

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'at': 
                rover_set.add(args[0]); waypoint_set.add(args[1])
                if args[0] not in self.at_rover_map: self.at_rover_map[args[0]] = {}
                self.at_rover_map[args[0]][args[1]] = f.global_id
            elif pred == 'at_lander': lander_waypoints.add(args[1])
            elif pred == 'visible': visibility[args[0]].append(args[1])
            elif pred == 'can_traverse': traversal[args[0]].append((args[1], args[2]))
            elif pred == 'equipped_for_soil_analysis': self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis': self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging': self.equipped_image.add(args[0])
            elif pred == 'store_of': self.rover_store[args[1]] = args[0]
            elif pred == 'full': self.full_store_map[args[0]] = f.global_id
            elif pred == 'on_board': rover_cams[args[1]].append(args[0])
            elif pred == 'supports': cam_modes[args[0]].append(args[1])
            elif pred == 'visible_from':
                if args[0] not in self.obj_vis: self.obj_vis[args[0]] = []
                self.obj_vis[args[0]].append(args[1])
            elif pred == 'communicated_soil_data': self.soil_goal_ids[args[0]] = f.global_id
            elif pred == 'communicated_rock_data': self.rock_goal_ids[args[0]] = f.global_id
            elif pred == 'communicated_image_data':
                if args[0] not in self.image_goal_ids: self.image_goal_ids[args[0]] = {}
                self.image_goal_ids[args[0]][args[1]] = f.global_id

        self.rovers = sorted(list(rover_set))
        self.waypoints = sorted(list(waypoint_set))
        
        for r in self.rovers:
            self.rover_modes[r] = set()
            for cam in rover_cams[r]:
                for m in cam_modes[cam]: self.rover_modes[r].add(m)

        self.trans_points = {w1 for w1 in self.waypoints for w2 in lander_waypoints if w2 in visibility[w1] or w1 == w2}

        for r in self.rovers:
            self.dist[r] = {w: {ww: 50 for ww in self.waypoints} for w in self.waypoints}
            edges = traversal.get(r, [])
            adj = collections.defaultdict(list)
            for u, v in edges:
                if v in visibility[u]: adj[u].append(v)
                if u in visibility[v]: adj[v].append(u)
            for start in self.waypoints:
                self.dist[r][start][start] = 0
                q = collections.deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if self.dist[r][start][v] == 50:
                            self.dist[r][start][v] = self.dist[r][start][u] + 1
                            q.append(v)

        for r in self.rovers:
            self.min_trans_dist[r] = {}
            for w in self.waypoints:
                m = min([self.dist[r][w][tp] for tp in self.trans_points], default=10)
                self.min_trans_dist[r][w] = m

        h_val = self._compute(initial_node.state, initial_node.task_network)
        self.initial_h = h_val
        return h_val

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        self.update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        h = 0
        at_r = {}
        for r in self.rovers:
            for w in self.waypoints:
                if (state >> self.at_rover_map.get(r, {}).get(w, 0)) & 1:
                    at_r[r] = w
                    break
            if r not in at_r: at_r[r] = self.waypoints[0]

        for task in task_network:
            tname, args = self._parse_task_name(task.name)
            if tname == 'get_soil_data':
                w = args[0]
                if (state >> self.soil_goal_ids.get(w, 0)) & 1: continue
                options = [self.dist[r][at_r[r]][w] + self.min_trans_dist[r][w] + 2 for r in self.equipped_soil]
                h += min(options) if options else 10
            elif tname == 'get_rock_data':
                w = args[0]
                if (state >> self.rock_goal_ids.get(w, 0)) & 1: continue
                options = [self.dist[r][at_r[r]][w] + self.min_trans_dist[r][w] + 2 for r in self.equipped_rock]
                h += min(options) if options else 10
            elif tname == 'get_image_data':
                obj, mode = args[0], args[1]
                if (state >> self.image_goal_ids.get(obj, {}).get(mode, 0)) & 1: continue
                eligible = [r for r in self.rovers if mode in self.rover_modes[r]]
                costs = []
                for r in eligible:
                    d_vis = min([self.dist[r][at_r[r]][wv] for wv in self.obj_vis.get(obj, [])], default=20)
                    d_trans = min([self.min_trans_dist[r][wv] for wv in self.obj_vis.get(obj, [])], default=5)
                    costs.append(d_vis + d_trans + 3)
                h += min(costs) if costs else 15
            elif tname == 'do_navigate1':
                r, to = args[0], args[1]
                h += self.dist.get(r, {}).get(at_r.get(r), {}).get(to, 5)
            elif tname == 'do_navigate2':
                r, f, to = args[0], args[1], args[2]
                h += self.dist.get(r, {}).get(f, {}).get(to, 5)
            elif tname == 'send_soil_data' or tname == 'send_rock_data':
                r, w = args[0], args[1]
                h += self.min_trans_dist[r][at_r[r]] + 1
            elif tname == 'send_image_data':
                r, o, m = args[0], args[1], args[2]
                h += self.min_trans_dist[r][at_r[r]] + 1
            elif tname in self.COSTS:
                h += self.COSTS[tname]
            else:
                h += 1
        return h
