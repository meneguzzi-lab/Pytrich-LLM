import collections
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    Bottleneck Analysis:
    The main bottleneck is the serial execution of movement for each rover between targets.
    Collecting and communicating data involves a strict sequence of navigation to a target location,
    executing a collection action (sample_soil, sample_rock, or take_image), and then navigating
    to a communication point visible to a lander.

    Admissibility Sketches:
    1. Navigation Lower Bound: The do_navigate1 and do_navigate2 hierarchy decomposes into visit, navigate, 
       and unvisit actions. For a path of length 'd', this requires at least 3*d atomic actions. 
       Using BFS to compute shortest paths ensures this is a lower bound.
    2. Sequence Lower Bound: Every high-level goal task (get_soil_data, get_rock_data, get_image_data) 
       requires a fixed minimum number of atomic actions (sampling + communication) plus the distance 
       to the target and then to a communication point. The cost (3*d1 + 1 + 1 + 3*d2 + 1) is an 
       accurate count of the minimum required atomic actions.
    3. Independence: Because data collections are distinct and waypoints are visited sequentially 
       by a single rover, summing the costs of tasks in the network provides a strong but conservative estimate.
    """

    def __init__(self, rover_gemini3flash_heuristic_0="blind"):
        super().__init__(rover_gemini3flash_heuristic_0)
        self.rovers = set()
        self.waypoints = set()
        self.dist = {}  # dist[rover][w1][w2]
        self.at_fact_id = {}  # at_fact_id[rover][wp]
        self.full_fact_id = {}  # full_fact_id[store]
        self.rover_store = {}
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.img_rovers = set()
        self.comm_wps = {}  # comm_wps[rover]
        self.obj_vis_wps = {}  # obj -> set of wps
        self.cam_on_rover = {} # cam -> rover
        self.cam_target = {} # cam -> obj
        self.min_dist_to_comm = {} # min_dist_to_comm[rover][wp]

    def _parse_name(self, name):
        name = name.strip()
        if name.startswith("+"): name = name[1:]
        if name.startswith("-"): name = name[1:]
        if "[" not in name:
            return name, []
        parts = name.split("[")
        base = parts[0]
        args = parts[1][:-1].split(",")
        return base, args

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify objects
        adj = {}
        for op in model.operators:
            p, a = self._parse_name(op.name)
            if not a: continue
            if p == "navigate":
                r, w1, w2 = a
                self.rovers.add(r)
                self.waypoints.add(w1)
                self.waypoints.add(w2)
                adj.setdefault(r, {}).setdefault(w1, set()).add(w2)
            elif p == "sample_soil":
                self.soil_rovers.add(a[0])
                self.rover_store[a[0]] = a[1]
            elif p == "sample_rock":
                self.rock_rovers.add(a[0])
                self.rover_store[a[0]] = a[1]
            elif p == "take_image":
                self.img_rovers.add(a[0])
                self.obj_vis_wps.setdefault(a[2], set()).add(a[1])
            elif p == "calibrate":
                # calibrate[r, c, o, p]
                self.cam_on_rover[a[1]] = a[0]
                self.cam_target[a[1]] = a[2]
                self.obj_vis_wps.setdefault(a[2], set()).add(a[3])
            elif "communicate_soil_data1" in p:
                self.comm_wps.setdefault(a[0], set()).add(a[3])
            elif "communicate_soil_data2" in p:
                self.comm_wps.setdefault(a[0], set()).add(a[2])
            elif "communicate_rock_data1" in p:
                self.comm_wps.setdefault(a[0], set()).add(a[3])
            elif "communicate_rock_data2" in p:
                self.comm_wps.setdefault(a[0], set()).add(a[2])
            elif "communicate_image_data" in p:
                self.comm_wps.setdefault(a[0], set()).add(a[4])

        # 2. Precompute shortest paths per rover
        for r in self.rovers:
            self.dist[r] = {w: {w2: 999 for w2 in self.waypoints} for w in self.waypoints}
            for w in self.waypoints:
                self.dist[r][w][w] = 0
                if w not in adj.get(r, {}): continue
                q = collections.deque([w])
                while q:
                    u = q.popleft()
                    for v in adj[r][u]:
                        if self.dist[r][w][v] == 999:
                            self.dist[r][w][v] = self.dist[r][w][u] + 1
                            q.append(v)

        # 3. Precompute distance to comm
        for r in self.rovers:
            self.min_dist_to_comm[r] = {}
            for w in self.waypoints:
                cwps = self.comm_wps.get(r, set())
                self.min_dist_to_comm[r][w] = min([self.dist[r][w][cw] for cw in cwps] + [999])

        # 4. Map facts
        self.at_fact_id = {r: {} for r in self.rovers}
        for f in model.facts:
            p, a = self._parse_name(f.name)
            if p == "at" and len(a) == 2 and a[0] in self.rovers:
                self.at_fact_id[a[0]][a[1]] = f.global_id
            elif p == "full":
                self.full_fact_id[a[0]] = f.global_id

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _cost_nav1(self, r, w1, w2):
        d = self.dist[r][w1].get(w2, 999)
        if d == 0: return 1
        if d >= 999: return 100
        return 3 * d

    def _get_rover_loc(self, state, r):
        for w, fid in self.at_fact_id.get(r, {}).items():
            if (state >> fid) & 1: return w
        return None

    def _compute(self, state, task_network):
        h = 0
        rover_locs = {r: self._get_rover_loc(state, r) for r in self.rovers}

        for task in task_network:
            ttype, targs = self._parse_name(task.name)
            if ttype == "nop": continue

            # Atomic Guidance Logic
            if ttype == "get_soil_data":
                wp = targs[0]
                h += min([self._cost_nav1(r, rover_locs[r], wp) + 3 + 3 * self.min_dist_to_comm[r][wp]
                          for r in self.soil_rovers if rover_locs[r]] + [100])
            elif ttype == "get_rock_data":
                wp = targs[0]
                h += min([self._cost_nav1(r, rover_locs[r], wp) + 3 + 3 * self.min_dist_to_comm[r][wp]
                          for r in self.rock_rovers if rover_locs[r]] + [100])
            elif ttype == "get_image_data":
                obj = targs[0]
                # min over cameras/waypoints
                h += min([self._cost_nav1(r, rover_locs[r], wv) + 3 + 3 * self.min_dist_to_comm[r][wv]
                          for r in self.img_rovers if rover_locs[r] 
                          for wv in self.obj_vis_wps.get(obj, [])] + [100])
            elif ttype == "do_navigate1":
                r, to = targs[0], targs[1]
                loc = rover_locs.get(r)
                if loc: h += self._cost_nav1(r, loc, to)
            elif ttype == "send_soil_data" or ttype == "send_rock_data":
                r, wp = targs[0], targs[1]
                loc = rover_locs.get(r)
                if loc: h += 3 * self.min_dist_to_comm[r][loc] + 1
            elif ttype == "send_image_data":
                r = targs[0]
                loc = rover_locs.get(r)
                if loc: h += 3 * self.min_dist_to_comm[r][loc] + 1
            elif ttype == "do_calibrate":
                r, cam = targs[0], targs[1]
                loc = rover_locs.get(r)
                target = self.cam_target.get(cam)
                if loc and target:
                    h += min([self._cost_nav1(r, loc, w) + 1 for w in self.obj_vis_wps.get(target, [])] + [5])
            else:
                h += 1 # Default cost for unmodeled tasks

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
