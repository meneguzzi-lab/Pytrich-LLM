from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

# Bottleneck: Multi-rover navigation to distinct data source waypoints and communication locations visible to landers.
# Admissible Term 1: Action count lower bound. Each unachieved communicated-data goal requires at least a sampling/imaging action and a communication action.
# Admissible Term 2: Navigation lower bound. Minimum travel actions (scaled by 3 to account for hierarchical visit/unvisit overhead) to reach required locations.
# Admissible Term 3: Store management. If a rover is required to sample but its store is already full, a drop action is mandatory.

class RoverGTOHPGemini3FlashHeuristic18(Heuristic):
    def __init__(self, name="rover_gemini3flash_heuristic_18"):
        super().__init__(name)
        self.rover_gemini3flash_heuristic_18 = self
        self.at_id = {} # (rover, wp) -> global_id
        self.have_soil_id = {} # (rover, wp) -> global_id
        self.have_rock_id = {} # (rover, wp) -> global_id
        self.have_image_id = {} # (rover, obj, mode) -> global_id
        self.full_id = {} # store -> global_id
        self.calibrated_id = {} # (camera, rover) -> global_id
        self.comm_soil_goal_id = {} # wp -> global_id
        self.comm_rock_goal_id = {} # wp -> global_id
        self.comm_image_goal_id = {} # (obj, mode) -> global_id

        self.rovers = []
        self.waypoints = []
        self.objectives = []
        self.modes = []
        self.dist = {} # rover -> wp1 -> wp2 -> min_hops
        self.comm_wps = {} # rover -> set of waypoints
        self.source_wps = {} # obj -> set of waypoints
        self.cam_on_board = {} # cam -> rover
        self.cam_supports = {} # cam -> set of modes
        self.cam_calib_target = {} # cam -> obj
        self.store_of = {} # rover -> store
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_image = set()
        self.op_costs = {}
        self.nav_step_cost = 3

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        fact_id_map = {f.name: f.global_id for f in model.facts}
        
        for f in model.facts:
            p, a = self._parse_fact_name(f.name)
            if p == "at": self.at_id[(a[0], a[1])] = f.global_id
            elif p == "have_soil_analysis": self.have_soil_id[(a[0], a[1])] = f.global_id
            elif p == "have_rock_analysis": self.have_rock_id[(a[0], a[1])] = f.global_id
            elif p == "have_image": self.have_image_id[(a[0], a[1], a[2])] = f.global_id
            elif p == "full": self.full_id[a[0]] = f.global_id
            elif p == "calibrated": self.calibrated_id[(a[0], a[1])] = f.global_id

        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            p, a = self._parse_fact_name(name)
            if p == "communicated_soil_data": self.comm_soil_goal_id[a[0]] = fact_id_map[name]
            elif p == "communicated_rock_data": self.comm_rock_goal_id[a[0]] = fact_id_map[name]
            elif p == "communicated_image_data": self.comm_image_goal_id[(a[0], a[1])] = fact_id_map[name]

        rovers_set = set()
        wps_set = set()
        objs_set = set()
        modes_set = set()

        for op in model.operators:
            p, a = self._parse_fact_name(op.name)
            self.op_costs[p] = op.cost
            if p == "navigate":
                rovers_set.add(a[0]); wps_set.add(a[1]); wps_set.add(a[2])
                if a[0] not in self.dist: self.dist[a[0]] = {}
                if a[1] not in self.dist[a[0]]: self.dist[a[0]][a[1]] = {a[1]: 0}
                self.dist[a[0]][a[1]][a[2]] = 1
            elif p == "sample_soil":
                self.rover_equipped_soil.add(a[0]); self.store_of[a[0]] = a[1]
            elif p == "sample_rock":
                self.rover_equipped_rock.add(a[0]); self.store_of[a[0]] = a[1]
            elif p == "take_image":
                self.rover_equipped_image.add(a[0]); self.cam_on_board[a[3]] = a[0]; 
                if a[3] not in self.cam_supports: self.cam_supports[a[3]] = set()
                self.cam_supports[a[3]].add(a[4]); 
                if a[2] not in self.source_wps: self.source_wps[a[2]] = set()
                self.source_wps[a[2]].add(a[1]); objs_set.add(a[2]); modes_set.add(a[4])
            elif p == "calibrate":
                self.cam_calib_target[a[1]] = a[2]
            if "communicate" in p:
                if a[0] not in self.comm_wps: self.comm_wps[a[0]] = set()
                for bit in op.get_precons():
                    fp, fa = self._parse_fact_name(model.facts[bit].name)
                    if fp == "at": self.comm_wps[fa[0]].add(fa[1])

        self.rovers = sorted(list(rovers_set))
        self.waypoints = sorted(list(wps_set))
        self.objectives = sorted(list(objs_set))
        self.modes = sorted(list(modes_set))
        self.nav_step_cost = self.op_costs.get("navigate", 1) + self.op_costs.get("visit", 1) + self.op_costs.get("unvisit", 1)

        for r in self.rovers:
            if r not in self.dist: self.dist[r] = {w: {w: 0} for w in self.waypoints}
            d = self.dist[r]
            for w in self.waypoints:
                if w not in d: d[w] = {w: 0}
                for w2 in self.waypoints:
                    if w2 not in d[w]: d[w][w2] = 1000
            for k in self.waypoints:
                for i in self.waypoints:
                    for j in self.waypoints:
                        if d[i][j] > d[i][k] + d[k][j]: d[i][j] = d[i][k] + d[k][j]

        self.min_comm_dist = {r: {w1: min([self.dist[r][w1][w2] for w2 in self.comm_wps.get(r, [])] + [1000]) for w1 in self.waypoints} for r in self.rovers}
        self.min_dist_to_obj = {r: {obj: {w1: min([self.dist[r][w1][w2] for w2 in self.source_wps.get(obj, [])] + [1000]) for w1 in self.waypoints} for obj in self.objectives} for r in self.rovers}
        
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, tn):
        if not tn and (state & self.model.goals) != self.model.goals: return 10000
        h = 0
        for task in tn:
            if self.model.iop_init <= task.global_id <= self.model.iop_end:
                h += task.cost
            else:
                name, args = self._parse_task(task.name)
                if name == 'get_soil_data': h += self._est_soil(state, args[0])
                elif name == 'get_rock_data': h += self._est_rock(state, args[0])
                elif name == 'get_image_data': h += self._est_image(state, args[0], args[1])
                elif name == 'send_soil_data': h += self._est_send(state, args[0], args[1], "soil")
                elif name == 'send_rock_data': h += self._est_send(state, args[0], args[1], "rock")
                elif name == 'send_image_data': h += self._est_send_img(state, args[0], args[1], args[2])
                elif name == 'do_navigate1': h += self._est_nav(state, args[0], args[1])
                elif name == 'do_calibrate': h += self._est_calib(state, args[0], args[1])
                elif name == 'empty_store': h += self._est_empty(state, args[0], args[1])
        return int(h)

    def _is_true(self, state, gid): return (state >> gid) & 1 if gid is not None else False
    def _get_pos(self, state, rover):
        for wp in self.waypoints:
            if self._is_true(state, self.at_id.get((rover, wp))): return wp
        return None

    def _est_soil(self, state, wp):
        if self._is_true(state, self.comm_soil_goal_id.get(wp)): return 0
        best = 1000
        for r in self.rovers:
            if r in self.rover_equipped_soil:
                if self._is_true(state, self.have_soil_id.get((r, wp))): cost = self.nav_step_cost * self.min_comm_dist[r][self._get_pos(state, r) or wp] + 1
                else: cost = self.nav_step_cost * (self.dist[r][self._get_pos(state, r) or self.waypoints[0]][wp] + self.min_comm_dist[r][wp]) + 2
                best = min(best, cost)
        return best

    def _est_rock(self, state, wp):
        if self._is_true(state, self.comm_rock_goal_id.get(wp)): return 0
        best = 1000
        for r in self.rovers:
            if r in self.rover_equipped_rock:
                if self._is_true(state, self.have_rock_id.get((r, wp))): cost = self.nav_step_cost * self.min_comm_dist[r][self._get_pos(state, r) or wp] + 1
                else: cost = self.nav_step_cost * (self.dist[r][self._get_pos(state, r) or self.waypoints[0]][wp] + self.min_comm_dist[r][wp]) + 2
                best = min(best, cost)
        return best

    def _est_image(self, state, obj, mode):
        if self._is_true(state, self.comm_image_goal_id.get((obj, mode))): return 0
        best = 1000
        for cam, rover in self.cam_on_board.items():
            if mode in self.cam_supports.get(cam, set()) and rover in self.rover_equipped_image:
                curr = self._get_pos(state, rover) or self.waypoints[0]
                if self._is_true(state, self.have_image_id.get((rover, obj, mode))): cost = self.nav_step_cost * self.min_comm_dist[rover][curr] + 1
                else:
                    cal_t = self.cam_calib_target.get(cam)
                    d1 = 0 if self._is_true(state, self.calibrated_id.get((cam, rover))) else (self.min_dist_to_obj[rover][cal_t][curr] * self.nav_step_cost + 1)
                    d2 = min([self.min_dist_to_obj[rover][obj][cw] for cw in self.source_wps.get(cal_t, [])] + [1000])
                    d3 = min([self.min_comm_dist[rover][sw] for sw in self.source_wps.get(obj, [])] + [1000])
                    cost = d1 + (d2 + d3) * self.nav_step_cost + 2
                best = min(best, cost)
        return best

    def _est_send(self, state, r, wp, t):
        if t == "soil" and self._is_true(state, self.comm_soil_goal_id.get(wp)): return 0
        if t == "rock" and self._is_true(state, self.comm_rock_goal_id.get(wp)): return 0
        return self.nav_step_cost * self.min_comm_dist[r][self._get_pos(state, r) or self.waypoints[0]] + 1

    def _est_send_img(self, state, r, o, m):
        if self._is_true(state, self.comm_image_goal_id.get((o, m))): return 0
        return self.nav_step_cost * self.min_comm_dist[r][self._get_pos(state, r) or self.waypoints[0]] + 1

    def _est_nav(self, state, r, to): return self.nav_step_cost * self.dist[r][self._get_pos(state, r) or self.waypoints[0]][to]
    def _est_calib(self, state, r, cam): 
        if self._is_true(state, self.calibrated_id.get((cam, r))): return 0
        return self.nav_step_cost * self.min_dist_to_obj[r][self.cam_calib_target.get(cam)][self._get_pos(state, r) or self.waypoints[0]] + 1
    def _est_empty(self, state, s, r): return 1 if self._is_true(state, self.full_id.get(s)) else 0

    def _parse_fact_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        b_idx = name.find("[")
        if b_idx > 0 and name.endswith("]"): return name[:b_idx], name[b_idx+1:-1].split(",")
        return name, []

    def _parse_task(self, name):
        if "[" in name: 
            base = name.split("[")[0]
            if base.startswith("+"): base = base[1:]
            return base, name.split("[")[1][:-1].split(",")
        return name, []
