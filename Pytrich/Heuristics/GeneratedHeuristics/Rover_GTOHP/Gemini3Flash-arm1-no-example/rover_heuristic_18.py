from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RoverGTOHPGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is the sequential navigation of waypoints to perform 
    actions (sampling, imaging) followed by travel to communication-ready waypoints to transmit data. 
    The store capacity (requiring 'drop') and camera calibration also add serial constraints.
    
    Admissibility Sketches:
    1. Goal Communication Cost (LB1): Every uncommunicated goal fact in the state requires 
       at least one primitive action ('communicate') and potentially navigation.
    2. Task Network Min-Cost (LB2): Every abstract task in the ordered network must decompose 
       into a set of primitive actions. The shortest path distance between the rover's 
       position and the task's required waypoints, plus the mandatory actions (sample, send), 
       provides a lower bound for that task.
    3. Total H = Max(LB1, LB2) plus tie-breaking penalties for symmetry.
    """

    def __init__(self, rover_gemini3flash_heuristic_18="rover_gemini3flash_heuristic_18"):
        super().__init__(rover_gemini3flash_heuristic_18)
        self.dist = {} # [rover][wp1][wp2]
        self.comm_wps = []
        self.obj_wps = {} # [objective] -> [waypoints]
        self.rover_equipped_soil = {}
        self.rover_equipped_rock = {}
        self.rover_equipped_image = {}
        self.camera_on_board = {} # [camera] -> rover
        self.camera_supports = {} # [mode] -> [cameras]
        self.camera_calib_target = {} # [camera] -> objective
        self.rover_at_ids = {} # [rover][wp] -> fact_id
        self.rover_have_soil_ids = {} # [rover][wp] -> fact_id
        self.rover_have_rock_ids = {} # [rover][wp] -> fact_id
        self.rover_have_image_ids = {} # [rover][obj][mode] -> fact_id
        self.calibrated_ids = {} # [camera][rover] -> fact_id
        self.empty_store_ids = {} # [store] -> fact_id
        self.store_of = {} # [store] -> rover
        self.goal_facts = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self.rovers = [o for o in model.desc.objects if model.desc.obj_type[o] == "rover"]
        self.waypoints = [o for o in model.desc.objects if model.desc.obj_type[o] == "waypoint"]
        self.cameras = [o for o in model.desc.objects if model.desc.obj_type[o] == "camera"]
        self.objectives = [o for o in model.desc.objects if model.desc.obj_type[o] == "objective"]
        self.modes = [o for o in model.desc.objects if model.desc.obj_type[o] == "mode"]
        self.stores = [o for o in model.desc.objects if model.desc.obj_type[o] == "store"]

        # 1. Parse static facts
        adj = {r: {wp: [] for wp in self.waypoints} for r in self.rovers}
        lander_wps = []
        visibility = {wp: [] for wp in self.waypoints}

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "visible":
                visibility[args[0]].append(args[1])
            elif pred == "can_traverse":
                adj[args[0]][args[1]].append(args[2])
            elif pred == "at_lander":
                lander_wps.append(args[1])
            elif pred == "visible_from":
                if args[0] not in self.obj_wps: self.obj_wps[args[0]] = []
                self.obj_wps[args[0]].append(args[1])
            elif pred == "equipped_for_soil_analysis":
                self.rover_equipped_soil[args[0]] = True
            elif pred == "equipped_for_rock_analysis":
                self.rover_equipped_rock[args[0]] = True
            elif pred == "equipped_for_imaging":
                self.rover_equipped_image[args[0]] = True
            elif pred == "on_board":
                self.camera_on_board[args[0]] = args[1]
            elif pred == "calibration_target":
                self.camera_calib_target[args[0]] = args[1]
            elif pred == "supports":
                if args[1] not in self.camera_supports: self.camera_supports[args[1]] = []
                self.camera_supports[args[1]].append(args[0])
            elif pred == "store_of":
                self.store_of[args[0]] = args[1]

            # Dynamic predicate IDs
            if pred == "at":
                if args[0] not in self.rover_at_ids: self.rover_at_ids[args[0]] = {}
                self.rover_at_ids[args[0]][args[1]] = f.global_id
            elif pred == "have_soil_analysis":
                if args[0] not in self.rover_have_soil_ids: self.rover_have_soil_ids[args[0]] = {}
                self.rover_have_soil_ids[args[0]][args[1]] = f.global_id
            elif pred == "have_rock_analysis":
                if args[0] not in self.rover_have_rock_ids: self.rover_have_rock_ids[args[0]] = {}
                self.rover_have_rock_ids[args[0]][args[1]] = f.global_id
            elif pred == "have_image":
                r, obj, mode = args[0], args[1], args[2]
                if r not in self.rover_have_image_ids: self.rover_have_image_ids[r] = {}
                if obj not in self.rover_have_image_ids[r]: self.rover_have_image_ids[r][obj] = {}
                self.rover_have_image_ids[r][obj][mode] = f.global_id
            elif pred == "calibrated":
                if args[0] not in self.calibrated_ids: self.calibrated_ids[args[0]] = {}
                self.calibrated_ids[args[0]][args[1]] = f.global_id
            elif pred == "empty":
                self.empty_store_ids[args[0]] = f.global_id

        # 2. Precompute distances using BFS for each rover (can_traverse is rover-specific)
        for r in self.rovers:
            self.dist[r] = {w1: {w2: 1000 for w2 in self.waypoints} for w1 in self.waypoints}
            for start_wp in self.waypoints:
                q = deque([(start_wp, 0)])
                self.dist[r][start_wp][start_wp] = 0
                while q:
                    curr, d = q.popleft()
                    for neighbor in adj[r][curr]:
                        if neighbor in visibility[curr]:
                            if self.dist[r][start_wp][neighbor] == 1000:
                                self.dist[r][start_wp][neighbor] = d + 1
                                q.append((neighbor, d + 1))

        # 3. Identify communication waypoints
        self.comm_wps = list(set([w1 for w1 in self.waypoints for w2 in lander_wps if w2 in visibility[w1]]))

        # 4. Cache navigation summaries
        self.min_dist_to_comm = {r: {wp: min([self.dist[r][wp][cw] for cw in self.comm_wps] + [1000]) for wp in self.waypoints} for r in self.rovers}
        
        # 5. Goal facts
        self.goal_facts = model.state_explicit_repr(model.goals)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if self.model.goal_reached(state, task_network): return 0
        
        # Goal Term: Simple count of unmet communication goals
        goal_h = 0
        explicit_state = self.model.state_explicit_repr(state)
        for g in self.goal_facts:
            if g not in explicit_state: goal_h += 2

        # Network Term: Sum of state-aware min-costs for ordered tasks
        task_h = 0
        rover_pos = {}
        for r in self.rovers:
            rover_pos[r] = next((wp for wp, fid in self.rover_at_ids.get(r, {}).items() if (state >> fid) & 1), None)

        for task in task_network:
            t_name, args = self._parse_task_name(task.name)
            t_name = t_name.lower()
            
            if t_name in ["nop", "visit", "unvisit"]: continue
            
            cost = 1
            if t_name == "do_navigate1" and len(args) >= 2:
                r, to_wp = args[0], args[1]
                if r in rover_pos and rover_pos[r] and to_wp in self.dist[r][rover_pos[r]]:
                    cost = self.dist[r][rover_pos[r]][to_wp]
                    rover_pos[r] = to_wp
            elif t_name == "get_soil_data" and len(args) >= 1:
                wp = args[0]
                possible_costs = []
                for r in self.rovers:
                    if r in self.rover_equipped_soil and rover_pos[r]:
                        d1 = self.dist[r][rover_pos[r]][wp]
                        d2 = self.min_dist_to_comm[r][wp]
                        possible_costs.append(d1 + 1 + d2 + 1)
                cost = min(possible_costs) if possible_costs else 10
            elif t_name == "get_rock_data" and len(args) >= 1:
                wp = args[0]
                possible_costs = []
                for r in self.rovers:
                    if r in self.rover_equipped_rock and rover_pos[r]:
                        d1 = self.dist[r][rover_pos[r]][wp]
                        d2 = self.min_dist_to_comm[r][wp]
                        possible_costs.append(d1 + 1 + d2 + 1)
                cost = min(possible_costs) if possible_costs else 10
            elif t_name == "get_image_data" and len(args) >= 2:
                obj, mode = args[0], args[1]
                possible_costs = []
                for cam in self.camera_supports.get(mode, []):
                    r = self.camera_on_board.get(cam)
                    if r and rover_pos[r]:
                        c_target = self.camera_calib_target.get(cam)
                        # Travel to calib -> calibrate -> travel to objective -> take image -> send
                        d1 = min([self.dist[r][rover_pos[r]][w] for w in self.obj_wps.get(c_target, [])] + [1000])
                        d2 = min([self.dist[r][w1][w2] for w1 in self.obj_wps.get(c_target, []) for w2 in self.obj_wps.get(obj, [])] + [1000])
                        d3 = min([self.dist[r][w][cw] for w in self.obj_wps.get(obj, []) for cw in self.comm_wps] + [1000])
                        possible_costs.append(d1 + 1 + d2 + 1 + d3 + 1)
                cost = min(possible_costs) if possible_costs else 15
            
            task_h += cost

        # Symmetry Breaking Tie-breaker
        penalty = 0
        if task_network:
            try:
                first_task_name, first_task_args = self._parse_task_name(task_network[0].name)
                for arg in first_task_args:
                    if "rover" in arg: 
                        num = int(''.join(filter(str.isdigit, arg)))
                        penalty = num * 0.001
            except: pass

        return int(max(goal_h, task_h)) + penalty

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")]
            return predicate, args
    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_name = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")]
            return task_name, args
        return name, []
