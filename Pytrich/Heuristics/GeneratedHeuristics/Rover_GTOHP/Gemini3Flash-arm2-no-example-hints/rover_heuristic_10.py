from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RoverGTOHPGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for Rover_GTOHP.
    
    Bottleneck:
    The main bottleneck is rover navigation between waypoints to gather samples and 
    then to waypoints visible from the lander to transmit data. Movement is 
    expensive as a single waypoint hop requires 3 actions (visit, navigate, unvisit).

    Admissibility Sketches:
    1. Travel Cost: Moving a distance 'd' waypoints requires exactly 3*d actions due to 
       the hierarchical visit/unvisit structure. 3 * shortest_path is a lower bound.
    2. Data Gathering: Each goal (soil, rock, image) requires at least one sampling/imaging 
       action and one communication action. Cost = 1 (sample) + 1 (comm).
    3. Calibration: Images require a calibration action (1) and navigation to a calibration target.
    """
    
    def __init__(self, rover_gemini3flash_heuristic_10="rover_gemini3flash_heuristic_10"):
        super().__init__(rover_gemini3flash_heuristic_10)
        self.at_fact = {} # rover -> wp -> global_id
        self.soil_comm_fact = {} # wp -> global_id
        self.rock_comm_fact = {} # wp -> global_id
        self.image_comm_fact = {} # (obj, mode) -> global_id
        self.have_soil_fact = {} # rover -> wp -> global_id
        self.have_rock_fact = {} # rover -> wp -> global_id
        self.empty_fact = {} # store -> global_id
        self.calibrated_fact = {} # camera -> rover -> global_id
        
        self.rovers = []
        self.waypoints = []
        self.objectives = []
        self.cameras = []
        self.modes = []
        self.stores = []
        
        self.adj = {} # rover -> wp -> set(wp)
        self.dist = {} # rover -> wp -> wp -> int
        self.visible_to_lander = set()
        self.visible_from_obj = {} # obj -> set(wp)
        self.rover_equipped_soil = set()
        self.rover_equipped_rock = set()
        self.rover_equipped_image = set()
        self.rover_store = {} # rover -> store
        self.rover_cameras = {} # rover -> set(camera)
        self.camera_target = {} # camera -> objective
        self.camera_modes = {} # camera -> set(mode)

    def _parse_name(self, name):
        name = name.strip()
        if "[" in name and name.endswith("]"):
            bracket_idx = name.find("[")
            type_part = name[:bracket_idx].split(":")[-1].split(" ")[-1]
            args_part = name[bracket_idx+1:-1]
            return type_part, [a.strip() for a in args_part.split(",")]
        return name.split(":")[-1].split(" ")[-1], []

    def _get_pos(self, rover, state):
        if rover not in self.at_fact: return None
        for wp, fact_id in self.at_fact[rover].items():
            if (state >> fact_id) & 1: return wp
        return None

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Collect objects and identify static properties from operator signatures
        temp_rovers = set()
        temp_waypoints = set()
        temp_objectives = set()
        temp_cameras = set()
        temp_modes = set()
        temp_stores = set()

        for op in model.operators:
            type_part, args = self._parse_name(op.name)
            if type_part == "navigate":
                r, w1, w2 = args[0], args[1], args[2]
                temp_rovers.add(r)
                temp_waypoints.add(w1); temp_waypoints.add(w2)
                if r not in self.adj: self.adj[r] = {}
                if w1 not in self.adj[r]: self.adj[r][w1] = set()
                self.adj[r][w1].add(w2)
            elif type_part.startswith("communicate_"):
                if type_part == "communicate_image_data":
                    self.visible_to_lander.add(args[4])
                    temp_objectives.add(args[2]); temp_modes.add(args[3])
                elif type_part.endswith("1"):
                    self.visible_to_lander.add(args[3])
                    temp_waypoints.add(args[2])
                elif type_part.endswith("2"):
                    self.visible_to_lander.add(args[2])
                    temp_waypoints.add(args[2])
            elif type_part == "sample_soil":
                self.rover_equipped_soil.add(args[0])
                self.rover_store[args[0]] = args[1]
                temp_stores.add(args[1])
            elif type_part == "sample_rock":
                self.rover_equipped_rock.add(args[0])
                self.rover_store[args[0]] = args[1]
            elif type_part == "calibrate":
                r, c, o, w = args
                self.rover_equipped_image.add(r)
                if o not in self.visible_from_obj: self.visible_from_obj[o] = set()
                self.visible_from_obj[o].add(w)
                self.camera_target[c] = o
                if r not in self.rover_cameras: self.rover_cameras[r] = set()
                self.rover_cameras[r].add(c)
                temp_cameras.add(c); temp_objectives.add(o)
            elif type_part == "take_image":
                r, w, o, c, m = args
                if o not in self.visible_from_obj: self.visible_from_obj[o] = set()
                self.visible_from_obj[o].add(w)
                if c not in self.camera_modes: self.camera_modes[c] = set()
                self.camera_modes[c].add(m)
                if r not in self.rover_cameras: self.rover_cameras[r] = set()
                self.rover_cameras[r].add(c)
                temp_modes.add(m)

        self.rovers = list(temp_rovers)
        self.waypoints = list(temp_waypoints)
        self.objectives = list(temp_objectives)
        self.cameras = list(temp_cameras)
        self.stores = list(temp_stores)

        # 2. Build Fact Maps
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if pred == "at" and len(args) == 2:
                if args[0] in self.rovers: 
                    if args[0] not in self.at_fact: self.at_fact[args[0]] = {}
                    self.at_fact[args[0]][args[1]] = f.global_id
            elif pred == "communicated_soil_data": self.soil_comm_fact[args[0]] = f.global_id
            elif pred == "communicated_rock_data": self.rock_comm_fact[args[0]] = f.global_id
            elif pred == "communicated_image_data": self.image_comm_fact[(args[0], args[1])] = f.global_id
            elif pred == "have_soil_analysis":
                if args[0] not in self.have_soil_fact: self.have_soil_fact[args[0]] = {}
                self.have_soil_fact[args[0]][args[1]] = f.global_id
            elif pred == "have_rock_analysis":
                if args[0] not in self.have_rock_fact: self.have_rock_fact[args[0]] = {}
                self.have_rock_fact[args[0]][args[1]] = f.global_id
            elif pred == "empty": self.empty_fact[args[0]] = f.global_id
            elif pred == "calibrated":
                if args[0] not in self.calibrated_fact: self.calibrated_fact[args[0]] = {}
                self.calibrated_fact[args[0]][args[1]] = f.global_id

        # 3. BFS All-Pairs Distances
        for r in self.rovers:
            self.dist[r] = {w1: {w2: 99 for w2 in self.waypoints} for w1 in self.waypoints}
            for w_start in self.waypoints:
                self.dist[r][w_start][w_start] = 0
                if r not in self.adj or w_start not in self.adj[r]: continue
                queue = deque([(w_start, 0)])
                visited = {w_start}
                while queue:
                    curr, d = queue.popleft()
                    self.dist[r][w_start][curr] = d
                    for neighbor in self.adj[r].get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

        # 4. Final H value
        h_value = self._compute_h(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute_h(self, state, task_network):
        if not task_network: return 0
        
        rover_pos = {}
        for r in self.rovers:
            rover_pos[r] = self._get_pos(r, state)

        h = 0
        for task in task_network:
            t_type, args = self._parse_name(task.name)
            h += self._estimate_task_cost(t_type, args, state, rover_pos)
        return h

    def _estimate_task_cost(self, t_type, args, state, rover_pos):
        if t_type == "get_soil_data":
            wp = args[0]
            if wp in self.soil_comm_fact and (state >> self.soil_comm_fact[wp]) & 1: return 0
            min_c = 99
            for r in self.rover_equipped_soil:
                pos = rover_pos.get(r)
                if not pos: continue
                d1 = self.dist[r][pos][wp]
                # min travel to lander
                d2 = min([self.dist[r][wp][wv] for wv in self.visible_to_lander]) if self.visible_to_lander else 99
                min_c = min(min_c, 3*d1 + 3*d2 + 2)
            return min_c if min_c < 99 else 10
            
        elif t_type == "get_rock_data":
            wp = args[0]
            if wp in self.rock_comm_fact and (state >> self.rock_comm_fact[wp]) & 1: return 0
            min_c = 99
            for r in self.rover_equipped_rock:
                pos = rover_pos.get(r)
                if not pos: continue
                d1 = self.dist[r][pos][wp]
                d2 = min([self.dist[r][wp][wv] for wv in self.visible_to_lander]) if self.visible_to_lander else 99
                min_c = min(min_c, 3*d1 + 3*d2 + 2)
            return min_c if min_c < 99 else 10

        elif t_type == "get_image_data":
            obj, mode = args[0], args[1]
            if (obj, mode) in self.image_comm_fact and (state >> self.image_comm_fact[(obj, mode)]) & 1: return 0
            min_c = 99
            for r in self.rover_equipped_image:
                pos = rover_pos.get(r)
                if not pos: continue
                # d to obj
                d_obj = min([self.dist[r][pos][wv] for wv in self.visible_from_obj.get(obj, [])]) if obj in self.visible_from_obj else 99
                # d from obj to lander
                d_land = min([min([self.dist[r][wo][wv] for wv in self.visible_to_lander]) for wo in self.visible_from_obj.get(obj, [])]) if (obj in self.visible_from_obj and self.visible_to_lander) else 99
                min_c = min(min_c, 3*d_obj + 3*d_land + 3)
            return min_c if min_c < 99 else 15

        elif t_type == "do_navigate1":
            r, target = args[0], args[1]
            pos = rover_pos.get(r)
            return 3 * self.dist[r][pos][target] if pos else 5

        elif t_type == "do_navigate2":
            r, target = args[0], args[2]
            pos = rover_pos.get(r)
            d = self.dist[r][pos][target] if pos else 2
            return max(0, 3*d - 2) if d > 0 else 0

        elif t_type == "send_soil_data":
            r, wp = args[0], args[1]
            pos = rover_pos.get(r)
            d = min([self.dist[r][pos][wv] for wv in self.visible_to_lander]) if self.visible_to_lander and pos else 5
            return 3*d + 1
            
        elif t_type == "empty_store":
            s, r = args[0], args[1]
            if s in self.empty_fact and not ((state >> self.empty_fact[s]) & 1): return 1
            return 0
            
        elif t_type in ["calibrate", "sample_soil", "sample_rock", "take_image", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data", "visit", "unvisit", "navigate", "drop"]:
            return 1
        elif t_type == "nop":
            return 0
        return 1

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute_h(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
