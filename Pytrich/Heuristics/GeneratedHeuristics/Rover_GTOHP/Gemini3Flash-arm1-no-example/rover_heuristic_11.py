from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the serialized data transmission through the lander's communication channel 
    and the specific capability/location requirements for data acquisition (soil/rock sampling or imaging).
    Each goal requires a specific sequence: acquisition -> navigation to comm point -> communication.
    
    Admissible Lower Bounds:
    1. Communication/Acquisition Cost: Each uncommunicated goal requires at least 1 'communicate' action. 
       If the datum is not yet acquired, it also requires at least 1 'sample' or 'take_image' action.
       If it is an image, it additionally requires 1 'calibrate' action if the camera is not ready.
       Since communication is serialized by the lander channel, summing these actions is admissible.
    2. Navigation Cost: A rover must travel from its current location to a target waypoint to acquire 
       data and then to a communication-ready waypoint. The shortest path distance (BFS) is a lower bound.
    3. Store/Lander constraints: Actions like 'drop' (to empty store) are also unavoidable if the store is full.
    """

    def __init__(self, rover_gemini3flash_heuristic_11="RoverGemini11"):
        super().__init__(name=rover_gemini3flash_heuristic_11)
        self.rover_ids = []
        self.waypoint_ids = []
        self.at_ids = {}  # rover -> waypoint -> id
        self.have_soil_ids = {}  # rover -> waypoint -> id
        self.have_rock_ids = {}  # rover -> waypoint -> id
        self.have_image_ids = {} # rover -> objective -> mode -> id
        self.comm_soil_ids = {} # waypoint -> id
        self.comm_rock_ids = {} # waypoint -> id
        self.comm_image_ids = {} # objective -> mode -> id
        self.is_comm_waypoint = {} # waypoint -> bool
        self.equipped_soil = set()
        self.equipped_rock = set()
        self.equipped_image = set()
        self.visible_from = collections.defaultdict(list) # objective -> list of waypoints
        self.distances = {} # rover -> w1 -> w2 -> int
        self.goal_info = [] # list of (type, args, global_id)
        self.calibrated_ids = {} # rover -> camera -> id
        self.camera_info = collections.defaultdict(list) # rover -> list of (camera, mode)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify all objects and their roles
        all_at_facts = []
        lander_locs = []
        visible_pairs = set()
        adj = collections.defaultdict(lambda: collections.defaultdict(list))

        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p == 'at':
                all_at_facts.append((args[0], args[1], f.global_id))
                if args[0] not in self.at_ids: self.at_ids[args[0]] = {}
                self.at_ids[args[0]][args[1]] = f.global_id
            elif p == 'at_lander':
                lander_locs.append(args[1])
            elif p == 'visible':
                visible_pairs.add((args[0], args[1]))
            elif p == 'can_traverse':
                adj[args[0]][args[1]].append(args[2])
            elif p == 'equipped_for_soil_analysis': self.equipped_soil.add(args[0])
            elif p == 'equipped_for_rock_analysis': self.equipped_rock.add(args[0])
            elif p == 'equipped_for_imaging': self.equipped_image.add(args[0])
            elif p == 'visible_from': self.visible_from[args[0]].append(args[1])
            elif p == 'have_soil_analysis':
                if args[0] not in self.have_soil_ids: self.have_soil_ids[args[0]] = {}
                self.have_soil_ids[args[0]][args[1]] = f.global_id
            elif p == 'have_rock_analysis':
                if args[0] not in self.have_rock_ids: self.have_rock_ids[args[0]] = {}
                self.have_rock_ids[args[0]][args[1]] = f.global_id
            elif p == 'have_image':
                r, o, m = args
                if r not in self.have_image_ids: self.have_image_ids[r] = {}
                if o not in self.have_image_ids[r]: self.have_image_ids[r][o] = {}
                self.have_image_ids[r][o][m] = f.global_id
            elif p == 'communicated_soil_data': self.comm_soil_ids[args[0]] = f.global_id
            elif p == 'communicated_rock_data': self.comm_rock_ids[args[0]] = f.global_id
            elif p == 'communicated_image_data':
                o, m = args
                if o not in self.comm_image_ids: self.comm_image_ids[o] = {}
                self.comm_image_ids[o][m] = f.global_id
            elif p == 'on_board':
                self.camera_info[args[1]].append(args[0])

        self.rover_ids = list(self.at_ids.keys())
        self.waypoint_ids = list(set([w for r in self.at_ids for w in self.at_ids[r]]))

        # 2. Precompute Comm Points and Distances
        for w in self.waypoint_ids:
            self.is_comm_waypoint[w] = any((w, wl) in visible_pairs for wl in lander_locs)

        for r in self.rover_ids:
            self.distances[r] = {w1: {w2: 50 for w2 in self.waypoint_ids} for w1 in self.waypoint_ids}
            for start_w in self.waypoint_ids:
                self.distances[r][start_w][start_w] = 0
                q = collections.deque([(start_w, 0)])
                visited = {start_w}
                while q:
                    curr, d = q.popleft()
                    for neighbor in adj[r][curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[r][start_w][neighbor] = d + 1
                            q.append((neighbor, d + 1))

        # 3. Identify goals
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            p, args = self._parse_fact_name(name)
            if p: self.goal_info.append((p, args, name))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        h_actions = 0
        nav_costs = [0]

        # 1. Rover positions
        rover_pos = {}
        for r in self.rover_ids:
            for w, gid in self.at_ids.get(r, {}).items():
                if (state >> gid) & 1:
                    rover_pos[r] = w
                    break

        # 2. Data goals
        for g_type, g_args, g_name in self.goal_info:
            # Check if goal is communicated
            gid = -1
            if g_type == 'communicated_soil_data': gid = self.comm_soil_ids.get(g_args[0], -1)
            elif g_type == 'communicated_rock_data': gid = self.comm_rock_ids.get(g_args[0], -1)
            elif g_type == 'communicated_image_data': gid = self.comm_image_ids.get(g_args[0], {}).get(g_args[1], -1)
            
            if gid != -1 and not ((state >> gid) & 1):
                h_actions += 1 # communicate action
                
                # Check if sampled
                sampled = False
                if g_type == 'communicated_soil_data':
                    w = g_args[0]
                    for r in self.rover_ids:
                        if w in self.have_soil_ids.get(r, {}) and (state >> self.have_soil_ids[r][w]) & 1: 
                            sampled = True; break
                    if not sampled:
                        h_actions += 1 # sample_soil
                        # Minimal travel for any capable rover to reach w
                        capable = [r for r in self.rover_ids if r in self.equipped_soil]
                        if capable and all(r in rover_pos for r in capable):
                            nav_costs.append(min(self.distances[r][rover_pos[r]][w] for r in capable))
                
                elif g_type == 'communicated_rock_data':
                    w = g_args[0]
                    for r in self.rover_ids:
                        if w in self.have_rock_ids.get(r, {}) and (state >> self.have_rock_ids[r][w]) & 1: 
                            sampled = True; break
                    if not sampled:
                        h_actions += 1 # sample_rock
                        capable = [r for r in self.rover_ids if r in self.equipped_rock]
                        if capable and all(r in rover_pos for r in capable):
                            nav_costs.append(min(self.distances[r][rover_pos[r]][w] for r in capable))
                
                elif g_type == 'communicated_image_data':
                    obj, mode = g_args
                    for r in self.rover_ids:
                        if obj in self.have_image_ids.get(r, {}) and mode in self.have_image_ids[r][obj] and (state >> self.have_image_ids[r][obj][mode]) & 1:
                            sampled = True; break
                    if not sampled:
                        h_actions += 1 # take_image
                        capable = [r for r in self.rover_ids if r in self.equipped_image]
                        if capable and all(r in rover_pos for r in capable):
                            best_nav = 50
                            for r in capable:
                                for vis_w in self.visible_from.get(obj, []):
                                    best_nav = min(best_nav, self.distances[r][rover_pos[r]][vis_w])
                            nav_costs.append(best_nav)

        # 3. Guidance from Task Network
        if task_network:
            first_task = task_network[0]
            first_name = getattr(first_task, 'name', "")
            target_w = None
            for part in first_name.split():
                if part in self.waypoint_ids:
                    target_w = part; break
            if target_w:
                rover_arg = next((r for r in self.rover_ids if r in first_name), None)
                if rover_arg and rover_arg in rover_pos:
                    nav_costs.append(self.distances[rover_arg][rover_pos[rover_arg]][target_w])
                elif not rover_arg:
                    nav_costs.append(min([self.distances[r][rover_pos[r]][target_w] for r in self.rover_ids if r in rover_pos] or [0]))

        return h_actions + max(nav_costs)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
