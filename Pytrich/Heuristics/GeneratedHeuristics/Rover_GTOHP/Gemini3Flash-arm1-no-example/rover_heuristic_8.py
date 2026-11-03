from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is the sequential nature of data acquisition and communication. 
    A rover must navigate to a waypoint, acquire data (sample or image), and then navigate to a waypoint 
    capable of communicating with a lander. Each sample requires an empty store, and transmission is 
    serialized by available channels and rover movement.

    Admissible Lower Bounds:
    1. Goal Count Bound: Each unachieved 'communicated_...' goal requires at least two primitive actions 
       (one to acquire the data and one to communicate it). Cost = 2 * count.
    2. Task Network Decomposition Bound: Each high-level task in the task network (e.g., get_soil_data) 
       requires a minimum sequence of actions (navigate, sample, navigate, communicate). 
       Summing the minimum remaining actions for each task in the network provides a lower bound.
    3. Navigation Lower Bound: The minimum travel distance (number of navigate actions) for a rover 
       to reach the waypoint required by its current task network sequence.
    """

    def __init__(self, rover_gemini3flash_heuristic_8="rover_gemini3flash_heuristic_8"):
        super().__init__(rover_gemini3flash_heuristic_8)
        self.at_ids = {} # rover -> wp -> bit
        self.goal_bits = [] # list of (type, target, bit)
        self.obj_visible_from = {} # obj -> [wp]
        self.comm_wps = set() # wps that can communicate to a lander
        self.soil_rovers = set()
        self.rock_rovers = set()
        self.image_rovers = collections.defaultdict(set) # mode -> [rover]
        self.dist = {} # rover -> wp -> wp -> int
        self.waypoints = []
        self.rovers = []
        self.rover_at_bits = collections.defaultdict(list) # rover -> [(wp, bit)]

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for args in args_str.split(",") for a in args.split() if a]
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.waypoints = [o for o in model.desc.objects if "waypoint" in o]
        self.rovers = [o for o in model.desc.objects if "rover" in o]
        lander_wps = set()
        visible_pairs = set()
        traversable = collections.defaultdict(set) # rover -> (wp1, wp2)

        # Parse facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                # at ?rover ?waypoint
                self.rover_at_bits[args[0]].append((args[1], f.global_id))
            elif pred == "at_lander":
                lander_wps.add(args[1])
            elif pred == "visible":
                visible_pairs.add((args[0], args[1]))
                visible_pairs.add((args[1], args[0]))
            elif pred == "can_traverse":
                traversable[args[0]].add((args[1], args[2]))
            elif pred == "equipped_for_soil_analysis":
                self.soil_rovers.add(args[0])
            elif pred == "equipped_for_rock_analysis":
                self.rock_rovers.add(args[0])
            elif pred == "supports":
                # supports ?camera ?mode. We need to link camera to rover later.
                pass 
            elif pred == "on_board":
                # on_board ?camera ?rover
                pass
            elif pred == "visible_from":
                if args[0] not in self.obj_visible_from: self.obj_visible_from[args[0]] = []
                self.obj_visible_from[args[0]].append(args[1])

        # Link cameras/modes to rovers
        cam_to_rover = {}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on_board": cam_to_rover[args[0]] = args[1]
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "supports":
                if args[0] in cam_to_rover:
                    self.image_rovers[args[1]].add(cam_to_rover[args[0]])

        # Communication waypoints
        for w1 in self.waypoints:
            for lw in lander_wps:
                if (w1, lw) in visible_pairs:
                    self.comm_wps.add(w1)
                    break

        # Precompute distances (BFS)
        for r in self.rovers:
            self.dist[r] = {}
            for w_start in self.waypoints:
                self.dist[r][w_start] = {w: 100 for w in self.waypoints}
                self.dist[r][w_start][w_start] = 0
                queue = collections.deque([w_start])
                visited = {w_start}
                while queue:
                    curr = queue.popleft()
                    for (f, t) in traversable[r]:
                        if f == curr and t not in visited and (f, t) in visible_pairs:
                            self.dist[r][w_start][t] = self.dist[r][w_start][curr] + 1
                            visited.add(t)
                            queue.append(t)

        # Goal Bits
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            for f in model.facts:
                if f.name == name:
                    self.goal_bits.append((pred, args, f.global_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # 1. Goal cost
        goal_count = 0
        for pred, args, bit in self.goal_bits:
            if not ((state >> bit) & 1):
                goal_count += 1
        h_goals = goal_count * 2

        # 2. Task Network cost
        tn_cost = 0
        # Identify current rover locations once per call
        rover_locs = {}
        for r, bits in self.rover_at_bits.items():
            for wp, bit_id in bits:
                if (state >> bit_id) & 1:
                    rover_locs[r] = wp
                    break

        for task in task_network:
            t_name = task.name.split()
            if not t_name: continue
            t_type = t_name[0]
            
            if t_type == "get_soil_data":
                target_wp = t_name[1]
                min_r_nav = 100
                for r in self.soil_rovers:
                    loc = rover_locs.get(r, None)
                    if loc: min_r_nav = min(min_r_nav, self.dist[r][loc][target_wp])
                tn_cost += 2 + (min_r_nav if min_r_nav < 100 else 5)
            elif t_type == "get_rock_data":
                target_wp = t_name[1]
                min_r_nav = 100
                for r in self.rock_rovers:
                    loc = rover_locs.get(r, None)
                    if loc: min_r_nav = min(min_r_nav, self.dist[r][loc][target_wp])
                tn_cost += 2 + (min_r_nav if min_r_nav < 100 else 5)
            elif t_type == "get_image_data":
                obj, mode = t_name[1], t_name[2]
                min_r_nav = 100
                possible_wps = self.obj_visible_from.get(obj, [])
                for r in self.image_rovers[mode]:
                    loc = rover_locs.get(r, None)
                    if loc:
                        for target_wp in possible_wps:
                            min_r_nav = min(min_r_nav, self.dist[r][loc][target_wp])
                tn_cost += 3 + (min_r_nav if min_r_nav < 100 else 5)
            elif t_type in ["do_navigate1", "do_navigate2"]:
                r_name, target_wp = t_name[1], t_name[-1]
                loc = rover_locs.get(r_name, None)
                if loc: tn_cost += self.dist[r_name][loc][target_wp]
            elif t_type in ["navigate"]:
                tn_cost += 1
            elif t_type in ["sample_soil", "sample_rock", "take_image", "drop", "calibrate", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data"]:
                tn_cost += 1
        
        return max(h_goals, tn_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
