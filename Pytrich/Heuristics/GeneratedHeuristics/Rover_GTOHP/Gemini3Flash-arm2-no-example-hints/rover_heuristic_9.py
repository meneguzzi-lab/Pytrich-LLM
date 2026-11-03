from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RoverGTOHPGemini3FlashHeuristic9(Heuristic):
    """
    Domain-Dependent Heuristic for Rover_GTOHP.

    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is navigation cost under the constraint of a single sample store.
    Each data collection requires traveling to a specific waypoint, sampling (which requires an empty store),
    traveling to a communication-ready waypoint (visible to lander), and transmitting. This sequence must often
    be serialized per rover, and the minimum action cost per hop between waypoints is 3 (visit + navigate + unvisit).

    Admissible Lower Bounds:
    1. Goal-Specific Action Count: Every goal requires at least a sampling/imaging action and a communication action (2-3 actions).
    2. Navigation Lower Bound: The minimum actions required for a rover to reach a target waypoint and then reach a lander-visible
       waypoint is (Dist(at, target) + Dist(target, comm_point)) * 3. This is a lower bound because it uses the shortest path hops.
    3. Store Emptying: If a rover's store is full and it needs to sample, a 'drop' action (+1) is unavoidable.
    """

    def __init__(self, rover_gemini3flash_heuristic_9="rover_gemini3flash_heuristic_9"):
        super().__init__(name=rover_gemini3flash_heuristic_9)
        self.rover_at_ids = {}      # [rover][wp] -> global_id
        self.have_soil_ids = {}    # [rover][wp] -> global_id
        self.have_rock_ids = {}    # [rover][wp] -> global_id
        self.have_image_ids = {}   # [rover][obj][mode] -> global_id
        self.store_empty_ids = {}  # [rover] -> global_id
        self.comm_done_ids = []    # list of (type, target, fact_id)
        self.dist_matrix = {}      # [rover][wp1][wp2] -> hops
        self.precomputed_sample_work = {} # [rover][wp] -> hops to comm
        self.precomputed_image_work = {}  # [rover][obj][mode] -> (hops calib+target+comm)
        self.rover_list = []
        self.waypoint_list = []
        self.rover_caps = {}

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
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse Predicates/Facts
        rovers = set()
        waypoints = set()
        objs = set()
        modes = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                rovers.add(args[0])
                waypoints.add(args[1])
                if args[0] not in self.rover_at_ids: self.rover_at_ids[args[0]] = {}
                self.rover_at_ids[args[0]][args[1]] = fact.global_id
            elif pred == "have_soil_analysis":
                if args[0] not in self.have_soil_ids: self.have_soil_ids[args[0]] = {}
                self.have_soil_ids[args[0]][args[1]] = fact.global_id
            elif pred == "have_rock_analysis":
                if args[0] not in self.have_rock_ids: self.have_rock_ids[args[0]] = {}
                self.have_rock_ids[args[0]][args[1]] = fact.global_id
            elif pred == "have_image":
                r, o, m = args[0], args[1], args[2]
                if r not in self.have_image_ids: self.have_image_ids[r] = {}
                if o not in self.have_image_ids[r]: self.have_image_ids[r][o] = {}
                self.have_image_ids[r][o][m] = fact.global_id
            elif pred == "empty":
                # Use store_of heuristic to link to rover if possible, 
                # but simplest is parsing the name drop[rover, store]
                pass

        self.rover_list = list(rovers)
        self.waypoint_list = list(waypoints)

        # 2. Reconstruct Traversal and Capabilities from Operators
        adj = {r: collections.defaultdict(list) for r in self.rover_list}
        comm_wps = {r: set() for r in self.rover_list}
        image_targets = collections.defaultdict(lambda: collections.defaultdict(set))
        calib_targets = collections.defaultdict(set)
        
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if not pred: continue
            if pred == "navigate":
                adj[args[0]][args[1]].append(args[2])
            elif pred == "sample_soil":
                if args[0] not in self.rover_caps: self.rover_caps[args[0]] = set()
                self.rover_caps[args[0]].add('soil')
            elif pred == "sample_rock":
                if args[0] not in self.rover_caps: self.rover_caps[args[0]] = set()
                self.rover_caps[args[0]].add('rock')
            elif pred == "take_image":
                if args[0] not in self.rover_caps: self.rover_caps[args[0]] = set()
                self.rover_caps[args[0]].add('image')
                image_targets[args[2]][args[4]].add(args[1])
            elif pred.startswith("communicate_soil"):
                wp_c = args[3] if "data1" in pred else args[2]
                comm_wps[args[0]].add(wp_c)
            elif pred.startswith("communicate_rock"):
                wp_c = args[3] if "data1" in pred else args[2]
                comm_wps[args[0]].add(wp_c)
            elif pred == "communicate_image_data":
                comm_wps[args[0]].add(args[4])
            elif pred == "calibrate":
                calib_targets[args[2]].add(args[3])
            elif pred == "drop":
                self.store_empty_ids[args[0]] = model.ifacts_init # Placeholder, will find correctly below

        # Correct empty store ids
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "empty":
                for r in self.rover_list:
                    if r in fact.name: # Panda grounder often includes rover name in store name
                        self.store_empty_ids[r] = fact.global_id

        # 3. BFS for all-pairs distances per rover
        self.dist_matrix = {r: {w: {w2: 999 for w2 in self.waypoint_list} for w in self.waypoint_list} for r in self.rover_list}
        for r in self.rover_list:
            for start in self.waypoint_list:
                self.dist_matrix[r][start][start] = 0
                queue = collections.deque([start])
                while queue:
                    curr = queue.popleft()
                    for neighbor in adj[r][curr]:
                        if self.dist_matrix[r][start][neighbor] == 999:
                            self.dist_matrix[r][start][neighbor] = self.dist_matrix[r][start][curr] + 1
                            queue.append(neighbor)

        # 4. Precompute Goal Requirements
        self.precomputed_sample_work = {r: {w: min([self.dist_matrix[r][w][cw] for cw in comm_wps[r]] + [999]) for w in self.waypoint_list} for r in self.rover_list}
        
        self.precomputed_image_work = {r: {} for r in self.rover_list}
        for r in self.rover_list:
            for obj in calib_targets:
                for mode, w_img_set in image_targets[obj].items():
                    best_path = 999
                    for w_cal in calib_targets[obj]:
                        for w_img in w_img_set:
                            path = self.dist_matrix[r][w_cal][w_img] + self.precomputed_sample_work[r][w_img]
                            if path < best_path: best_path = path
                    self.precomputed_image_work[r][(obj, mode)] = best_path

        # 5. Goal mapping
        goal_names = model.state_explicit_repr(model.goals)
        for fact_name in goal_names:
            pred, args = self._parse_fact_name(fact_name)
            fact_id = -1
            for f in model.facts: 
                if f.name == fact_name: fact_id = f.global_id; break
            if pred == "communicated_soil_data":
                self.comm_done_ids.append(('soil', args[0], fact_id))
            elif pred == "communicated_rock_data":
                self.comm_done_ids.append(('rock', args[0], fact_id))
            elif pred == "communicated_image_data":
                self.comm_done_ids.append(('image', (args[0], args[1]), fact_id))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state, tn):
        rover_pos = {}
        for r in self.rover_list:
            rover_pos[r] = None
            for wp, fid in self.rover_at_ids.get(r, {}).items():
                if (state >> fid) & 1:
                    rover_pos[r] = wp
                    break

        total_h = 0
        max_single_goal = 0

        for g_type, target, g_fid in self.comm_done_ids:
            if (state >> g_fid) & 1: continue
            
            min_g_cost = 999
            for r in self.rover_list:
                r_pos = rover_pos[r]
                if not r_pos: continue
                
                cost = 999
                if g_type == 'soil' and 'soil' in self.rover_caps.get(r, set()):
                    if (state >> self.have_soil_ids.get(r, {}).get(target, 0)) & 1:
                        cost = self.precomputed_sample_work[r][r_pos] * 3 + 1
                    else:
                        store_empty = (state >> self.store_empty_ids.get(r, 0)) & 1
                        cost = (self.dist_matrix[r][r_pos][target] + self.precomputed_sample_work[r][target]) * 3 + 2 + (0 if store_empty else 1)
                elif g_type == 'rock' and 'rock' in self.rover_caps.get(r, set()):
                    if (state >> self.have_rock_ids.get(r, {}).get(target, 0)) & 1:
                        cost = self.precomputed_sample_work[r][r_pos] * 3 + 1
                    else:
                        store_empty = (state >> self.store_empty_ids.get(r, 0)) & 1
                        cost = (self.dist_matrix[r][r_pos][target] + self.precomputed_sample_work[r][target]) * 3 + 2 + (0 if store_empty else 1)
                elif g_type == 'image' and 'image' in self.rover_caps.get(r, set()):
                    obj, mode = target
                    if (state >> self.have_image_ids.get(r, {}).get(obj, {}).get(mode, 0)) & 1:
                        cost = self.precomputed_sample_work[r][r_pos] * 3 + 1
                    else:
                        # Path: current -> calib -> img -> comm
                        base_work = self.precomputed_image_work[r].get((obj, mode), 999)
                        if base_work < 999:
                            min_to_cal = min([self.dist_matrix[r][r_pos][cw] for cw in calib_targets.get(obj, [])] + [999])
                            cost = (min_to_cal + base_work) * 3 + 3
                
                if cost < min_g_cost: min_g_cost = cost
            
            if min_g_cost < 999:
                total_h += min_g_cost
                if min_g_cost > max_single_goal: max_single_goal = min_g_cost

        # term1: estimate of total primitive work / parallel factor
        term1 = total_h / max(1, len(self.rover_list))
        # term2: single most expensive remaining task sequence
        term2 = max_single_goal
        # term3: tie-breaking guidance for decomposition depth
        term3 = 0.1 * len(tn)

        return int(max(term1, term2) + term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
