from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

# BOTTLENECK ANALYSIS:
# The Robot domain's primary bottleneck is the single arm (armempty constraint), 
# which serializes all package transportation. A robot must pick up, move, 
# and put down each package individually. Movement between rooms is constrained 
# by door locations and door status (open/closed).

# ADMISSIBILITY SKETCHES:
# 1. Action Count Term: Each package not at its goal location requires at least 2 actions (pickup + putdown) 
#    if it is currently in a room, or 1 action (putdown) if it is currently being held. 
#    Because the robot can only handle one package at a time, these actions are strictly serial.
# 2. Movement Term: For every package that must be moved from Room A to Room B, the robot must 
#    perform at least dist(A, B) move actions. Additionally, the robot must move from its 
#    current location to the first package's location (or the goal location if holding). 
#    The sum of these minimum distances is a lower bound on total moves due to the serial constraint.
# 3. Symmetry Breaking: Small tie-breakers are added for the number of closed doors and task network 
#    complexity to guide the search toward opening doors and reducing the task list.

class RobotGemini3FlashHeuristic14(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_14="robot_heuristic"):
        super().__init__(robot_gemini3flash_heuristic_14)
        self.rooms = []
        self.pkgs = []
        self.doors = []
        self.pkg_goal_rooms = {} # pkg -> set of goal rooms
        self.rloc_fact_ids = {} # room -> global_id
        self.in_fact_ids = collections.defaultdict(dict) # pkg -> room -> global_id
        self.holding_fact_ids = {} # pkg -> global_id
        self.closed_fact_ids = {} # door -> global_id
        self.dist_matrix = {}
        
        self.all_rloc_mask = 0
        self.pkg_in_mask = {}
        self.all_holding_mask = 0
        self.fact_id_to_room = {}
        self.fact_id_to_pkg = {}
        self.all_closed_mask = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                r_name = args[0]
                if r_name not in self.rooms: self.rooms.append(r_name)
                self.rloc_fact_ids[r_name] = f.global_id
                self.all_rloc_mask |= (1 << f.global_id)
                self.fact_id_to_room[f.global_id] = r_name
            elif pred == "in":
                p_name, r_name = args[0], args[1]
                if p_name not in self.pkgs: self.pkgs.append(p_name)
                if r_name not in self.rooms: self.rooms.append(r_name)
                self.in_fact_ids[p_name][r_name] = f.global_id
                self.fact_id_to_room[f.global_id] = r_name
                self.fact_id_to_pkg[f.global_id] = p_name
            elif pred == "holding":
                p_name = args[0]
                self.holding_fact_ids[p_name] = f.global_id
                self.all_holding_mask |= (1 << f.global_id)
                self.fact_id_to_pkg[f.global_id] = p_name
            elif pred == "closed":
                d_name = args[0]
                if d_name not in self.doors: self.doors.append(d_name)
                self.closed_fact_ids[d_name] = f.global_id
                self.all_closed_mask |= (1 << f.global_id)

        # Build pkg_in_mask
        for p in self.pkgs:
            mask = 0
            for f_id in self.in_fact_ids[p].values():
                mask |= (1 << f_id)
            self.pkg_in_mask[p] = mask

        # Parse goal locations (static goal_in facts from initial state)
        initial_state = model.initial_state
        for f in model.facts:
            if (initial_state >> f.global_id) & 1:
                pred, args = self._parse_fact_name(f.name)
                if pred == "goal_in":
                    p_name, r_name = args[0], args[1]
                    if p_name not in self.pkg_goal_rooms:
                        self.pkg_goal_rooms[p_name] = set()
                    self.pkg_goal_rooms[p_name].add(r_name)

        # Room connectivity and distance matrix
        adj = collections.defaultdict(list)
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                adj[r1].append(r2)
                adj[r2].append(r1)

        for start_node in self.rooms:
            self.dist_matrix[start_node] = {r: 1000 for r in self.rooms}
            self.dist_matrix[start_node][start_node] = 0
            queue = collections.deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.dist_matrix[start_node][v] == 1000:
                        self.dist_matrix[start_node][v] = self.dist_matrix[start_node][u] + 1
                        queue.append(v)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        # 1. Find Robot Location
        rloc_bits = state & self.all_rloc_mask
        curr_rloc = None
        if rloc_bits:
            curr_rloc = self.fact_id_to_room.get(rloc_bits.bit_length() - 1)
        if not curr_rloc:
            return 0 # Final state

        total_h = 0
        min_dist_to_first_target = 1000
        packages_to_handle = False

        # 2. Analyze every package
        for p in self.pkgs:
            goal_rooms = self.pkg_goal_rooms.get(p, set())
            if not goal_rooms: continue

            # Check if being held
            if (state >> self.holding_fact_ids.get(p, 0)) & 1:
                packages_to_handle = True
                # Must move to goal and putdown
                min_g_dist = min(self.dist_matrix[curr_rloc].get(g, 1000) for g in goal_rooms)
                total_h += 1 + min_g_dist
                min_dist_to_first_target = min(min_dist_to_first_target, min_g_dist)
            else:
                # Check current room
                p_loc_bits = state & self.pkg_in_mask.get(p, 0)
                if p_loc_bits:
                    p_loc = self.fact_id_to_room.get(p_loc_bits.bit_length() - 1)
                    if p_loc not in goal_rooms:
                        packages_to_handle = True
                        # Must pickup, move to goal, putdown
                        min_g_dist = min(self.dist_matrix[p_loc].get(g, 1000) for g in goal_rooms)
                        total_h += 2 + min_g_dist
                        # Robot must move to pickup location
                        r_to_p_dist = self.dist_matrix[curr_rloc].get(p_loc, 1000)
                        min_dist_to_first_target = min(min_dist_to_first_target, r_to_p_dist)

        if packages_to_handle:
            if min_dist_to_first_target < 1000:
                total_h += min_dist_to_first_target
        
        # 3. Small Tie-breakers
        num_closed = bin(state & self.all_closed_mask).count('1')
        total_h += (0.001 * num_closed) + (0.0001 * len(task_network))

        return int(total_h) if total_h > 1000000 else int(total_h)

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
