from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic10(Heuristic):
    """
    Robot Domain Heuristic
    
    Bottleneck Analysis:
    The primary bottleneck is the single-carry constraint. The robot can only hold one package at a time.
    Every package delivery is a sequential process: move to package, pickup, move to goal, putdown.
    The total cost is the sum of these mandatory actions for all undelivered packages.
    
    Admissible Terms:
    1. Delivery Cost (Mandatory Actions): For every package not at its goal, at least 2 actions are required (pickup + putdown). 
       If currently holding a package, only 1 action (putdown) remains for that specific package.
    2. Travel Cost (Moves): The robot must travel from its current location to a package's location and then to its goal.
       Since the robot carries only one item, the distance from each undelivered package's current location to its 
       target room is an unavoidable lower bound on 'move' actions.
    3. Coordination Cost: If the robot is empty-handed, it must travel to the first package. This distance is a lower bound.
    
    Admissibility Sketch:
    Each term represents a disjoint set of mandatory operations. Because the robot is a serial bottleneck (single-carry),
    the distances for each package delivery do not overlap and must be performed in sequence.
    """

    def __init__(self, robot_gemini3flash_heuristic_10="robot_gemini3flash_heuristic_10"):
        super().__init__(robot_gemini3flash_heuristic_10)
        self.room_to_id = {}
        self.id_to_room = []
        self.pkg_to_id = {}
        self.id_to_pkg = []
        self.target_room = {}  # pkg_id -> room_id
        self.dist_matrix = []
        
        self.rloc_bits = {}  # room_id -> global_id
        self.pkg_holding_bit = {} # pkg_id -> global_id
        self.pkg_in_bits = {} # pkg_id -> {room_id: global_id}
        
        self.room_pkg_in_mask = [] # room_id -> bitmask of all in(p, room) facts
        self.all_undelivered_in_mask = 0
        self.all_holding_mask = 0
        self.fact_id_to_pkg_id = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify Objects
        rooms = sorted(list(set(args[0] for f in model.facts if (pred := self._parse_fact_name(f.name)[0]) == "rloc" for args in [self._parse_fact_name(f.name)[1]])))
        # If rloc didn't work (empty domain), look at door
        if not rooms:
            rooms = sorted(list(set(a for f in model.facts if (p := self._parse_fact_name(f.name)[0]) == "door" for a in self._parse_fact_name(f.name)[1][:2])))
        
        self.id_to_room = rooms
        self.room_to_id = {name: i for i, name in enumerate(rooms)}
        num_rooms = len(rooms)

        # 2. Map Facts
        pkg_names = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                pkg_names.add(args[0])
            elif pred == "holding":
                pkg_names.add(args[0])
        
        self.id_to_pkg = sorted(list(pkg_names))
        self.pkg_to_id = {name: i for i, name in enumerate(self.id_to_pkg)}
        num_pkgs = len(self.id_to_pkg)

        # 3. BFS All-Pairs Distances
        self.dist_matrix = [[1000] * num_rooms for _ in range(num_rooms)]
        adj = [[] for _ in range(num_rooms)]
        for i in range(num_rooms): self.dist_matrix[i][i] = 0
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "door":
                u, v = self.room_to_id.get(args[0]), self.room_to_id.get(args[1])
                if u is not None and v is not None:
                    adj[u].append(v)
        
        for start_node in range(num_rooms):
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            while q:
                curr, d = q.popleft()
                self.dist_matrix[start_node][curr] = d
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, d + 1))

        # 4. Target rooms and Predicate Masks
        self.room_pkg_in_mask = [0] * num_rooms
        self.all_undelivered_in_mask = 0
        self.all_holding_mask = 0
        
        # Extract goal_in markers
        pkg_targets_names = {}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "goal_in":
                pkg_targets_names[args[0]] = args[1]
        
        for p_name in self.id_to_pkg:
            p_id = self.pkg_to_id[p_name]
            t_name = pkg_targets_names.get(p_name)
            self.target_room[p_id] = self.room_to_id.get(t_name, -1)

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                self.rloc_bits[self.room_to_id[args[0]]] = f.global_id
            elif pred == "holding":
                p_id = self.pkg_to_id[args[0]]
                self.pkg_holding_bit[p_id] = f.global_id
                self.all_holding_mask |= (1 << f.global_id)
                self.fact_id_to_pkg_id[f.global_id] = p_id
            elif pred == "in":
                p_id = self.pkg_to_id[args[0]]
                r_id = self.room_to_id[args[1]]
                self.room_pkg_in_mask[r_id] |= (1 << f.global_id)
                self.fact_id_to_pkg_id[f.global_id] = p_id
                if r_id != self.target_room[p_id]:
                    self.all_undelivered_in_mask |= (1 << f.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        if not task_network:
            return 0

        curr_r_id = -1
        for r_id, bit in self.rloc_bits.items():
            if (state >> bit) & 1:
                curr_r_id = r_id
                break
        
        if curr_r_id == -1: return 0

        h = 0
        is_holding = False
        held_mask = state & self.all_holding_mask
        if held_mask:
            is_holding = True
            held_fact_id = (held_mask & -held_mask).bit_length() - 1
            held_p_id = self.fact_id_to_pkg_id[held_fact_id]
            target_r = self.target_room[held_p_id]
            h += 1 + self.dist_matrix[curr_r_id][target_r]

        min_dist_to_pickup = 1000
        has_ground_pkgs = False
        
        for r_id in range(len(self.id_to_room)):
            room_undelivered = state & self.room_pkg_in_mask[r_id] & self.all_undelivered_in_mask
            if room_undelivered:
                has_ground_pkgs = True
                d_to_r = self.dist_matrix[curr_r_id][r_id]
                if d_to_r < min_dist_to_pickup:
                    min_dist_to_pickup = d_to_r
                
                while room_undelivered:
                    fact_id = (room_undelivered & -room_undelivered).bit_length() - 1
                    p_id = self.fact_id_to_pkg_id[fact_id]
                    target_r = self.target_room[p_id]
                    h += 2 + self.dist_matrix[r_id][target_r]
                    room_undelivered &= (room_undelivered - 1)

        if not is_holding and has_ground_pkgs:
            h += min_dist_to_pickup

        # Tiny tie-breaker for search guidance
        return int(h) + (0.0001 * len(task_network))

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
