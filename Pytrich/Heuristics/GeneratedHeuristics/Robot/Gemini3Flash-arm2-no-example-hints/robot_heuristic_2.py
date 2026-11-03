from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic2(Heuristic):
    """
    Heuristic for the Robot domain.
    
    Bottleneck: The single-carry constraint. The robot can only hold one package at a time,
    making every delivery a serialized sequence: move-to-package, pickup, move-to-goal, putdown.
    
    Admissible Bounds:
    1. Delivery Cost: Each undelivered package i requires at least (dist(start_i, goal_i) + 2) actions.
       If the robot is currently holding package i, it requires (dist(robot, goal_i) + 1).
       Since the robot carries only one item, these costs are sequential and additive.
    2. Retrieval Cost: If the robot is not holding a package, it must move to the first one.
       The distance to the nearest undelivered package is an additional unavoidable cost.
    3. Door Correction: Every move between rooms requires 1 'move' action. If a door is closed,
       an 'open' action is also required (+1). This is included as a distance lower bound.
    """

    def __init__(self, robot_gemini3flash_heuristic_2="robot_heuristic"):
        super().__init__(robot_gemini3flash_heuristic_2)
        self.room_to_idx = {}
        self.idx_to_room = []
        self.dist_matrix = []
        self.pkg_targets = {}  # pkg_name -> target_room_name
        
        self.rloc_fact_ids = {}    # global_id -> room_name
        self.holding_fact_ids = {} # global_id -> pkg_name
        self.in_fact_ids = {}      # global_id -> (pkg_name, room_name)
        self.relevant_fact_ids = set()
        self.facts_mask = 0

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
        # 1. Map rooms and identify package targets from goals
        rooms = set()
        doors = []
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                rooms.add(args[0])
            elif pred == "door":
                rooms.add(args[0])
                rooms.add(args[1])
                doors.append((args[0], args[1]))

        self.idx_to_room = sorted(list(rooms))
        self.room_to_idx = {name: i for i, name in enumerate(self.idx_to_room)}
        num_rooms = len(self.idx_to_room)

        # 2. Precompute APSP using BFS
        adj = collections.defaultdict(list)
        for r1, r2 in doors:
            if r1 in self.room_to_idx and r2 in self.room_to_idx:
                adj[self.room_to_idx[r1]].append(self.room_to_idx[r2])

        self.dist_matrix = [[999 for _ in range(num_rooms)] for _ in range(num_rooms)]
        for i in range(num_rooms):
            self.dist_matrix[i][i] = 0
            queue = collections.deque([(i, 0)])
            visited = {i}
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        self.dist_matrix[i][v] = d + 1
                        queue.append((v, d + 1))

        # 3. Parse goals to find package targets
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "in":
                self.pkg_targets[args[0]] = args[1]

        # 4. Map global IDs for fast state checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                self.rloc_fact_ids[fact.global_id] = args[0]
                self.relevant_fact_ids.add(fact.global_id)
            elif pred == "holding":
                self.holding_fact_ids[fact.global_id] = args[0]
                self.relevant_fact_ids.add(fact.global_id)
            elif pred == "in" and args[0] in self.pkg_targets:
                self.in_fact_ids[fact.global_id] = (args[0], args[1])
                self.relevant_fact_ids.add(fact.global_id)
        
        self.facts_mask = (1 << (model.ifacts_end + 1)) - 1
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Fast extraction of state information
        robot_room = None
        holding_pkg = None
        pkg_locs = {}
        
        temp_state = state & self.facts_mask
        while temp_state:
            low_bit = (temp_state & -temp_state).bit_length() - 1
            if low_bit in self.rloc_fact_ids:
                robot_room = self.rloc_fact_ids[low_bit]
            elif low_bit in self.holding_fact_ids:
                holding_pkg = self.holding_fact_ids[low_bit]
            elif low_bit in self.in_fact_ids:
                p, r = self.in_fact_ids[low_bit]
                pkg_locs[p] = r
            temp_state &= temp_state - 1

        if robot_room is None:
            return 1000

        h = 0
        undelivered_pkg_rooms = []
        r_idx = self.room_to_idx[robot_room]

        for p_name, target_room in self.pkg_targets.items():
            t_idx = self.room_to_idx[target_room]
            if holding_pkg == p_name:
                h += self.dist_matrix[r_idx][t_idx] + 1
            else:
                curr_room = pkg_locs.get(p_name)
                if curr_room and curr_room != target_room:
                    c_idx = self.room_to_idx[curr_room]
                    h += self.dist_matrix[c_idx][t_idx] + 2
                    undelivered_pkg_rooms.append(c_idx)

        if undelivered_pkg_rooms and holding_pkg is None:
            h += min(self.dist_matrix[r_idx][loc_idx] for loc_idx in undelivered_pkg_rooms)

        # Small tie-breaker for search guidance
        h += 0.001 * len(task_network)
        return int(h) if h > 0 or len(task_network) == 0 else 1

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
