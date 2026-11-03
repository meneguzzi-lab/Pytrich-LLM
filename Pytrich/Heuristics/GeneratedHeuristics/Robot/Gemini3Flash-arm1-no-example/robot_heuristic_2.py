from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic2(Heuristic):
    """
    Domain-Dependent Heuristic for the Robot HDDL Domain.

    Bottleneck Analysis:
    The central bottleneck is that the robot can only carry one package at a time (armempty vs holding).
    Every package must be individually picked up, transported, and put down in its specific goal location.
    This creates a linear sequence of unavoidable actions for every misplaced package.

    Admissible Lower Bounds:
    1. h_packages: Every package not at its goal room requires at least a 'pickup' and a 'putdown' (2 actions).
       If a package is currently held, it requires at least a 'putdown' (1 action).
       Admissibility: pickup and putdown are unique actions that must occur for each package to reach its goal_in state.
    2. h_moves: If the robot is holding a package, it must travel to that package's goal room.
       The number of 'move' actions is at least the BFS distance between the current room and the goal room.
       Admissibility: Move actions are distinct from pickup/putdown actions and BFS distance is a true lower bound on travel.
    3. Tie-breaking: A tiny penalty per task in the task network helps guide the planner toward completing subtasks.
    """

    def __init__(self, robot_gemini3flash_heuristic_2="RobotGemini3FlashHeuristic2"):
        super().__init__(robot_gemini3flash_heuristic_2)
        self.package_to_goal = {}  # package_name -> goal_room_name
        self.in_ids = {}           # (package, room) -> global_id
        self.holding_ids = {}      # package -> global_id
        self.rloc_ids = {}         # room -> global_id
        self.room_to_idx = {}
        self.dist_matrix = []
        self.packages = []
        self.rooms = []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to map the domain objects
        rooms_set = set()
        packages_set = set()
        adj = collections.defaultdict(set)

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == "rloc":
                rooms_set.add(args[0])
                self.rloc_ids[args[0]] = fact.global_id
            elif pred == "in":
                packages_set.add(args[0])
                rooms_set.add(args[1])
                self.in_ids[(args[0], args[1])] = fact.global_id
            elif pred == "holding":
                packages_set.add(args[0])
                self.holding_ids[args[0]] = fact.global_id
            elif pred == "goal_in":
                # goal_in(p, r) is static information defining targets
                # Check if this specific fact is true in the initial state
                if (model.initial_state >> fact.global_id) & 1:
                    self.package_to_goal[args[0]] = args[1]
            elif pred == "door":
                # door(r1, r2, d)
                r1, r2 = args[0], args[1]
                rooms_set.add(r1)
                rooms_set.add(r2)
                adj[r1].add(r2)

        self.rooms = sorted(list(rooms_set))
        self.packages = sorted(list(packages_set))
        self.room_to_idx = {name: i for i, name in enumerate(self.rooms)}

        # 2. Precompute BFS Distance Matrix
        num_rooms = len(self.rooms)
        self.dist_matrix = [[float('inf')] * num_rooms for _ in range(num_rooms)]
        for i in range(num_rooms):
            self.dist_matrix[i][i] = 0
            start_node = self.rooms[i]
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            while queue:
                curr, d = queue.popleft()
                self.dist_matrix[i][self.room_to_idx[curr]] = d
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        # Initial h-value calculation
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        # 1. Identify current robot location
        curr_room_idx = -1
        for r_name in self.rooms:
            if (state >> self.rloc_ids[r_name]) & 1:
                curr_room_idx = self.room_to_idx[r_name]
                break

        # 2. Count misplaced and held packages
        num_misplaced = 0
        holding_p = None
        for p in self.packages:
            if (state >> self.holding_ids[p]) & 1:
                holding_p = p
            else:
                goal_r = self.package_to_goal.get(p)
                if goal_r:
                    in_goal_bit = self.in_ids.get((p, goal_r))
                    if in_goal_bit is not None and not ((state >> in_goal_bit) & 1):
                        num_misplaced += 1

        # 3. Calculate h terms
        # h_packages: 2 for each misplaced package (pickup + putdown), 1 for held (putdown)
        h_packages = (2 * num_misplaced) + (1 if holding_p else 0)

        # h_moves: distance to goal if holding, or 1 if misplaced packages exist elsewhere
        h_moves = 0
        if holding_p:
            goal_r = self.package_to_goal.get(holding_p)
            if goal_r and curr_room_idx != -1:
                h_moves = self.dist_matrix[curr_room_idx][self.room_to_idx[goal_r]]
        elif num_misplaced > 0:
            # At least one move is needed if we are not holding anything and something is misplaced
            h_moves = 1

        # 4. Total and Tie-breakers
        # len(task_network) * 0.001 is a small penalty to favor decomposition progress
        h_total = h_packages + h_moves + (0.001 * len(task_network))

        return int(h_total) if h_total < 1000000 else 1000000

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
