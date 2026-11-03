from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic17(Heuristic):
    """
    Heuristic for the Robot delivery domain.
    
    Bottleneck: Single-carry constraint. The robot can only hold one package at a time, 
    forcing a sequential pickup-move-putdown workflow for every delivery.

    Admissible Terms:
    1. Action Cost: Every undelivered package requires 1 pickup and 1 putdown (2 actions). 
       If currently holding a package, that package only needs its putdown (1 action).
    2. Navigation Cost (Carrying): Each package must be moved from its current location to its goal location. 
       The shortest path distance (number of moves) is a lower bound.
    3. Navigation Cost (Empty): If not holding a package, the robot must move to the next package's location.
       If holding, it must move to the current package's goal.
    
    Admissibility Sketch:
    Each term represents a disjoint or unavoidable set of actions. Pickups and putdowns are mandatory. 
    The robot must traverse the shortest path between start and goal locations for every package 
    because it can only move one at a time. The sum is a lower bound on the total plan length.
    """

    def __init__(self, robot_gemini3flash_heuristic_17="RobotGemini3FlashHeuristic17"):
        super().__init__(robot_gemini3flash_heuristic_17)
        self.rooms = []
        self.packages = []
        self.package_goal_room = {}
        self.package_start_room = {}
        self.fact_in = {}  # pkg -> room -> global_id
        self.fact_holding = {}  # pkg -> global_id
        self.fact_rloc = {}  # room -> global_id
        self.dist = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse objects
        self.rooms = []
        self.packages = []
        self.adj = {}

        # Identify rooms and packages from facts to ensure we have the right names
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                if args[0] not in self.rooms: self.rooms.append(args[0])
            elif pred == "holding":
                if args[0] not in self.packages: self.packages.append(args[0])
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                if r1 not in self.adj: self.adj[r1] = []
                self.adj[r1].append(r2)

        # Mapping fact IDs for O(1) checks in __call__
        self.fact_in = {p: {} for p in self.packages}
        self.fact_holding = {}
        self.fact_rloc = {}

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in":
                p, r = args[0], args[1]
                if p in self.fact_in: self.fact_in[p][r] = fact.global_id
            elif pred == "holding":
                self.fact_holding[args[0]] = fact.global_id
            elif pred == "rloc":
                self.fact_rloc[args[0]] = fact.global_id
            elif pred == "goal_in":
                # This predicate defines the goal location for a package
                # It is usually static in the initial state
                if (model.initial_state >> fact.global_id) & 1:
                    self.package_goal_room[args[0]] = args[1]

        # Determine start rooms from initial state
        for p in self.packages:
            for r in self.rooms:
                if r in self.fact_in[p] and (model.initial_state >> self.fact_in[p][r]) & 1:
                    self.package_start_room[p] = r
                    break

        # Precompute All-Pairs Shortest Paths via BFS
        self.dist = {r: {other: 1000 for other in self.rooms} for r in self.rooms}
        for start_r in self.rooms:
            self.dist[start_r][start_r] = 0
            if start_r not in self.adj: continue
            queue = deque([(start_r, 0)])
            visited = {start_r}
            while queue:
                curr, d = queue.popleft()
                self.dist[start_r][curr] = d
                for neighbor in self.adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # 1. Identify robot location and held package
        robot_room = None
        for r in self.rooms:
            if (state >> self.fact_rloc[r]) & 1:
                robot_room = r
                break
        if not robot_room: return 0

        held_p = None
        for p in self.packages:
            if (state >> self.fact_holding[p]) & 1:
                held_p = p
                break

        # 2. Identify undelivered packages (not at their target room)
        undelivered = []
        for p in self.packages:
            goal_r = self.package_goal_room.get(p)
            if goal_r:
                # If not in the goal room, it's undelivered
                if not ((state >> self.fact_in[p][goal_r]) & 1):
                    undelivered.append(p)
        
        if not undelivered:
            return 0

        # 3. Calculate Actions and Movements
        # Base action cost: 2 per undelivered (pickup + putdown)
        action_cost = 2 * len(undelivered)
        if held_p: action_cost -= 1 # Pickup already done

        # Movement cost: robot must carry each package from its current loc to goal loc
        move_cost = 0
        for p in undelivered:
            goal_r = self.package_goal_room[p]
            if p == held_p:
                move_cost += self.dist[robot_room][goal_r]
            else:
                # If not holding and not at goal, it must be at its start location (or previous drop loc)
                # Simplified: use current room if known, otherwise start room
                p_room = self.package_start_room.get(p)
                for r in self.rooms:
                    if r in self.fact_in[p] and (state >> self.fact_in[p][r]) & 1:
                        p_room = r
                        break
                move_cost += self.dist[p_room][goal_r]

        # Inter-delivery movement: If not holding, robot must move to the closest undelivered package
        if not held_p:
            min_to_pickup = 1000
            for p in undelivered:
                p_room = self.package_start_room.get(p)
                for r in self.rooms:
                    if r in self.fact_in[p] and (state >> self.fact_in[p][r]) & 1:
                        p_room = r
                        break
                min_to_pickup = min(min_to_pickup, self.dist[robot_room][p_room])
            move_cost += min_to_pickup

        # Tie-breaker: minor penalty for task network complexity to prefer deeper/closer states
        tie_breaker = 0.001 * len(task_network)

        return int(action_cost + move_cost + tie_breaker)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return predicate, args
        return None, []
