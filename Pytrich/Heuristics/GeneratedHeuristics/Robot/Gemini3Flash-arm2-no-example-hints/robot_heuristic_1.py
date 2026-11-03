from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic1(Heuristic):
    """
    Robot Domain Heuristic
    
    Bottleneck: Single-carry constraint. The robot can hold at most one package at a time.
    This creates a serial execution pattern where each delivery consists of a move-to-pickup,
    a pickup, a move-to-goal, and a putdown.
    
    Admissibility Sketch:
    1. Each undelivered package requires at least one 'pickup' and one 'putdown' action (Cost: 2).
       If a package is currently being held, it only requires one 'putdown' (Cost: 1).
    2. For every package p, the robot must traverse the distance from its current location to its goal.
       These 'move' actions are mandatory for transportation.
    3. The robot must also move to reach the first package or move from a goal location to the next package's pickup location.
       These 'positioning moves' are disjoint from transportation moves.
    4. Total cost = Pickups + Putdowns + Transporation Moves + Positioning Moves.
    """

    def __init__(self, robot_gemini3flash_heuristic_1="robot_gemini3flash_heuristic_1"):
        super().__init__(robot_gemini3flash_heuristic_1)
        self.goal_rooms = {}  # pkg -> room
        self.dist_matrix = {}
        self.pkg_in_ids = {}   # pkg -> {room: global_id}
        self.pkg_holding_ids = {} # pkg -> global_id
        self.rloc_ids = {}     # room -> global_id
        self.packages = []
        self.rooms = []

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
        # 1. Parse all facts to identify rooms and packages
        temp_rooms = set()
        temp_packages = set()
        adj = collections.defaultdict(list)

        # Identify predicates and build static maps
        for i in range(model.ifacts_init, model.ifacts_end + 1):
            fact = model.get_component(i)
            pred, args = self._parse_fact_name(fact.name)
            
            if pred == 'rloc':
                room = args[0]
                temp_rooms.add(room)
                self.rloc_ids[room] = fact.global_id
            elif pred == 'in':
                pkg, room = args[0], args[1]
                temp_packages.add(pkg)
                if pkg not in self.pkg_in_ids: self.pkg_in_ids[pkg] = {}
                self.pkg_in_ids[pkg][room] = fact.global_id
            elif pred == 'holding':
                pkg = args[0]
                temp_packages.add(pkg)
                self.pkg_holding_ids[pkg] = fact.global_id
            elif pred == 'door':
                # args: loc1, loc2, door
                adj[args[0]].append(args[1])
            
            # Static facts from initial state
            if (model.initial_state >> fact.global_id) & 1:
                if pred == 'goal_in':
                    self.goal_rooms[args[0]] = args[1]

        self.rooms = list(temp_rooms)
        self.packages = list(temp_packages)

        # 2. Precompute all-pairs shortest paths using BFS
        for start_room in self.rooms:
            self.dist_matrix[start_room] = self._bfs(start_room, adj)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _bfs(self, start, adj):
        distances = {r: 1000000 for r in self.rooms}
        distances[start] = 0
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if distances[v] == 1000000:
                    distances[v] = distances[u] + 1
                    queue.append(v)
        return distances

    def _compute(self, state, task_network):
        if not task_network and self.model.goal_reached(state, []):
            return 0

        # Find current robot location
        robot_room = None
        for r in self.rooms:
            if (state >> self.rloc_ids[r]) & 1:
                robot_room = r
                break
        
        if not robot_room: return 0

        # Find package state
        undelivered = [] # (pkg, curr_loc)
        holding_pkg = None

        for p in self.packages:
            if (state >> self.pkg_holding_ids[p]) & 1:
                holding_pkg = p
                continue
            
            # Where is the package?
            # Optimization: check the goal room first
            goal = self.goal_rooms.get(p)
            if goal:
                goal_fid = self.pkg_in_ids.get(p, {}).get(goal)
                if goal_fid is not None and (state >> goal_fid) & 1:
                    continue # Already at goal
                
                # Find where it is
                for r, fid in self.pkg_in_ids.get(p, {}).items():
                    if (state >> fid) & 1:
                        undelivered.append((p, r))
                        break

        h = 0
        current_pos = robot_room

        # Cost for currently held package
        if holding_pkg:
            target = self.goal_rooms.get(holding_pkg)
            if target:
                h += 1 # putdown
                h += self.dist_matrix[robot_room].get(target, 1000)
                current_pos = target
        
        # Cost for all other packages
        if undelivered:
            # 2 actions (pickup+putdown) + travel from start to goal for each
            for p, start_loc in undelivered:
                target = self.goal_rooms.get(p)
                h += 2
                if target:
                    h += self.dist_matrix[start_loc].get(target, 1000)
            
            # Positioning moves: robot must move to the first package
            # We take the minimum distance to any undelivered package as a lower bound
            dist_to_first = min(self.dist_matrix[current_pos].get(sl, 1000) for p, sl in undelivered)
            h += dist_to_first

            # Between deliveries, the robot must move from the previous goal to the next start.
            # Admissible bound: at least one move per additional package.
            h += len(undelivered) - (0 if holding_pkg else 1)

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
