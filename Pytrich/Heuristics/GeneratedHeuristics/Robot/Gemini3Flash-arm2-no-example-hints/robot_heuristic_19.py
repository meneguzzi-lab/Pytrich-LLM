from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic19(Heuristic):
    """
    Robot Domain Heuristic
    
    Bottleneck Analysis:
    The primary bottleneck is the single-carry constraint. The robot can only hold one package at a time.
    This serializes all deliveries, meaning total cost is at least the sum of (pickup + carry + putdown) 
    for all undelivered packages, plus the travel between goal locations and the next package.

    Admissibility Sketches:
    1. Delivery Cost: Each undelivered package requires at least a pickup (1) and a putdown (1) action.
       If already holding the package, only the putdown (1) is required. This is strictly admissible.
    2. Carry Cost: Each package must be moved from its current room to its goal room. The shortest 
       path (BFS distance) in the room graph is a lower bound on the move actions required.
    3. Navigation Cost: The robot must move to the nearest undelivered package or its current 
       held package's goal. BFS distance is a lower bound.
    4. Door Cost: Each closed door on the current active path must be opened (1 action).
    """

    def __init__(self, robot_gemini3flash_heuristic_19="robot_gemini3flash_heuristic_19"):
        super().__init__(robot_gemini3flash_heuristic_19)
        self.pkg_goal = {}
        self.rooms = []
        self.packages = []
        self.doors = []
        self.rloc_ids = {}
        self.holding_ids = {}
        self.in_ids = {} # (pkg, room) -> id
        self.door_closed_id = {}
        self.adj = {}
        self.dist = {}
        self.path_doors = {}
        self.pkg_in_fact_ids = {}

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
        room_set = set()
        pkg_set = set()
        door_set = set()

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'rloc':
                room_set.add(args[0])
                self.rloc_ids[args[0]] = f.global_id
            elif pred == 'holding':
                pkg_set.add(args[0])
                self.holding_ids[args[0]] = f.global_id
            elif pred == 'in':
                pkg_set.add(args[0])
                room_set.add(args[1])
                self.in_ids[(args[0], args[1])] = f.global_id
            elif pred == 'door':
                room_set.add(args[0])
                room_set.add(args[1])
                door_set.add(args[2])
            elif pred == 'closed':
                door_set.add(args[0])
                self.door_closed_id[args[0]] = f.global_id
            elif pred == 'goal_in':
                pkg_set.add(args[0])
                room_set.add(args[1])
                self.pkg_goal[args[0]] = args[1]

        self.rooms = list(room_set)
        self.packages = list(pkg_set)
        self.doors = list(door_set)

        # Re-parse door facts for adjacency after collecting all rooms
        self.adj = {r: [] for r in self.rooms}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'door':
                self.adj[args[0]].append((args[1], args[2]))

        # BFS for all-pairs distances
        for start in self.rooms:
            queue = deque([(start, 0, [])])
            visited = {start}
            self.dist[(start, start)] = 0
            self.path_doors[(start, start)] = []
            while queue:
                curr, d, p_doors = queue.popleft()
                for neighbor, door_name in self.adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = d + 1
                        new_doors = p_doors + [door_name]
                        self.dist[(start, neighbor)] = new_dist
                        self.path_doors[(start, neighbor)] = new_doors
                        queue.append((neighbor, new_dist, new_doors))

        # Pre-group 'in' facts for efficient searching
        for p in self.packages:
            self.pkg_in_fact_ids[p] = []
            for r in self.rooms:
                if (p, r) in self.in_ids:
                    self.pkg_in_fact_ids[p].append((r, self.in_ids[(p, r)]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        if not self.pkg_goal:
            return 0

        # 1. Identify robot location
        curr_r = None
        for r in self.rooms:
            if (state >> self.rloc_ids[r]) & 1:
                curr_r = r
                break
        if curr_r is None: return 0

        # 2. Identify undelivered packages and held package
        undelivered = []
        held_p = None
        for p in self.packages:
            g_room = self.pkg_goal.get(p)
            if g_room:
                is_at_goal = (state >> self.in_ids.get((p, g_room), 0)) & 1
                is_held = (state >> self.holding_ids.get(p, 0)) & 1
                if not is_at_goal:
                    undelivered.append(p)
                    if is_held: held_p = p

        if not undelivered: return 0

        # 3. Sum delivery costs
        h = 0
        pkg_positions = {}
        
        for p in undelivered:
            g_room = self.pkg_goal[p]
            if p == held_p:
                h += 1 + self.dist.get((curr_r, g_room), 0)
                for d_name in self.path_doors.get((curr_r, g_room), []):
                    d_id = self.door_closed_id.get(d_name)
                    if d_id is not None and (state >> d_id) & 1:
                        h += 1
            else:
                # Find current room
                pos_p = None
                for r, f_id in self.pkg_in_fact_ids[p]:
                    if (state >> f_id) & 1:
                        pos_p = r
                        break
                if pos_p:
                    pkg_positions[p] = pos_p
                    h += 2 + self.dist.get((pos_p, g_room), 0)
        
        # 4. Navigation cost to first package
        if held_p is None and undelivered:
            min_nav = 1000000
            best_p = None
            for p, pos_p in pkg_positions.items():
                d = self.dist.get((curr_r, pos_p), 0)
                if d < min_nav:
                    min_nav = d
                    best_p = p
            h += min_nav
            # Add door cost for the first navigation path
            if best_p:
                for d_name in self.path_doors.get((curr_r, pkg_positions[best_p]), []):
                    d_id = self.door_closed_id.get(d_name)
                    if d_id is not None and (state >> d_id) & 1:
                        h += 1

        # 5. Tie-breaker for task network length
        h += 0.001 * len(task_network)

        return int(h) if h < 1000000 else 1000

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
