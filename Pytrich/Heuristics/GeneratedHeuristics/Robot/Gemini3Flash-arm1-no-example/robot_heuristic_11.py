from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque, defaultdict

class RobotGemini3FlashHeuristic11(Heuristic):
    """
    Robot Domain Bottleneck: Single-arm capacity sequentially serializes work.
    Every package must be picked up and delivered one at a time.
    
    Admissibility Sketches:
    - Action Cost: Every package not at its target must be put down (1 action).
    - Action Cost: Every package not at its target and not held must be picked up (1 action).
    - Move Cost: For every package p, the robot must traverse dist(loc[p], target[p]).
    - These are all mandatory, disjoint actions for a single-arm robot.
    - Connecting Cost: At least min(dist(robot, loc[first_p])) is required to start.
    """
    def __init__(self, robot_gemini3flash_heuristic_11="RobotGemini3FlashHeuristic11"):
        super().__init__(robot_gemini3flash_heuristic_11)
        self.packages = set()
        self.rooms = set()
        self.target_rooms = {}
        self.holding_ids = {}
        self.in_ids = defaultdict(dict)
        self.rloc_ids = {}
        self.adj = defaultdict(set)
        self.distances = defaultdict(lambda: defaultdict(lambda: 999999))

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
        self.model = model
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                pkg, room = args[0], args[1]
                self.packages.add(pkg)
                self.rooms.add(room)
                self.in_ids[pkg][room] = f.global_id
            elif pred == "holding":
                pkg = args[0]
                self.packages.add(pkg)
                self.holding_ids[pkg] = f.global_id
            elif pred == "rloc":
                room = args[0]
                self.rooms.add(room)
                self.rloc_ids[room] = f.global_id
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                self.rooms.add(r1)
                self.rooms.add(r2)
                self.adj[r1].add(r2)
            elif pred == "goal_in":
                if (model.initial_state >> f.global_id) & 1:
                    self.target_rooms[args[0]] = args[1]
        for start_node in self.rooms:
            self.distances[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for v in self.adj[u]:
                    if self.distances[start_node][v] == 999999:
                        self.distances[start_node][v] = self.distances[start_node][u] + 1
                        queue.append(v)
        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        robot_room = None
        for room, rid in self.rloc_ids.items():
            if (state >> rid) & 1: 
                robot_room = room
                break
        if not robot_room: return 0
        todo_packages = []
        held_pkg = None
        for pkg in self.packages:
            if (state >> self.holding_ids[pkg]) & 1:
                held_pkg = pkg
                todo_packages.append((pkg, None))
            else:
                target = self.target_rooms.get(pkg)
                if target:
                    in_target_id = self.in_ids[pkg].get(target)
                    if in_target_id is not None and not ((state >> in_target_id) & 1):
                        for r, in_id in self.in_ids[pkg].items():
                            if (state >> in_id) & 1:
                                todo_packages.append((pkg, r))
                                break
        if not todo_packages: return 0
        h = 0
        if held_pkg:
            h += 1 + self.distances[robot_room][self.target_rooms[held_pkg]]
            for p, r in todo_packages:
                if p == held_pkg: continue
                h += 2 + self.distances[r][self.target_rooms[p]]
        else:
            min_start_dist = 999999
            for p, r in todo_packages:
                h += 2 + self.distances[r][self.target_rooms[p]]
                d = self.distances[robot_room][r]
                if d < min_start_dist: min_start_dist = d
            h += min_start_dist
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
