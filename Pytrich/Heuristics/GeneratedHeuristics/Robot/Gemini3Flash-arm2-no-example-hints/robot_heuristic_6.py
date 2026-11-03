from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic6(Heuristic):
    """
    Robot Domain Heuristic

    Bottleneck: Single-carry constraint. The robot can only transport one package at a time.
    Each delivery is a serial sequence of: move to package, pickup, move to goal, putdown.

    Admissibility Sketches:
    1. Action Lower Bound: Each undelivered package requires at least 2 actions (pickup/putdown).
       If currently holding a package, it needs at least 1 (putdown). These are independent 
       because of the single-carry limit.
    2. Movement Lower Bound: Each delivery requires at least dist(loc, goal) move actions.
       Since the robot can only carry one, these moves cannot be combined.
    3. Path Correction: The total cost includes the minimum distance from the robot's 
       current position to the first required pickup.
    """
    def __init__(self, robot_gemini3flash_heuristic_6="RobotGemini3FlashHeuristic6"):
        super().__init__(robot_gemini3flash_heuristic_6)
        self.pkg_to_goal_room = {}
        self.pkg_in_fact = {} 
        self.pkg_holding_fact = {}
        self.room_to_rloc_fact = {}
        self.dist = {}
        self.package_ids = []
        self.room_ids = []

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        adj = collections.defaultdict(list)
        rooms = set()
        packages = set()
        temp_goal_in = []

        for f in model.facts:
            pred, args, prefix = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "rloc":
                r = args[0]
                rooms.add(r)
                self.room_to_rloc_fact[r] = f.global_id
            elif pred == "in":
                p, r = args[0], args[1]
                packages.add(p)
                rooms.add(r)
                if p not in self.pkg_in_fact: self.pkg_in_fact[p] = {}
                self.pkg_in_fact[p][r] = f.global_id
            elif pred == "holding":
                p = args[0]
                packages.add(p)
                self.pkg_holding_fact[p] = f.global_id
            elif pred == "goal_in":
                p, r = args[0], args[1]
                temp_goal_in.append((p, r))
            elif pred == "door":
                r1, r2 = args[0], args[1]
                rooms.add(r1)
                rooms.add(r2)
                adj[r1].append(r2)

        self.package_ids = list(packages)
        self.room_ids = list(rooms)

        for p, r in temp_goal_in:
            self.pkg_to_goal_room[p] = r

        for r_start in self.room_ids:
            self.dist[r_start] = {r: 1000 for r in self.room_ids}
            self.dist[r_start][r_start] = 0
            queue = collections.deque([r_start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.dist[r_start][v] == 1000:
                        self.dist[r_start][v] = self.dist[r_start][u] + 1
                        queue.append(v)

        h_init = self._compute(initial_node.state)
        return super().initialize(model, h_init)

    def _compute(self, state):
        robot_room = None
        for r, f_id in self.room_to_rloc_fact.items():
            if (state >> f_id) & 1:
                robot_room = r
                break
        if robot_room is None: return 0

        h = 0
        undelivered = []
        curr_robot_pos = robot_room

        for p in self.package_ids:
            if (state >> self.pkg_holding_fact.get(p, 0)) & 1:
                g_room = self.pkg_to_goal_room.get(p)
                if g_room:
                    h += 1 + self.dist[robot_room].get(g_room, 0)
                    curr_robot_pos = g_room
            else:
                g_room = self.pkg_to_goal_room.get(p)
                if not g_room: continue
                at_goal_fact = self.pkg_in_fact.get(p, {}).get(g_room)
                if at_goal_fact is not None and (state >> at_goal_fact) & 1:
                    continue
                p_room = None
                for r, f_id in self.pkg_in_fact.get(p, {}).items():
                    if (state >> f_id) & 1:
                        p_room = r
                        break
                if p_room:
                    undelivered.append((p_room, g_room))

        if undelivered:
            min_d_to_pickup = 1000
            for p_room, g_room in undelivered:
                h += 2 + self.dist[p_room].get(g_room, 0)
                d = self.dist[curr_robot_pos].get(p_room, 1000)
                if d < min_d_to_pickup: min_d_to_pickup = d
            if min_d_to_pickup < 1000: h += min_d_to_pickup
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state)
        super().update_info(h_val)
        return h_val
