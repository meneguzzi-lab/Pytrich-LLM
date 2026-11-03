from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic5(Heuristic):
    """
    Heuristic for the Robot HDDL domain.
    
    BOTTLENECK: Single-carry constraint. The robot can only carry one package at a time.
    This means every package delivery (pickup, travel to goal, putdown) is essentially
    a disjoint sequence of actions that must be performed serially.
    
    ADMISSIBILITY SKETCHES:
    1. Travel Cost: Sum of shortest-path distances from each undelivered package's 
       start room to its goal room is admissible because these movements are mandatory 
       and cannot be shared due to the single-carry constraint.
    2. Pickup/Putdown Cost: Each undelivered package requires exactly one pickup and 
       one putdown action (+2 cost per package).
    3. Robot Transition: The distance from the robot's current location to the first 
       package (or its goal if holding one) is an additional unavoidable cost.
    4. Door Opening: If a room is only accessible through a specific door, and that door 
       is closed, an 'open' action (+1) is mandatory if any package must enter or exit that room.
    """

    def __init__(self, robot_gemini3flash_heuristic_5="robot_gemini3flash_heuristic_5"):
        super().__init__(robot_gemini3flash_heuristic_5)
        self.package_goal_room = {}  # pkg_id -> room_id
        self.room_dist = {}          # r1 -> r2 -> distance
        self.mandatory_door = {}     # room_id -> door_name
        self.door_to_closed_id = {}  # door_name -> global_id
        self.pkg_ids = []
        self.room_ids = []
        self.fact_map = {}

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
        # 1. Map predicates and constants
        rooms = set()
        packages = set()
        doors = set()
        adj = {}
        initial_state = initial_node.state

        # Parse all facts for structural info
        for f_id in range(model.ifacts_init, model.ifacts_end + 1):
            fact = model.get_component(f_id)
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "rloc":
                rooms.add(args[0])
            elif pred == "in":
                packages.add(args[0])
                rooms.add(args[1])
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                rooms.add(r1); rooms.add(r2); doors.add(d)
                if r1 not in adj: adj[r1] = []
                adj[r1].append((r2, d))
            elif pred == "closed":
                self.door_to_closed_id[args[0]] = fact.global_id
            elif pred == "goal_in":
                self.package_goal_room[args[0]] = args[1]
            
            self.fact_map[fact.name] = fact.global_id

        # 2. Compute All-Pairs Shortest Paths (APSP)
        self.room_ids = list(rooms)
        self.pkg_ids = list(packages)
        for r1 in self.room_ids:
            self.room_dist[r1] = {r2: 1000 for r2 in self.room_ids}
            self.room_dist[r1][r1] = 0
            queue = deque([r1])
            while queue:
                u = queue.popleft()
                for v, d in adj.get(u, []):
                    if self.room_dist[r1][v] == 1000:
                        self.room_dist[r1][v] = self.room_dist[r1][u] + 1
                        queue.append(v)
        
        # 3. Identify mandatory doors (star topology optimization)
        for r in self.room_ids:
            if r in adj and len(adj[r]) == 1:
                self.mandatory_door[r] = adj[r][0][1]

        # Cache some common fact IDs for fast access in __call__
        self.armempty_id = self.fact_map.get("+armempty[]", -1)
        self.pkg_in_room_ids = {p: {r: self.fact_map.get(f"+in[{p},{r}]", -1) for r in rooms} for p in packages}
        self.holding_ids = {p: self.fact_map.get(f"+holding[{p}]", -1) for p in packages}
        self.rloc_ids = {r: self.fact_map.get(f"+rloc[{r}]", -1) for r in rooms}

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _compute(self, state, task_network):
        # Identify Robot state
        rloc = None
        for r in self.room_ids:
            if (state >> self.rloc_ids[r]) & 1:
                rloc = r
                break
        
        is_armempty = (state >> self.armempty_id) & 1 if self.armempty_id != -1 else False
        
        held_pkg = None
        for p in self.pkg_ids:
            if (state >> self.holding_ids[p]) & 1:
                held_pkg = p
                break

        # Identify Undelivered Packages
        undelivered = []
        pkg_locs = {}
        for p in self.pkg_ids:
            if p == held_pkg:
                undelivered.append(p)
            else:
                goal = self.package_goal_room.get(p)
                for r in self.room_ids:
                    if (state >> self.pkg_in_room_ids[p][r]) & 1:
                        pkg_locs[p] = r
                        if r != goal:
                            undelivered.append(p)
                        break

        if not undelivered and not held_pkg:
            return 0

        # Calculate Cost Summation
        h = 0
        needed_rooms = set()
        
        if held_pkg:
            goal = self.package_goal_room.get(held_pkg)
            h += self.room_dist[rloc][goal] + 1
            needed_rooms.add(rloc)
            needed_rooms.add(goal)
            for p in undelivered:
                if p == held_pkg: continue
                start, goal = pkg_locs[p], self.package_goal_room[p]
                h += self.room_dist[start][goal] + 2
                needed_rooms.add(start)
                needed_rooms.add(goal)
        else:
            # Robot must move to the nearest undelivered package
            min_move = 1000
            for p in undelivered:
                start, goal = pkg_locs[p], self.package_goal_room[p]
                h += self.room_dist[start][goal] + 2
                min_move = min(min_move, self.room_dist[rloc][start])
                needed_rooms.add(start)
                needed_rooms.add(goal)
            h += min_move if min_move != 1000 else 0

        # Closed Door Cost (Mandatory doors)
        for r in needed_rooms:
            door = self.mandatory_door.get(r)
            if door:
                closed_id = self.door_to_closed_id.get(door)
                if closed_id is not None and (state >> closed_id) & 1:
                    h += 1

        # Tie-breaker for progress
        h += 0.001 * len(task_network)
        
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
