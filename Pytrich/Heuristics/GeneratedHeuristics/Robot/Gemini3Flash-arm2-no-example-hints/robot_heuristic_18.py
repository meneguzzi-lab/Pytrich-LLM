from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

# BOTTLENECK: Single-carry constraint — robot can only hold one package at a time. Each delivery requires a disjoint set of actions (pickup, travel, putdown).
# ADMISSIBILITY SKETCH:
# 1. Action Count: Every package not at its goal room requires at least 1 pickup and 1 putdown (2 actions), or 1 putdown if already held. These are independent as the robot arm is a serial resource.
# 2. Travel Count: Robot must move between rooms to pick up and drop off packages. Sum of shortest-path distances (BFS) for each delivery is a lower bound because packages are moved one by one.
# 3. Door Count: Every closed door on a required shortest path must be opened at least once. Since doors stay open, the count of unique closed doors on required paths is admissible.

class RobotGemini3FlashHeuristic18(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_18="robot_heuristic_18"):
        super().__init__(robot_gemini3flash_heuristic_18)
        self.package_goal_room = {} # package -> room_name
        self.dist = {} # r1 -> r2 -> int
        self.doors_on_path = {} # r1 -> r2 -> set(door_names)
        self.packages = []
        
        self.in_bits = {} # (package, room) -> global_id
        self.holding_bits = {} # package -> global_id
        self.rloc_bits = {} # room -> global_id
        self.closed_bits = {} # door -> global_id
        
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
        rooms = set()
        all_packages = set()
        doors = set()
        adj = {}

        # 1. Parse initial state for static properties and fact mapping
        init_facts = model.state_explicit_repr(model.initial_state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "goal_in":
                self.package_goal_room[args[0]] = args[1]
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                rooms.add(r1)
                rooms.add(r2)
                doors.add(d)
                if r1 not in adj: adj[r1] = []
                adj[r1].append((r2, d))

        # 2. Build Fact ID Maps for fast bitwise checking
        for f_id in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(f_id)
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                self.in_bits[(args[0], args[1])] = f.global_id
                all_packages.add(args[0])
            elif pred == "holding":
                self.holding_bits[args[0]] = f.global_id
            elif pred == "rloc":
                self.rloc_bits[args[0]] = f.global_id
            elif pred == "closed":
                self.closed_bits[args[0]] = f.global_id

        self.packages = list(all_packages)
        self.rooms = list(rooms)

        # 3. Precompute BFS All-Pairs Shortest Paths and Doors
        for start_r in self.rooms:
            self.dist[start_r] = {r: 0 for r in self.rooms}
            self.doors_on_path[start_r] = {r: set() for r in self.rooms}
            queue = deque([(start_r, [])])
            visited = {start_r}
            while queue:
                curr_r, path_doors = queue.popleft()
                self.dist[start_r][curr_r] = len(path_doors)
                self.doors_on_path[start_r][curr_r] = set(path_doors)
                if curr_r in adj:
                    for neighbor_r, door_d in adj[curr_r]:
                        if neighbor_r not in visited:
                            visited.add(neighbor_r)
                            queue.append((neighbor_r, path_doors + [door_d]))

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int) -> int:
        # 1. Identify robot location
        robot_loc = None
        for rname, rid in self.rloc_bits.items():
            if (state >> rid) & 1:
                robot_loc = rname
                break
        if not robot_loc: return 0

        # 2. Track currently held package
        held_package = None
        for p in self.packages:
            if (state >> self.holding_bits[p]) & 1:
                held_package = p
                break

        # 3. Analyze work per package
        h = 0
        undelivered_locs = []
        required_doors = set()

        if held_package:
            goal_r = self.package_goal_room.get(held_package)
            if goal_r:
                h += 1 # putdown action
                h += self.dist[robot_loc].get(goal_r, 0)
                required_doors.update(self.doors_on_path[robot_loc].get(goal_r, []))

        for p in self.packages:
            if p == held_package: continue
            goal_r = self.package_goal_room.get(p)
            if not goal_r: continue
            
            # Check if package is at its goal room
            goal_bit = self.in_bits.get((p, goal_r))
            if goal_bit is not None and (state >> goal_bit) & 1:
                continue
            
            # Package needs delivery. Find its current location.
            current_loc = None
            for r in self.rooms:
                in_bit = self.in_bits.get((p, r))
                if in_bit is not None and (state >> in_bit) & 1:
                    current_loc = r
                    break
            
            if current_loc:
                h += 2 # pickup + putdown
                h += self.dist[current_loc].get(goal_r, 0)
                required_doors.update(self.doors_on_path[current_loc].get(goal_r, []))
                undelivered_locs.append(current_loc)

        # 4. Transition to nearest pickup if empty-handed
        if not held_package and undelivered_locs:
            min_trans = min(self.dist[robot_loc].get(loc, 0) for loc in undelivered_locs)
            h += min_trans

        # 5. Add closed door opening costs
        for dname in required_doors:
            closed_bit = self.closed_bits.get(dname)
            if closed_bit is not None and (state >> closed_bit) & 1:
                h += 1

        return h
