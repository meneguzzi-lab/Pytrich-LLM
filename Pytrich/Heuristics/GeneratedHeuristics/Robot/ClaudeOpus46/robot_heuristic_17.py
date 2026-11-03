# Robot Domain Heuristic
# Bottleneck: Single arm serializes all package delivery. Each misplaced package
# requires pickup + travel + putdown. Doors may need opening.
#
# Admissible lower bounds:
# T1: Sum of (2 + shortest_path(src, dst)) for each misplaced package, divided by
#     max parallelism (which is 1 since single arm), but we use max single-package cost
#     plus total pickup/putdown count as a composite.
# T2: Number of misplaced packages * 2 (pickup + putdown actions minimum)
# T3: For the package requiring most travel, its full cost is a lower bound since
#     the arm is serial.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic17(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_17"):
        super().__init__(name)
        self._initialized = False

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
        self._initialized = True

        # Build fact maps
        self.in_fact_ids = {}       # (package, room) -> global_id
        self.goal_in_facts = {}     # (package, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # package -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door_name

        # All rooms and packages
        self.all_rooms = set()
        self.all_packages = set()
        self.all_doors = set()

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                self.in_fact_ids[(args[0], args[1])] = f.global_id
                self.all_packages.add(args[0])
                self.all_rooms.add(args[1])
            elif pred == "goal_in":
                self.goal_in_facts[(args[0], args[1])] = f.global_id
            elif pred == "rloc":
                self.rloc_fact_ids[args[0]] = f.global_id
                self.all_rooms.add(args[0])
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = f.global_id
                self.all_packages.add(args[0])
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "closed":
                self.closed_fact_ids[args[0]] = f.global_id
                self.all_doors.add(args[0])
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                self.door_facts[(r1, r2)] = d
                self.all_rooms.add(r1)
                self.all_rooms.add(r2)

        # Build adjacency for rooms (ignoring door state - optimistic)
        self.room_adj = {}
        for r in self.all_rooms:
            self.room_adj[r] = set()
        for (r1, r2), d in self.door_facts.items():
            self.room_adj[r1].add(r2)
            # r2->r1 should also exist as separate door fact

        # Precompute BFS shortest distances between all pairs of rooms
        # Distance counts number of move actions (ignoring open actions - optimistic)
        self.room_list = sorted(self.all_rooms)
        self.room_index = {r: i for i, r in enumerate(self.room_list)}
        n = len(self.room_list)
        self.dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            self.dist[i][i] = 0

        for r in self.room_list:
            ri = self.room_index[r]
            # BFS from r
            visited = {r}
            queue = deque([(r, 0)])
            while queue:
                cur, d = queue.popleft()
                ci = self.room_index[cur]
                self.dist[ri][ci] = d
                for nb in self.room_adj.get(cur, set()):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, d + 1))

        # Also precompute distances considering doors need opening
        # Each closed door on the path adds 1 action (open)
        # But for admissibility, the optimistic (no-open) distance is fine as lower bound

        # Parse goal facts to find which packages need to go where
        # goal_in predicates tell us the target
        self.package_goals = {}  # package -> goal_room
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "in":
                self.package_goals[args[0]] = args[1]

        # Also build from goal_in facts for quick lookup
        self.goal_in_map = {}  # package -> goal_room (from goal_in predicates)
        for (pkg, room), gid in self.goal_in_facts.items():
            # Check if this goal_in is in initial state (it's static)
            self.goal_in_map[pkg] = room

        # Build adjacency with door info for weighted BFS
        # We'll precompute distances that include minimum door-opening costs
        # For a more informed heuristic, compute distances accounting for closed doors
        # But closed doors change during execution, so we use state-dependent computation

        # Precompute: for each pair of adjacent rooms, store the door
        self.room_door = {}  # (r1, r2) -> door_name
        for (r1, r2), d in self.door_facts.items():
            self.room_door[(r1, r2)] = d

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        """Get current robot location from state using bit tests."""
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_holding(self, state):
        """Get package currently being held, if any."""
        for pkg, gid in self.holding_fact_ids.items():
            if (state >> gid) & 1:
                return pkg
        return None

    def _is_arm_empty(self, state):
        if self.armempty_id is not None:
            return (state >> self.armempty_id) & 1
        return True

    def _room_dist(self, r1, r2):
        """Get precomputed shortest distance between two rooms."""
        if r1 == r2:
            return 0
        i1 = self.room_index.get(r1)
        i2 = self.room_index.get(r2)
        if i1 is None or i2 is None:
            return float('inf')
        return self.dist[i1][i2]

    def _shortest_path_rooms(self, r1, r2, state):
        """BFS shortest path considering closed doors need opening (state-dependent).
        Returns number of actions (moves + opens).
        For performance, we use the precomputed optimistic distance.
        """
        # Use precomputed distance as base, it's admissible
        return self._room_dist(r1, r2)

    def _count_misplaced(self, state):
        """Count packages not at their goal location."""
        count = 0
        for pkg, goal_room in self.package_goals.items():
            fact_id = self.in_fact_ids.get((pkg, goal_room))
            if fact_id is not None:
                if not ((state >> fact_id) & 1):
                    count += 1
            else:
                # Package can never be at goal? Count as misplaced
                count += 1
        return count

    def _compute(self, state, task_network):
        if not self._initialized:
            return 0

        robot_loc = self._get_robot_location(state)
        if robot_loc is None:
            return 0

        holding_pkg = self._get_holding(state)
        arm_empty = self._is_arm_empty(state)

        # Identify misplaced packages
        misplaced = []  # (package, current_room, goal_room)
        for pkg, goal_room in self.package_goals.items():
            fact_id = self.in_fact_ids.get((pkg, goal_room))
            if fact_id is not None:
                if (state >> fact_id) & 1:
                    continue  # Already at goal

            # Package is misplaced - find where it is
            if pkg == holding_pkg:
                # Robot is holding it, need to deliver
                misplaced.append((pkg, None, goal_room))  # None means held
            else:
                # Find current location
                current_room = None
                for (p, r), fid in self.in_fact_ids.items():
                    if p == pkg and (state >> fid) & 1:
                        current_room = r
                        break
                if current_room is not None:
                    if current_room != goal_room:
                        misplaced.append((pkg, current_room, goal_room))

        if not misplaced:
            # Check task network - if there are remaining tasks, small cost
            if task_network:
                return len(task_network)
            return 0

        num_misplaced = len(misplaced)

        # Term 1: Each misplaced package needs at least pickup(1) + putdown(1) = 2 actions
        # If held, just putdown(1) + travel to goal
        # This is clearly admissible.
        term1 = 0
        for pkg, cur_room, goal_room in misplaced:
            if cur_room is None:  # held
                term1 += 1  # putdown
            else:
                term1 += 2  # pickup + putdown

        # Term 2: Maximum single-package delivery cost
        # Since robot has single arm, it must deliver packages one at a time.
        # The cost for each package includes: travel_to_pkg + pickup + travel_to_goal + putdown
        # The max of these is a lower bound (admissible because at least that much work is needed
        # for that one package, and it can't be parallelized).
        max_single_cost = 0
        for pkg, cur_room, goal_room in misplaced:
            if cur_room is None:  # held
                cost = self._room_dist(robot_loc, goal_room) + 1  # travel + putdown
            else:
                travel_to_pkg = self._room_dist(robot_loc, cur_room)
                travel_to_goal = self._room_dist(cur_room, goal_room)
                cost = travel_to_pkg + 1 + travel_to_goal + 1  # go + pickup + go + putdown
            if cost > max_single_cost:
                max_single_cost = cost
        term2 = max_single_cost

        # Term 3: Serial delivery lower bound
        # Since arm is single, packages must be delivered one at a time.
        # Lower bound: sum of (pickup + putdown) for all misplaced + minimum total travel
        # For travel: at minimum, robot must visit all source rooms and all dest rooms.
        # A simple admissible bound: sum of min(travel_to_src, travel_to_dst) won't work well.
        # Instead: total_actions >= num_misplaced * 2 (pickup+putdown each)
        #   + sum of dist(src_i, dst_i) for each package (travel while carrying)
        # The sum of carrying distances is admissible because each package must be carried.
        carry_distances = 0
        for pkg, cur_room, goal_room in misplaced:
            if cur_room is None:  # held
                carry_distances += self._room_dist(robot_loc, goal_room)
            else:
                carry_distances += self._room_dist(cur_room, goal_room)

        term3 = term1 + carry_distances  # pickup/putdown actions + carrying travel

        # Term 4: Greedy serial estimate (admissible variant)
        # The robot delivers packages one by one. A lower bound:
        # For the currently held package: travel_to_goal + putdown
        # For all others: pickup + carry + putdown (minimum 2 + carry_dist each)
        # Plus: the robot must travel between deliveries (deadhead travel)
        # Lower bound on deadhead: 0 (packages could be collocated)
        # So term4 = sum of per-package costs, which double-counts robot position
        # but each package's carry distance is independent and necessary.
        # Actually term3 already captures this.

        # Term 5: Closed doors that must be opened
        # Count closed doors on shortest paths - but this is expensive to compute per call.
        # Instead, count doors that are on critical paths.
        # For admissibility, we skip this and rely on other terms.

        # Term 6: Task network lower bound
        # Count primitive tasks and estimate abstract task minimum decomposition costs
        tn_cost = 0
        if task_network:
            from Pytrich.model import Operator as OpClass
            for task in task_network:
                if isinstance(task, OpClass):
                    tn_cost += 1
                # Abstract tasks: minimum 0 (could be finished/noop)
                # But achieve-goals with misplaced packages needs work

        h = max(term1, term2, term3, tn_cost)

        # Tie-breaking: slight preference for states closer to having packages at goals
        # and for having the arm empty (ready for next pickup)
        tie_break = 0.0
        if not arm_empty and holding_pkg:
            # Holding something - check if we're at the goal
            goal_r = self.package_goals.get(holding_pkg)
            if goal_r and robot_loc != goal_r:
                tie_break += 0.001 * self._room_dist(robot_loc, goal_r)

        return h + tie_break

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
