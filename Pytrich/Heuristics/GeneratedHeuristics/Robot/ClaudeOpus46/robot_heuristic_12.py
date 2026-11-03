# Robot Domain Heuristic
# Bottleneck: Single-carry constraint — robot holds one package at a time.
# Each misplaced package requires pickup + putdown + travel.
#
# Admissible lower bounds:
# T1: 2 * misplaced_packages (each needs at least pickup + putdown)
# T2: Sum of shortest path distances for each misplaced package (from pkg room
#     to goal room), representing minimum travel. This is admissible because
#     the robot must traverse at least that distance for each package.
# T3: Task network structural lower bound — count of abstract tasks that
#     require nontrivial decomposition weighted by state-aware cost.
#
# We return max(T1, T2, T3) for the heuristic value.

from collections import deque

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotAnthropicOpus46Heuristic12(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_12"):
        super().__init__(name)
        self.in_fact_ids = {}       # (pkg, room) -> global_id
        self.goal_in_fact_ids = {}  # (pkg, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # pkg -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door
        
        # Package goal mapping: pkg -> goal_room
        self.pkg_goal_room = {}
        # Package set
        self.packages = set()
        # Room set
        self.rooms = set()
        # Adjacency for rooms (room -> set of (neighbor_room, door))
        self.room_adj = {}
        # BFS shortest distances between rooms (ignoring door states)
        self.room_dist = {}
        # All in facts for a package: pkg -> list of (room, global_id)
        self.pkg_in_facts = {}

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

    def _bfs_distances(self, start_room):
        """BFS from start_room to all other rooms, ignoring door states (optimistic)."""
        dist = {start_room: 0}
        queue = deque([start_room])
        while queue:
            r = queue.popleft()
            if r not in self.room_adj:
                continue
            for neighbor, door in self.room_adj[r]:
                if neighbor not in dist:
                    dist[neighbor] = dist[r] + 1
                    queue.append(neighbor)
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                pkg, room = args[0], args[1]
                self.in_fact_ids[(pkg, room)] = f.global_id
                self.packages.add(pkg)
                self.rooms.add(room)
                if pkg not in self.pkg_in_facts:
                    self.pkg_in_facts[pkg] = []
                self.pkg_in_facts[pkg].append((room, f.global_id))
            elif pred == "goal_in":
                pkg, room = args[0], args[1]
                self.goal_in_fact_ids[(pkg, room)] = f.global_id
                self.pkg_goal_room[pkg] = room
            elif pred == "rloc":
                room = args[0]
                self.rloc_fact_ids[room] = f.global_id
                self.rooms.add(room)
            elif pred == "holding":
                pkg = args[0]
                self.holding_fact_ids[pkg] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "closed":
                door = args[0]
                self.closed_fact_ids[door] = f.global_id
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                self.door_facts[(r1, r2)] = d
                if r1 not in self.room_adj:
                    self.room_adj[r1] = []
                self.room_adj[r1].append((r2, d))
                self.rooms.add(r1)
                self.rooms.add(r2)
        
        # Precompute BFS distances between all rooms
        for room in self.rooms:
            self.room_dist[room] = self._bfs_distances(room)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_pkg_current_room(self, pkg, state):
        """Find which room a package is currently in, or None if being held."""
        if pkg in self.pkg_in_facts:
            for room, fid in self.pkg_in_facts[pkg]:
                if (state >> fid) & 1:
                    return room
        return None  # Package is being held

    def _get_dist(self, r1, r2):
        if r1 == r2:
            return 0
        if r1 in self.room_dist and r2 in self.room_dist[r1]:
            return self.room_dist[r1][r2]
        return 100  # Large default if unreachable

    def _count_closed_doors_on_path_estimate(self, state):
        """Count closed doors that likely need opening. Simple: count all closed doors."""
        count = 0
        for door, fid in self.closed_fact_ids.items():
            if (state >> fid) & 1:
                count += 1
        return count

    def _compute(self, state, task_network):
        robot_loc = self._get_robot_location(state)
        if robot_loc is None:
            return 0
        
        # Check if robot is holding something
        held_pkg = None
        for pkg, fid in self.holding_fact_ids.items():
            if (state >> fid) & 1:
                held_pkg = pkg
                break
        
        misplaced = []
        for pkg, goal_room in self.pkg_goal_room.items():
            # Check if package is already at goal
            key = (pkg, goal_room)
            if key in self.in_fact_ids:
                fid = self.in_fact_ids[key]
                if (state >> fid) & 1:
                    continue  # Already at goal
            misplaced.append(pkg)
        
        num_misplaced = len(misplaced)
        
        if num_misplaced == 0:
            # Check task network size as secondary
            if task_network is not None and len(task_network) > 0:
                return len(task_network)
            return 0
        
        # T1: Each misplaced package needs at least pickup + putdown = 2 actions
        t1 = 2 * num_misplaced
        
        # T2: Travel-based lower bound
        # For each misplaced package, compute:
        #   - If held: distance from robot to goal room + 1 (putdown)
        #   - If not held: distance from current room to goal room (minimum travel to deliver)
        # Plus robot must visit each package location and each goal location.
        # 
        # A tighter bound: use TSP-like reasoning.
        # Admissible approximation: 
        #   For each misplaced package: pickup(1) + travel(src->dst) + putdown(1)
        #   But robot travels between deliveries, so we can't just sum.
        #   Lower bound: sum of max(1, dist(src, dst)) for each package (movement actions)
        #   Plus 2 * num_misplaced for pickup/putdown
        #   This overcounts robot travel, so we use a different approach.
        #
        # Actually admissible: for the package requiring the most travel,
        #   robot must go to it, pick up, travel to goal, put down.
        #   So at minimum: dist(robot, pkg_room) + dist(pkg_room, goal_room) + 2
        # For all packages combined:
        #   We need at least 2*num_misplaced (pickup+putdown) + sum of dist(pkg_room, goal_room)
        #   But this double-counts movement. 
        #   
        # Safe admissible: max over all packages of:
        #   dist(robot, pkg_room) + dist(pkg_room, goal_room) + 2
        # Also: sum of dist(pkg_room, goal_room) for all misplaced (admissible if paths don't overlap... not guaranteed)
        # Let's use the max single-package cost + 2*(remaining) as a compromise
        
        total_inner_dist = 0
        max_single_cost = 0
        
        for pkg in misplaced:
            goal_room = self.pkg_goal_room[pkg]
            if pkg == held_pkg:
                # Already picked up, just need to go to goal and putdown
                d = self._get_dist(robot_loc, goal_room)
                single_cost = d + 1  # travel + putdown
                inner_dist = 0
            else:
                pkg_room = self._get_pkg_current_room(pkg, state)
                if pkg_room is None:
                    # Shouldn't happen if not held and not found, treat as 0 dist
                    inner_dist = 0
                    single_cost = 2
                else:
                    inner_dist = self._get_dist(pkg_room, goal_room)
                    robot_to_pkg = self._get_dist(robot_loc, pkg_room)
                    single_cost = robot_to_pkg + inner_dist + 2  # go to pkg, pickup, travel, putdown
            
            total_inner_dist += inner_dist
            if single_cost > max_single_cost:
                max_single_cost = single_cost
        
        # T2a: The most expensive single delivery
        t2a = max_single_cost
        
        # T2b: All packages need internal travel + pickup/putdown
        # But only one package can use the robot at a time,
        # so sum of inner distances + 2*num_misplaced is a lower bound
        # (robot at least traverses each inner distance once and does pickup+putdown)
        t2b = total_inner_dist + 2 * num_misplaced
        
        t2 = max(t2a, t2b)
        
        # T3: Task network lower bound
        # Count tasks that will produce real actions
        t3 = 0
        if task_network is not None:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                # Check if it's an operator (primitive task)
                if hasattr(component, 'add_effects'):
                    t3 += 1
                elif hasattr(component, 'decompositions'):
                    # Abstract task - estimate minimum decomposition cost
                    task_name = component.name
                    if 'achieve-goals' in task_name:
                        # This task will eventually produce all remaining deliveries
                        # Don't double count - just add 0
                        pass
                    elif 'pickup_abstract' in task_name:
                        t3 += 1
                    elif 'putdown_abstract' in task_name:
                        t3 += 1
                    elif 'move_abstract' in task_name:
                        t3 += 1
                    elif 'open_abstract' in task_name:
                        t3 += 1
                    elif 'release' in task_name:
                        # Release involves at least putdown + achieve-goals
                        t3 += 1
        
        h = max(t1, t2, t3)
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
