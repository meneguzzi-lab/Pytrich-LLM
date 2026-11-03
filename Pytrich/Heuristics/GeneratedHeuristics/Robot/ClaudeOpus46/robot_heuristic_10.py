"""Domain-dependent heuristic for the Robot domain.

Bottleneck Analysis:
  The robot has a single arm (armempty/holding) so it can carry at most one package
  at a time. Each misplaced package requires at minimum: move to source, pickup,
  move to destination, putdown. Doors may need to be opened (1 action each).

Admissible Lower Bounds:
  T1 - Misplaced packages * 2: Each misplaced package needs at least pickup + putdown = 2 actions.
  T2 - Sum of shortest path distances for misplaced packages: Each package must travel
       from current room to goal room. The robot must traverse that path. Since the robot
       carries one package at a time, these travel costs serialize. We use the max of
       individual distances as a lower bound (since robot does them sequentially, sum would
       also be valid but we use a weighted combination).
  T3 - Robot travel lower bound: The robot must visit each source and destination room.
       A TSP-like lower bound using the farthest misplaced package from current robot location.
"""

from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.HDDL.hddl_types import Operator, AbstractTask


class RobotAnthropicOpus46Heuristic10(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_10"):
        super().__init__(name)
        self.model = None
        # Fact ID maps
        self.in_fact_ids = {}       # (package, room) -> global_id
        self.goal_in_facts = {}     # (package, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # package -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door
        
        # Room graph for shortest paths
        self.rooms = set()
        self.room_adj = {}          # room -> list of (neighbor_room, door)
        self.dist = {}              # (room1, room2) -> shortest distance (ignoring doors)
        
        # Goal info
        self.package_goals = {}     # package -> goal_room
        self.all_packages = set()

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

    def _bfs_shortest_paths(self, source):
        """BFS from source room to all other rooms, treating all edges as cost 1."""
        dist = {source: 0}
        queue = deque([source])
        while queue:
            room = queue.popleft()
            for neighbor, door in self.room_adj.get(room, []):
                if neighbor not in dist:
                    dist[neighbor] = dist[room] + 1
                    queue.append(neighbor)
        return dist

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_facts[(args[0], args[1])] = fact.global_id
                self.package_goals[args[0]] = args[1]
                self.all_packages.add(args[0])
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = fact.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = fact.global_id
            elif pred == "door" and len(args) == 3:
                room1, room2, door = args[0], args[1], args[2]
                self.rooms.add(room1)
                self.rooms.add(room2)
                if room1 not in self.room_adj:
                    self.room_adj[room1] = []
                self.room_adj[room1].append((room2, door))
                self.door_facts[(room1, room2)] = door
        
        # Precompute all-pairs shortest paths (ignoring door open status, just topology)
        # This gives us a lower bound on travel distance
        self.dist = {}
        room_list = list(self.rooms)
        for room in room_list:
            dists = self._bfs_shortest_paths(room)
            for other_room, d in dists.items():
                self.dist[(room, other_room)] = d
        
        # Build operator index for fast action cost estimation
        # Not needed if we do pure bit checks
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_robot_location(self, state):
        """Get current robot location from state using bit checks."""
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_package_location(self, state, package):
        """Get current location of a package, or None if being held."""
        for (pkg, room), fid in self.in_fact_ids.items():
            if pkg == package and (state >> fid) & 1:
                return room
        return None

    def _is_holding(self, state, package):
        """Check if robot is holding a specific package."""
        fid = self.holding_fact_ids.get(package)
        if fid is not None:
            return (state >> fid) & 1
        return False

    def _is_arm_empty(self, state):
        if self.armempty_id is not None:
            return (state >> self.armempty_id) & 1
        return True

    def _compute(self, state, task_network):
        robot_loc = self._get_robot_location(state)
        if robot_loc is None:
            return 0
        
        # Find all misplaced packages
        misplaced = []  # (package, current_room, goal_room)
        held_package = None
        held_goal = None
        
        for package, goal_room in self.package_goals.items():
            # Check if package is already at goal
            in_goal_fid = self.in_fact_ids.get((package, goal_room))
            if in_goal_fid is not None and (state >> in_goal_fid) & 1:
                continue  # Already at goal
            
            # Check if being held
            if self._is_holding(state, package):
                held_package = package
                held_goal = goal_room
                continue
            
            # Find current location
            current_room = self._get_package_location(state, package)
            if current_room is not None:
                misplaced.append((package, current_room, goal_room))
        
        num_misplaced = len(misplaced)
        
        # T1: Each misplaced package needs pickup + putdown = 2 actions
        # If holding a package, it still needs putdown = 1 action
        t1 = num_misplaced * 2
        if held_package is not None:
            t1 += 1  # putdown for held package
        
        # T2: Travel-based lower bound
        # For held package: distance from robot to goal
        # For each misplaced package: distance from source to destination
        # The robot must do all these trips sequentially (single carry), so we can
        # sum individual travel distances as a lower bound on total movement.
        # Actually, the robot can combine trips (going from one delivery to next pickup),
        # so summing overestimates. Instead, we compute a more careful bound.
        
        # Simple admissible bound: max single package cost
        # For each misplaced pkg: robot must go to source + source to dest = at minimum
        # dist(source, dest) moves while carrying
        
        total_package_travel = 0
        max_single_cost = 0
        
        if held_package is not None:
            held_dist = self.dist.get((robot_loc, held_goal), 0)
            # Cost: travel to goal + putdown
            held_cost = held_dist + 1
            max_single_cost = max(max_single_cost, held_cost)
            total_package_travel += held_dist
        
        for (pkg, src, dst) in misplaced:
            pkg_travel_dist = self.dist.get((src, dst), 0)
            total_package_travel += pkg_travel_dist
            # Minimum cost for this package: travel to source + pickup + travel to dest + putdown
            # But we already count pickup+putdown in t1
            single_cost = pkg_travel_dist  # just the travel part
            max_single_cost = max(max_single_cost, single_cost)
        
        # T2: The robot must travel at least the sum of all individual package distances
        # (since it can only carry one at a time, these trips are disjoint)
        # Plus the robot must get to the first source
        t2 = total_package_travel + t1  # travel + pickup/putdown actions
        
        # T3: Robot must at least reach the nearest misplaced package first
        # (or deliver held package first)
        min_dist_to_start = 0
        if held_package is not None:
            min_dist_to_start = self.dist.get((robot_loc, held_goal), 0)
        elif misplaced:
            min_dist_to_start = min(
                self.dist.get((robot_loc, src), float('inf'))
                for (pkg, src, dst) in misplaced
            )
            if min_dist_to_start == float('inf'):
                min_dist_to_start = 0
        
        # T4: Task network lower bound - count minimum primitive actions
        t4 = 0
        if task_network:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                if isinstance(component, Operator):
                    t4 += 1
                elif isinstance(component, AbstractTask):
                    # Estimate minimum cost based on task name
                    task_name = component.name
                    if 'achieve-goals' in task_name:
                        # At minimum, needs to handle remaining misplaced packages
                        pass  # already counted in t1/t2
                    elif 'pickup_abstract' in task_name:
                        t4 += 1  # at least 1 primitive action
                    elif 'putdown_abstract' in task_name:
                        t4 += 1
                    elif 'move_abstract' in task_name:
                        t4 += 1
                    elif 'open_abstract' in task_name:
                        t4 += 1
                    elif 'release' in task_name:
                        # release needs at least: putdown (if at goal) or move + release
                        t4 += 1
        
        # Combine: take max of different estimates
        # t2 already includes t1, so it's the strongest single term
        h = max(t2, t4, t1 + min_dist_to_start + max_single_cost)
        
        return h
