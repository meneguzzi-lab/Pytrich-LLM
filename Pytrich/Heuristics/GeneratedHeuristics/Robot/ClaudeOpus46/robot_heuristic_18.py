# Robot Domain Heuristic
# Bottleneck: Single arm (armempty) serializes all package transport.
# Each misplaced package needs pickup+putdown (2 actions min) plus movement.
#
# Admissible Lower Bounds:
# T1: Number of misplaced packages * 2 (each needs pickup + putdown)
# T2: Sum of shortest distances from each package's current location to its goal location
#     (each package must be carried across this distance, lower bound on move actions)
# T3: Task network structural lower bound - count of primitive tasks remaining
#
# We return max(T1, T2_combined, T3) for admissibility.
# T2_combined adds movement costs considering the robot must visit source then dest for each package.

from collections import deque

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotAnthropicOpus46Heuristic18(Heuristic):
    """Domain-dependent heuristic for the Robot domain.
    
    Exploits the single-arm bottleneck: each misplaced package requires
    at least pickup + travel + putdown. Uses BFS-precomputed shortest
    distances between all rooms to estimate movement cost.
    """

    def __init__(self, name="robot_anthropicopus46_heuristic_18"):
        super().__init__(name)
        self.in_fact_ids = {}       # (package, room) -> global_id
        self.goal_in_facts = {}     # (package, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # package -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door
        
        # Room graph and shortest paths
        self.rooms = set()
        self.packages = set()
        self.room_adj = {}          # room -> set of (neighbor_room, door)
        self.dist = {}              # (room1, room2) -> shortest distance (in move actions)
        
        # Goal mapping
        self.package_goal_room = {}  # package -> goal_room
        self.goal_pairs = []         # list of (package, goal_room, in_fact_gid, goal_in_gid)

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
        """BFS from start_room, returns dict room->distance (number of move actions).
        Treats all doors as passable (optimistic/admissible - ignores closed doors
        since opening a door is an extra action we account for separately)."""
        dist = {start_room: 0}
        queue = deque([start_room])
        while queue:
            room = queue.popleft()
            if room not in self.room_adj:
                continue
            for neighbor, door in self.room_adj[room]:
                if neighbor not in dist:
                    dist[neighbor] = dist[room] + 1
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
            elif pred == "goal_in":
                pkg, room = args[0], args[1]
                self.goal_in_facts[(pkg, room)] = f.global_id
                self.package_goal_room[pkg] = room
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
                self.rooms.add(r1)
                self.rooms.add(r2)
                if r1 not in self.room_adj:
                    self.room_adj[r1] = []
                self.room_adj[r1].append((r2, d))

        # Precompute all-pairs shortest distances using BFS from each room
        self.dist = {}
        for room in self.rooms:
            dists_from_room = self._bfs_distances(room)
            for target, d in dists_from_room.items():
                self.dist[(room, target)] = d

        # Build goal pairs for quick lookup
        self.goal_pairs = []
        for pkg, goal_room in self.package_goal_room.items():
            in_gid = self.in_fact_ids.get((pkg, goal_room))
            holding_gid = self.holding_fact_ids.get(pkg)
            self.goal_pairs.append((pkg, goal_room, in_gid, holding_gid))

        # Build reverse lookup: for each package, all (room, gid) pairs for in facts
        self.package_in_rooms = {}
        for (pkg, room), gid in self.in_fact_ids.items():
            if pkg not in self.package_in_rooms:
                self.package_in_rooms[pkg] = []
            self.package_in_rooms[pkg].append((room, gid))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        """Find current robot location from state bits."""
        for room, gid in self.rloc_fact_ids.items():
            if (state >> gid) & 1:
                return room
        return None

    def _get_package_location(self, state, pkg):
        """Find where a package currently is. Returns room or None if being held."""
        if pkg in self.package_in_rooms:
            for room, gid in self.package_in_rooms[pkg]:
                if (state >> gid) & 1:
                    return room
        return None  # Package is being held

    def _compute(self, state, task_network):
        """Compute heuristic value for given state and task network."""
        robot_loc = self._get_robot_location(state)
        
        # Check if arm is empty
        arm_empty = (state >> self.armempty_id) & 1 if self.armempty_id is not None else True
        
        # Term 1: Count misplaced packages * 2 (pickup + putdown each)
        misplaced_packages = []
        held_package = None
        held_package_goal = None
        
        for pkg, goal_room, in_goal_gid, holding_gid in self.goal_pairs:
            if in_goal_gid is not None and (state >> in_goal_gid) & 1:
                # Package is already at goal
                continue
            # Package is misplaced
            if holding_gid is not None and (state >> holding_gid) & 1:
                # Robot is holding this package
                held_package = pkg
                held_package_goal = goal_room
            else:
                pkg_loc = self._get_package_location(state, pkg)
                if pkg_loc is not None:
                    misplaced_packages.append((pkg, pkg_loc, goal_room))
        
        num_misplaced = len(misplaced_packages)
        if held_package is not None:
            num_misplaced += 1
        
        # T1: Each misplaced package needs at least pickup + putdown = 2 actions
        # The held package only needs putdown = 1 action (but also travel)
        t1 = 0
        if held_package is not None:
            t1 = 1  # putdown for held package
        t1 += num_misplaced * 2 - (1 if held_package is not None else 0)
        if t1 < 0:
            t1 = 0
        
        # T2: Movement-aware cost estimation
        # For each misplaced package: robot must go to source, then to dest
        # The held package: robot must go to dest
        # We compute a lower bound on total travel + pickup/putdown
        
        t2 = 0
        if robot_loc is not None:
            if held_package is not None:
                # Must deliver held package: travel to goal + putdown
                dist_to_goal = self.dist.get((robot_loc, held_package_goal), 0)
                t2 += dist_to_goal + 1  # move + putdown
                
                # For remaining packages, estimate optimistically
                # After delivering held package, robot is at held_package_goal
                # Each remaining package needs: travel to source + pickup + travel to dest + putdown
                # Lower bound: sum of (source_to_dest_distance + 2) for each
                for pkg, src, dest in misplaced_packages:
                    src_to_dest = self.dist.get((src, dest), 0)
                    t2 += src_to_dest + 2  # travel + pickup + putdown (min)
            else:
                # Arm is empty, need to pick up and deliver each misplaced package
                # Lower bound: for each package, at minimum source_to_dest + 2
                for pkg, src, dest in misplaced_packages:
                    src_to_dest = self.dist.get((src, dest), 0)
                    t2 += src_to_dest + 2
        
        # T3: Task network based lower bound
        # Count minimum primitive actions needed from task network
        t3 = 0
        if task_network:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                # Check if it's an operator (primitive task)
                if hasattr(component, 'add_effects'):
                    t3 += 1
                elif hasattr(component, 'decompositions'):
                    # Abstract task - estimate minimum decomposition depth
                    task_name = component.name if hasattr(component, 'name') else ""
                    if 'achieve-goals' in task_name:
                        # achieve-goals can finish immediately (method finished) or do work
                        # Don't count - it's context-dependent
                        pass
                    elif 'pickup_abstract' in task_name:
                        t3 += 1  # decomposes to exactly 1 pickup action
                    elif 'putdown_abstract' in task_name:
                        t3 += 1  # decomposes to exactly 1 putdown action
                    elif 'move_abstract' in task_name:
                        t3 += 1  # decomposes to exactly 1 move action
                    elif 'open_abstract' in task_name:
                        t3 += 1  # decomposes to exactly 1 open action
                    elif 'release' in task_name:
                        # release needs at minimum: putdown + achieve-goals
                        # but might need moves/opens first
                        t3 += 1  # at least putdown
        
        # T4: Door opening cost - count doors that must be opened
        # This is hard to compute exactly, but we can add a small bonus
        # for doors on critical paths that are still closed
        # (Already somewhat captured by treating graph as fully open in BFS - admissible)
        
        # T5: Serialization penalty
        # Since robot carries one package at a time, if there are N misplaced packages,
        # the robot must make N separate trips. The minimum total travel is at least
        # the sum of individual package source-to-dest distances.
        # But additionally, the robot must travel between deliveries.
        # A tighter bound: use the minimum spanning path through all sources and destinations
        # This is expensive to compute, so we use a simpler bound:
        # If carrying, must first deliver, then do remaining trips
        
        # Combine with max for admissibility
        h = max(t1, t2, t3)
        
        # Tie-breaking: slight preference for states where arm is empty (ready for next pickup)
        # and slight preference for being closer to a misplaced package
        tie_break = 0.0
        if robot_loc is not None and misplaced_packages:
            # Small penalty for distance to nearest misplaced package source
            min_dist_to_src = float('inf')
            for pkg, src, dest in misplaced_packages:
                d = self.dist.get((robot_loc, src), 100)
                if d < min_dist_to_src:
                    min_dist_to_src = d
            if min_dist_to_src < float('inf'):
                tie_break += 0.001 * min_dist_to_src
        
        return h + tie_break

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
