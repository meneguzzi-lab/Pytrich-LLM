# Robot Domain Heuristic
# Bottleneck: Single-carry capacity — robot holds one package at a time.
# Each misplaced package requires pickup + travel + putdown = at least 2 + shortest_path actions.
#
# Admissible lower bounds:
# T1: Sum of (2 + shortest_path(src, dst)) for each misplaced package. Since packages are
#     carried one at a time, this sum is a lower bound (no shared work between packages).
# T2: Task network lower bound — count of primitive tasks + minimum decomposition cost
#     for abstract tasks remaining.
# T3: Max single-package displacement — the hardest single package to deliver.
#
# We return max(T1_adjusted, T2, T3) but T1 (the sum) typically dominates.
# T1 is admissible because each misplaced package independently requires those actions
# and they cannot overlap (single arm).

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic19(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_19"):
        super().__init__(name)
        self.in_fact_ids = {}      # (package, room) -> global_id
        self.goal_in_facts = {}    # (package, room) -> global_id
        self.rloc_fact_ids = {}    # room -> global_id
        self.holding_fact_ids = {} # package -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}  # door -> global_id
        self.door_connections = {} # (room1, room2) -> door
        self.room_adj = {}        # room -> set of (neighbor_room, door)
        self.all_rooms = set()
        self.all_packages = set()
        self.shortest_dist = {}    # (room1, room2) -> int (ignoring closed doors, counting open as 1 action)
        self.package_goals = {}    # package -> goal_room
        self.package_src_init = {} # package -> initial_room (will be dynamic via state)
        # For task network analysis
        self.operator_base_names = {}  # operator global_id -> base action name

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
        """BFS from start_room, treating all doors as passable (admissible: ignores closed doors
        but each move still costs 1 action minimum). For a tighter bound we could add +1 for
        closed doors, but we keep it simple and admissible."""
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

    def _bfs_distances_with_doors(self, start_room):
        """BFS from start_room, counting +1 for each closed door on the path.
        This gives a tighter admissible bound: move costs 1, open+move costs 2 if door is closed.
        But door state is static (doors only open, never close), so we precompute with initial closed set.
        Actually, doors change during search, so we use optimistic (all open) for precomputation."""
        # We'll use the simpler version: all moves cost 1 (admissible lower bound)
        return self._bfs_distances(start_room)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "goal_in":
                self.goal_in_facts[(args[0], args[1])] = f.global_id
                self.package_goals[args[0]] = args[1]
                self.all_packages.add(args[0])
            elif pred == "rloc":
                self.rloc_fact_ids[args[0]] = f.global_id
                self.all_rooms.add(args[0])
            elif pred == "holding":
                self.holding_fact_ids[args[0]] = f.global_id
            elif pred == "armempty":
                self.armempty_id = f.global_id
            elif pred == "closed":
                self.closed_fact_ids[args[0]] = f.global_id
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                self.all_rooms.add(r1)
                self.all_rooms.add(r2)
                if r1 not in self.room_adj:
                    self.room_adj[r1] = []
                self.room_adj[r1].append((r2, d))
                self.door_connections[(r1, r2)] = d

        # Parse operator names for task network analysis
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred:
                self.operator_base_names[op.global_id] = pred

        # Precompute all-pairs shortest distances using BFS
        all_rooms_list = list(self.all_rooms)
        for room in all_rooms_list:
            dists = self._bfs_distances(room)
            for room2, d in dists.items():
                self.shortest_dist[(room, room2)] = d

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_location(self, state):
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _compute(self, state, task_network):
        # Term 1: Sum of displacement costs for misplaced packages
        # Each misplaced package needs at least: pickup(1) + travel(src->dst) + putdown(1)
        # = 2 + shortest_dist(src, dst)
        # Since robot carries one at a time, these are additive (admissible).
        
        robot_loc = self._get_robot_location(state)
        
        misplaced_packages = []
        holding_pkg = None
        holding_pkg_goal = None
        
        # Check if robot is holding something
        for pkg, fid in self.holding_fact_ids.items():
            if (state >> fid) & 1:
                holding_pkg = pkg
                if pkg in self.package_goals:
                    holding_pkg_goal = self.package_goals[pkg]
                break
        
        total_displacement = 0
        max_single_cost = 0
        num_misplaced = 0
        
        for pkg, goal_room in self.package_goals.items():
            if pkg == holding_pkg:
                # Package is being held — need to travel to goal and putdown
                if robot_loc and goal_room in self.shortest_dist.get((robot_loc, goal_room), {goal_room: 0}):
                    pass
                dist_to_goal = self.shortest_dist.get((robot_loc, goal_room), 0) if robot_loc else 0
                cost = dist_to_goal + 1  # travel + putdown
                total_displacement += cost
                if cost > max_single_cost:
                    max_single_cost = cost
                num_misplaced += 1
                continue
            
            # Check if package is at goal
            key = (pkg, goal_room)
            if key in self.in_fact_ids:
                fid = self.in_fact_ids[key]
                if (state >> fid) & 1:
                    # Package is at goal — cost 0
                    continue
            
            # Package is misplaced — find its current location
            pkg_room = None
            for (p, r), fid in self.in_fact_ids.items():
                if p == pkg and (state >> fid) & 1:
                    pkg_room = r
                    break
            
            if pkg_room is None:
                # Package might be held by robot or lost — shouldn't happen if not holding_pkg
                continue
            
            # Cost: pickup(1) + travel(pkg_room -> goal_room) + putdown(1)
            dist = self.shortest_dist.get((pkg_room, goal_room), 0)
            cost = 2 + dist
            total_displacement += cost
            if cost > max_single_cost:
                max_single_cost = cost
            num_misplaced += 1
        
        # Term 2: Task network lower bound
        # Count minimum primitive actions needed from task network
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task — estimate minimum cost
                task_name = task.name if hasattr(task, 'name') else ''
                pred, args = self._parse_fact_name(task_name)
                if pred == 'achieve-goals':
                    # This will generate all the work — don't double count with T1
                    pass
                elif pred == 'release':
                    # At least a putdown (1 action)
                    tn_cost += 1
                elif pred == 'pickup_abstract':
                    tn_cost += 1
                elif pred == 'putdown_abstract':
                    tn_cost += 1
                elif pred == 'move_abstract':
                    tn_cost += 1
                elif pred == 'open_abstract':
                    tn_cost += 1
            else:
                # Primitive operator — costs 1
                tn_cost += 1
        
        # Term 3: Robot travel lower bound
        # The robot must reach each misplaced package's source room and each goal room.
        # A lower bound: if holding nothing, robot must travel to nearest misplaced package first.
        robot_travel = 0
        if robot_loc and num_misplaced > 0 and holding_pkg is None:
            # Must travel to at least one package source
            min_dist_to_any_pkg = float('inf')
            for pkg, goal_room in self.package_goals.items():
                key = (pkg, goal_room)
                if key in self.in_fact_ids:
                    fid = self.in_fact_ids[key]
                    if (state >> fid) & 1:
                        continue
                # Find package location
                for (p, r), fid in self.in_fact_ids.items():
                    if p == pkg and (state >> fid) & 1:
                        d = self.shortest_dist.get((robot_loc, r), float('inf'))
                        if d < min_dist_to_any_pkg:
                            min_dist_to_any_pkg = d
                        break
            if min_dist_to_any_pkg < float('inf'):
                robot_travel = min_dist_to_any_pkg
        
        # Combine: total_displacement already includes per-package travel
        # Add robot_travel as the initial travel to reach first package
        h1 = total_displacement + robot_travel
        h2 = tn_cost
        h3 = max_single_cost
        
        return max(h1, h2, h3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
