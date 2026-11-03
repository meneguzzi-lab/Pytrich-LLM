# Robot Domain Heuristic
# Bottleneck: Single-carry constraint - robot holds one package at a time.
# Each misplaced package requires pickup + travel + putdown.
#
# Admissible lower bounds:
# T1: Sum of 2 per misplaced package (pickup + putdown) - admissible since each needs both.
# T2: BFS shortest path distances between rooms (precomputed ignoring closed doors for admissibility,
#     but using actual door states for tighter bounds). For each misplaced package, we need
#     robot travel to source + source to dest. We take the max single-package trip as one bound.
# T3: Task network structural lower bound - count of primitive-expanding abstract tasks remaining.

from collections import deque

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotAnthropicOpus46Heuristic11(Heuristic):
    """Domain-dependent heuristic for the Robot package delivery domain.
    
    Uses multi-component max of:
    - Misplaced package action cost (pickup+putdown per misplaced pkg)
    - BFS travel distance estimates
    - Task network lower bound
    """

    def __init__(self, name="robot_anthropicopus46_heuristic_11"):
        super().__init__(name)
        self.rloc_facts = {}       # room_name -> global_id
        self.in_facts = {}         # (pkg_name, room_name) -> global_id
        self.holding_facts = {}    # pkg_name -> global_id
        self.armempty_id = None
        self.goal_in_facts = {}    # (pkg_name, room_name) -> global_id
        self.closed_facts = {}     # door_name -> global_id
        self.door_facts = {}       # (room1, room2, door) -> global_id
        
        self.rooms = set()
        self.packages = set()
        self.doors = set()
        
        # Goal mapping: pkg -> goal_room
        self.pkg_goal_room = {}
        
        # Adjacency for BFS: room -> list of (neighbor_room, door_name)
        self.adjacency = {}
        
        # Precomputed all-pairs shortest path (ignoring door states - pure topology)
        self.topo_dist = {}  # (room1, room2) -> int
        
        # Fact ID lookups for fast bitwise checks
        self.rloc_id = {}      # room_name -> global_id
        self.in_id = {}        # (pkg, room) -> global_id  
        self.holding_id = {}   # pkg -> global_id
        self.closed_id = {}    # door -> global_id
        
        # For each package: source room from goal_in facts
        self.pkg_goal = {}     # pkg_name -> room_name
        
        # Operator and abstract task tracking
        self.pickup_op_ids = set()
        self.putdown_op_ids = set()
        self.move_op_ids = set()
        self.open_op_ids = set()

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

    def _bfs_all_pairs(self):
        """Compute all-pairs shortest path on room graph ignoring door states."""
        for src in self.rooms:
            dist = {src: 0}
            queue = deque([src])
            while queue:
                r = queue.popleft()
                for neighbor, door in self.adjacency.get(r, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[r] + 1
                        queue.append(neighbor)
            for dest in self.rooms:
                self.topo_dist[(src, dest)] = dist.get(dest, 999)

    def _bfs_with_closed_doors(self, state):
        """BFS from robot location considering closed doors need opening.
        Returns dict: room -> min actions to reach from current robot location.
        Each closed door costs 2 (open + move), each open door costs 1 (move).
        """
        # Find robot location
        robot_room = None
        for room, fid in self.rloc_id.items():
            if (state >> fid) & 1:
                robot_room = room
                break
        if robot_room is None:
            return {}, None
        
        dist = {robot_room: 0}
        queue = deque([robot_room])
        while queue:
            r = queue.popleft()
            for neighbor, door in self.adjacency.get(r, []):
                door_closed = False
                if door in self.closed_id:
                    door_closed = (state >> self.closed_id[door]) & 1
                cost = dist[r] + (2 if door_closed else 1)
                if neighbor not in dist or cost < dist[neighbor]:
                    dist[neighbor] = cost
                    queue.append(neighbor)
        return dist, robot_room

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc" and len(args) == 1:
                self.rloc_id[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "in" and len(args) == 2:
                self.in_id[(args[0], args[1])] = fact.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "holding" and len(args) == 1:
                self.holding_id[args[0]] = fact.global_id
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = fact.global_id
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_facts[(args[0], args[1])] = fact.global_id
                self.pkg_goal[args[0]] = args[1]
            elif pred == "closed" and len(args) == 1:
                self.closed_id[args[0]] = fact.global_id
                self.doors.add(args[0])
            elif pred == "door" and len(args) == 3:
                self.door_facts[(args[0], args[1], args[2])] = fact.global_id
                self.rooms.add(args[0])
                self.rooms.add(args[1])
                self.doors.add(args[2])
                if args[0] not in self.adjacency:
                    self.adjacency[args[0]] = []
                self.adjacency[args[0]].append((args[1], args[2]))
        
        # Parse operators for identification
        for op in model.operators:
            op_name = op.name.strip()
            if op_name.startswith("+") or op_name.startswith("-"):
                op_name = op_name[1:]
            bracket_idx = op_name.find("[")
            if bracket_idx > 0:
                op_pred = op_name[:bracket_idx]
                if op_pred == "pickup":
                    self.pickup_op_ids.add(op.global_id)
                elif op_pred == "putdown":
                    self.putdown_op_ids.add(op.global_id)
                elif op_pred == "move":
                    self.move_op_ids.add(op.global_id)
                elif op_pred == "open":
                    self.open_op_ids.add(op.global_id)
        
        # Compute all-pairs shortest path
        self._bfs_all_pairs()
        
        # Compute initial heuristic
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        """Compute heuristic value from state and task network."""
        # Term 1: Misplaced packages cost
        # Each misplaced package needs at minimum pickup(1) + putdown(1) = 2 actions
        # Plus travel from current location to source, source to dest
        misplaced_packages = []
        already_at_goal = 0
        holding_pkg = None
        
        # Check if robot is holding something
        for pkg, fid in self.holding_id.items():
            if (state >> fid) & 1:
                holding_pkg = pkg
                break
        
        for pkg, goal_room in self.pkg_goal.items():
            if pkg == holding_pkg:
                # Package is being held - needs putdown (and maybe travel)
                misplaced_packages.append((pkg, None, goal_room, True))
                continue
            
            # Check if package is already at goal
            key = (pkg, goal_room)
            if key in self.in_id:
                fid = self.in_id[key]
                if (state >> fid) & 1:
                    already_at_goal += 1
                    continue
            
            # Find current location of package
            pkg_room = None
            for room in self.rooms:
                k = (pkg, room)
                if k in self.in_id:
                    if (state >> self.in_id[k]) & 1:
                        pkg_room = room
                        break
            
            if pkg_room is not None:
                misplaced_packages.append((pkg, pkg_room, goal_room, False))
        
        num_misplaced = len(misplaced_packages)
        
        if num_misplaced == 0:
            # Check task network - if still has tasks, return small value
            if task_network:
                return len(task_network)
            return 0
        
        # Term 1: Base pickup+putdown cost
        # Each misplaced package needs at least 2 actions (pickup + putdown)
        # But held package only needs putdown (1) + possible travel
        t1_base = 0
        for pkg, src, dest, is_held in misplaced_packages:
            if is_held:
                t1_base += 1  # putdown
            else:
                t1_base += 2  # pickup + putdown
        
        # Term 2: Travel cost estimate using Dijkstra-like BFS with door states
        # Find robot location
        robot_room = None
        for room, fid in self.rloc_id.items():
            if (state >> fid) & 1:
                robot_room = room
                break
        
        if robot_room is None:
            # Fallback
            h = t1_base + len(task_network)
            return h
        
        # Compute distances from robot with door awareness
        dist_from_robot, _ = self._bfs_with_closed_doors(state)
        
        # For travel estimation: robot must visit each package source then its dest
        # Lower bound: sum of source-to-dest distances for all misplaced packages
        # Plus min distance from robot to nearest package source (greedy lower bound)
        t2_travel = 0
        min_robot_to_src = 999
        
        for pkg, src, dest, is_held in misplaced_packages:
            if is_held:
                # Need to travel to dest
                robot_to_dest = dist_from_robot.get(dest, self.topo_dist.get((robot_room, dest), 999))
                t2_travel += robot_to_dest
            else:
                # Need travel from source to dest
                src_to_dest = self.topo_dist.get((src, dest), 999)
                t2_travel += src_to_dest
                robot_to_src = dist_from_robot.get(src, self.topo_dist.get((robot_room, src), 999))
                if robot_to_src < min_robot_to_src:
                    min_robot_to_src = robot_to_src
        
        # Add minimum robot-to-first-package distance
        if min_robot_to_src < 999 and not (num_misplaced == 1 and misplaced_packages[0][3]):
            t2_travel += min_robot_to_src
        
        # Term 3: Serialization cost - robot carries one at a time
        # After delivering one package, robot must travel to next package source.
        # Minimum additional travel for serialization: (num_misplaced - 1) trips between
        # consecutive delivery destinations and next pickup sources.
        # Conservative lower bound: at least 0 additional (already covered above)
        # Better: if more than 1 non-held misplaced, at least (n-1) additional moves
        non_held_misplaced = sum(1 for _, _, _, is_held in misplaced_packages if not is_held)
        t3_serial = 0
        if non_held_misplaced > 1:
            # Each package after the first requires the robot to return from
            # the previous delivery destination to the next pickup source.
            # Minimum cost: at least 0 per trip (if dest == next src), but
            # on average nonzero. We use 0 to stay admissible.
            t3_serial = 0
        
        # Term 4: Task network lower bound
        # Count tasks that must expand to at least one action
        t4_tn = 0
        if task_network:
            for task_id in task_network:
                component = self.model.get_component(task_id)
                # Check if it's an operator (primitive)
                if hasattr(component, 'add_effects'):
                    t4_tn += 1
                elif hasattr(component, 'decompositions'):
                    # Abstract task - estimate minimum decomposition cost
                    task_name = component.name
                    if task_name.startswith("+") or task_name.startswith("-"):
                        task_name = task_name[1:]
                    bracket_idx = task_name.find("[")
                    if bracket_idx > 0:
                        task_pred = task_name[:bracket_idx]
                    else:
                        task_pred = task_name
                    
                    if task_pred == "achieve-goals":
                        # This task could be finished (0 cost) or needs more work
                        # It's 0 if all goals met, otherwise at least pickup+release chain
                        pass  # Don't add, covered by package analysis
                    elif task_pred == "release":
                        # Release involves putdown + achieve-goals or move + release
                        t4_tn += 1  # At least 1 action (putdown or move)
                    elif task_pred == "pickup_abstract":
                        t4_tn += 1  # pickup action
                    elif task_pred == "putdown_abstract":
                        t4_tn += 1  # putdown action
                    elif task_pred == "move_abstract":
                        t4_tn += 1  # move action
                    elif task_pred == "open_abstract":
                        t4_tn += 1  # open action
        
        # Combine: max of different estimates
        h_actions = t1_base + t2_travel  # Action cost + travel
        h_combined = max(h_actions, t4_tn)
        
        return h_combined

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
