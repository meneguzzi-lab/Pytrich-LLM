"""Domain-dependent heuristic for the Robot domain (HDDL).

Bottleneck Analysis:
  The robot can carry only ONE package at a time (single-carry constraint).
  Every misplaced package requires at least: move to source, pickup, move to dest, putdown = 4 actions minimum.
  Packages already at their goal location cost 0. The robot must visit every source and destination room.

Admissible Lower Bounds:
  T1 - Misplaced package count * 2: Each misplaced package needs at least pickup + putdown = 2 primitive actions.
       This is admissible because no single action can resolve more than one package.
  T2 - Sum of BFS distances: For each misplaced package, the minimum number of move actions from its
       current room to its goal room (ignoring doors being closed). This is a relaxed lower bound since
       we ignore the constraint that the robot must physically travel to pick up each package.
       We take max(sum_of_min_distances, 0) as a lower bound on total moves needed.
  T3 - If holding a package, we need at least 1 action (putdown) plus the distance to the goal room.
       Combined with the number of remaining misplaced packages.

The heuristic returns max(T1, T2_adjusted, T3) for strong guidance.

Preprocessing (initialize):
  - Parse all facts to build lookup maps for predicates: in, goal_in, rloc, holding, door, closed, armempty.
  - Build room adjacency graph and compute all-pairs shortest path (BFS) distances.
  - Index goal_in facts for quick lookup.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic3(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_3"):
        super().__init__(name)
        # Precomputed maps
        self.in_facts = {}        # (obj, room) -> global_id
        self.goal_in_facts = {}   # (obj, room) -> global_id
        self.rloc_facts = {}      # room -> global_id
        self.holding_facts = {}   # obj -> global_id
        self.closed_facts = {}    # door -> global_id
        self.door_facts = {}      # (room1, room2, door) -> global_id
        self.armempty_id = None
        
        # Goal info
        self.goal_destinations = {}  # obj -> goal_room
        self.goal_in_ids = {}        # obj -> global_id of goal_in fact
        
        # Room graph and distances
        self.rooms = set()
        self.objects = set()
        self.adjacency = {}       # room -> set of (neighbor_room, door)
        self.dist = {}            # (room1, room2) -> shortest distance (ignoring closed doors)
        
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
        
        # Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in":
                self.in_facts[(args[0], args[1])] = f.global_id
            elif pred == "goal_in":
                self.goal_in_facts[(args[0], args[1])] = f.global_id
                self.goal_destinations[args[0]] = args[1]
                self.goal_in_ids[args[0]] = f.global_id
                self.objects.add(args[0])
            elif pred == "rloc":
                self.rloc_facts[args[0]] = f.global_id
                self.rooms.add(args[0])
            elif pred == "holding":
                self.holding_facts[args[0]] = f.global_id
            elif pred == "closed":
                self.closed_facts[args[0]] = f.global_id
            elif pred == "door":
                self.door_facts[(args[0], args[1], args[2])] = f.global_id
                self.rooms.add(args[0])
                self.rooms.add(args[1])
            elif pred == "armempty":
                self.armempty_id = f.global_id
        
        # Build adjacency graph (all doors, ignoring closed status for distance computation)
        for room in self.rooms:
            self.adjacency[room] = set()
        
        for (r1, r2, d), fid in self.door_facts.items():
            self.adjacency[r1].add(r2)
        
        # Compute all-pairs shortest path using BFS from each room
        self._compute_all_pairs_distances()
        
        # Build reverse lookup: for each object, find which in_facts exist
        # (obj, room) pairs for quick state checking
        self.obj_room_pairs = {}  # obj -> list of rooms where in[obj,room] fact exists
        for (obj, room), fid in self.in_facts.items():
            if obj not in self.obj_room_pairs:
                self.obj_room_pairs[obj] = []
            self.obj_room_pairs[obj].append((room, fid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)
    
    def _compute_all_pairs_distances(self):
        """BFS from each room to compute shortest distances."""
        room_list = list(self.rooms)
        for src in room_list:
            # BFS from src
            dist_from_src = {src: 0}
            queue = deque([src])
            while queue:
                curr = queue.popleft()
                for neighbor in self.adjacency.get(curr, set()):
                    if neighbor not in dist_from_src:
                        dist_from_src[neighbor] = dist_from_src[curr] + 1
                        queue.append(neighbor)
            for dst, d in dist_from_src.items():
                self.dist[(src, dst)] = d
    
    def _get_robot_location(self, state):
        """Get current robot location from state."""
        for room, fid in self.rloc_facts.items():
            if (state >> fid) & 1:
                return room
        return None
    
    def _get_held_object(self, state):
        """Get currently held object, if any."""
        for obj, fid in self.holding_facts.items():
            if (state >> fid) & 1:
                return obj
        return None
    
    def _get_object_location(self, state, obj):
        """Get the room where obj currently is (via in predicate)."""
        if obj in self.obj_room_pairs:
            for room, fid in self.obj_room_pairs[obj]:
                if (state >> fid) & 1:
                    return room
        return None
    
    def _compute(self, state, task_network):
        """Compute the heuristic value."""
        if not task_network:
            return 0
        
        robot_loc = self._get_robot_location(state)
        held_obj = self._get_held_object(state)
        
        # Find all misplaced packages
        misplaced = []  # (obj, current_room, goal_room)
        for obj in self.goal_destinations:
            goal_room = self.goal_destinations[obj]
            
            # Check if obj is being held
            if held_obj == obj:
                # It's held, not yet at goal
                misplaced.append((obj, None, goal_room))  # None means held
                continue
            
            # Check if obj is at its goal location
            in_goal_id = self.in_facts.get((obj, goal_room))
            if in_goal_id is not None and (state >> in_goal_id) & 1:
                # Already at goal
                continue
            
            # Find current location
            curr_room = self._get_object_location(state, obj)
            if curr_room is not None:
                misplaced.append((obj, curr_room, goal_room))
        
        if not misplaced:
            # All packages at goal, but might still have tasks
            # Return small value based on remaining task network size
            return len(task_network)
        
        num_misplaced = len(misplaced)
        
        # T1: Each misplaced package needs at least pickup + putdown = 2 actions
        # If currently holding one, it just needs putdown (1 action) + the rest need 2 each
        if held_obj is not None and held_obj in self.goal_destinations:
            # Check if held obj is misplaced
            held_is_misplaced = any(obj == held_obj for obj, _, _ in misplaced)
            if held_is_misplaced:
                t1 = 1 + (num_misplaced - 1) * 2
            else:
                t1 = num_misplaced * 2
        else:
            t1 = num_misplaced * 2
        
        # T2: Sum of minimum distances for each misplaced package
        # For each package: distance(current_loc -> goal_loc) as move actions
        # Plus we need robot travel between packages (TSP-like lower bound)
        total_pkg_dist = 0
        min_dist_to_any_source = float('inf')
        
        for obj, curr_room, goal_room in misplaced:
            if curr_room is None:
                # Held by robot - distance from robot to goal
                if robot_loc and (robot_loc, goal_room) in self.dist:
                    d = self.dist[(robot_loc, goal_room)]
                else:
                    d = 1  # At least putdown
                total_pkg_dist += d
            else:
                # Distance from package source to package destination
                d = self.dist.get((curr_room, goal_room), 0)
                total_pkg_dist += d
                
                # Distance from robot to this package's source
                if robot_loc:
                    rd = self.dist.get((robot_loc, curr_room), float('inf'))
                    min_dist_to_any_source = min(min_dist_to_any_source, rd)
        
        # T2: travel lower bound
        # At minimum, robot must travel to nearest misplaced package source + sum of per-package distances
        # But since robot carries one at a time, we need at least total_pkg_dist moves
        # plus travel between packages (at least to the nearest one from current position)
        if min_dist_to_any_source == float('inf'):
            min_dist_to_any_source = 0
        
        t2 = total_pkg_dist + min_dist_to_any_source
        
        # T3: Holding-aware estimate
        # If holding something, we need to deliver it first
        t3 = 0
        if held_obj is not None and held_obj in self.goal_destinations:
            goal_room = self.goal_destinations[held_obj]
            if robot_loc:
                deliver_dist = self.dist.get((robot_loc, goal_room), 0)
                t3 = deliver_dist + 1  # moves to goal + putdown
                # Plus remaining misplaced (excluding held)
                remaining = num_misplaced - 1 if any(o == held_obj for o, _, _ in misplaced) else num_misplaced
                t3 += remaining * 2  # Each remaining needs at least pickup + putdown
        
        # T4: Task network lower bound - count minimum primitive actions needed
        # Count abstract tasks that must expand to at least 1 action
        t4 = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task - at minimum needs 1 primitive action
                t4 += 1
            else:
                # Primitive task (operator) - costs exactly 1
                t4 += 1
        
        # Take minimum of t4 and other estimates to keep it reasonable as a separate bound
        # Actually t4 is also a valid lower bound on its own
        
        return max(t1, t2, t3, t4)
    
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
