# Robot Domain Heuristic
# Bottleneck: Single-carry constraint — robot holds at most one package at a time,
# serializing all deliveries. Each misplaced package needs pickup + travel + putdown.
#
# Admissible lower bounds:
# T1: Each misplaced package needs >= 2 actions (pickup + putdown). Sum of misplaced * 2 is NOT
#     admissible as a sum (robot can interleave), but max over individual package costs IS.
#     We use: number_of_misplaced_packages * 2 as a lower bound since each package needs
#     at least pickup+putdown and these are serialized (only one held at a time).
# T2: Minimum total navigation cost: robot must visit source and dest of each misplaced package.
#     We use a TSP-style lower bound via minimum spanning considerations.
# T3: Task network depth estimate: minimum decomposition steps remaining.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque


class RobotAnthropicOpus46Heuristic1(Heuristic):
    def __init__(self, name="robot_anthropicopus46_heuristic_1"):
        super().__init__(name)
        self.model = None
        # Fact ID maps
        self.in_fact_ids = {}       # (pkg, room) -> global_id
        self.goal_in_fact_ids = {}  # (pkg, room) -> global_id
        self.rloc_fact_ids = {}     # room -> global_id
        self.holding_fact_ids = {}  # pkg -> global_id
        self.armempty_id = None
        self.closed_fact_ids = {}   # door -> global_id
        self.door_facts = {}        # (room1, room2) -> door
        
        # Goal info
        self.goals_by_pkg = {}      # pkg -> dest_room
        self.pkg_initial_room = {}  # pkg -> initial source room (from goal_in matching)
        
        # Room graph and shortest paths
        self.room_names = set()
        self.adj = {}               # room -> set of (neighbor_room, door)
        self.dist = {}              # (room1, room2) -> shortest distance (number of moves)
        
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
        """Compute all-pairs shortest path distances between rooms using BFS on the room graph.
        Distances are in terms of number of move actions (ignoring door open costs)."""
        rooms = list(self.room_names)
        for src in rooms:
            if src not in self.adj:
                continue
            # BFS from src
            dist_from_src = {src: 0}
            queue = deque([src])
            while queue:
                curr = queue.popleft()
                if curr not in self.adj:
                    continue
                for neighbor, door in self.adj[curr]:
                    if neighbor not in dist_from_src:
                        dist_from_src[neighbor] = dist_from_src[curr] + 1
                        queue.append(neighbor)
            for dest, d in dist_from_src.items():
                self.dist[(src, dest)] = d

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Parse all facts and build lookup maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "goal_in" and len(args) == 2:
                self.goal_in_fact_ids[(args[0], args[1])] = f.global_id
                self.goals_by_pkg[args[0]] = args[1]
            elif pred == "rloc" and len(args) == 1:
                self.rloc_fact_ids[args[0]] = f.global_id
                self.room_names.add(args[0])
            elif pred == "holding" and len(args) == 1:
                self.holding_fact_ids[args[0]] = f.global_id
            elif pred == "armempty" and len(args) == 0:
                self.armempty_id = f.global_id
            elif pred == "closed" and len(args) == 1:
                self.closed_fact_ids[args[0]] = f.global_id
            elif pred == "door" and len(args) == 3:
                room1, room2, door = args[0], args[1], args[2]
                self.room_names.add(room1)
                self.room_names.add(room2)
                if room1 not in self.adj:
                    self.adj[room1] = set()
                self.adj[room1].add((room2, door))
                self.door_facts[(room1, room2)] = door
        
        # Compute all-pairs shortest paths
        self._bfs_all_pairs()
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_robot_room(self, state):
        """Get current robot room from state using bitwise checks."""
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_holding_pkg(self, state):
        """Get the package the robot is currently holding, if any."""
        for pkg, fid in self.holding_fact_ids.items():
            if (state >> fid) & 1:
                return pkg
        return None

    def _compute(self, state, task_network):
        """Compute heuristic value."""
        robot_room = self._get_robot_room(state)
        holding_pkg = self._get_holding_pkg(state)
        
        # Find all misplaced packages
        misplaced = []  # list of (pkg, current_room_or_None, dest_room)
        for pkg, dest_room in self.goals_by_pkg.items():
            # Check if package is already at its goal
            in_key = (pkg, dest_room)
            if in_key in self.in_fact_ids:
                fid = self.in_fact_ids[in_key]
                if (state >> fid) & 1:
                    continue  # Already at goal
            
            # Package is misplaced - find its current location
            if pkg == holding_pkg:
                # Robot is holding this package
                misplaced.append((pkg, None, dest_room))  # None means robot has it
            else:
                # Find where the package currently is
                current_room = None
                for (p, r), fid in self.in_fact_ids.items():
                    if p == pkg and (state >> fid) & 1:
                        current_room = r
                        break
                if current_room is not None:
                    misplaced.append((pkg, current_room, dest_room))
        
        num_misplaced = len(misplaced)
        
        if num_misplaced == 0:
            # Check if task network is empty
            if len(task_network) == 0:
                return 0
            else:
                # Tasks remain but no misplaced packages - small cost for remaining tasks
                return len(task_network)
        
        # Term 1: Serialized delivery cost
        # Each misplaced package needs at least pickup(1) + putdown(1) = 2 actions
        # Since robot can carry only one at a time, these are serialized
        # But if robot is already holding a package, that one only needs putdown
        t1_serial = 0
        for pkg, src_room, dest_room in misplaced:
            if src_room is None:
                # Robot is holding this package, only need putdown
                t1_serial += 1
            else:
                # Need pickup + putdown
                t1_serial += 2
        
        # Term 2: Navigation cost lower bound
        # We compute a lower bound on total navigation needed.
        # For the package currently held (if any), robot must go to dest.
        # For each other package, robot must go to src, then to dest.
        # This is like a TSP but we use a greedy lower bound.
        
        # Simple but effective: sum of min travel for each package
        # divided by some factor to stay admissible, since robot trips overlap.
        # Actually, for a true lower bound, we can compute:
        # - If holding a package: distance to its destination
        # - Then for remaining packages: minimum of (travel to src + src to dest)
        # But these overlap. A simple admissible bound:
        # max over all packages of (distance_to_reach_src + src_to_dest)
        
        t2_nav = 0
        
        if robot_room is not None:
            # Compute individual package delivery costs
            pkg_costs = []
            for pkg, src_room, dest_room in misplaced:
                if src_room is None:
                    # Holding this package, just need to go to dest
                    d = self.dist.get((robot_room, dest_room), 0)
                    pkg_costs.append(d)
                else:
                    # Need to go to src, then to dest
                    d_to_src = self.dist.get((robot_room, src_room), 0)
                    d_src_to_dest = self.dist.get((src_room, dest_room), 0)
                    pkg_costs.append(d_to_src + d_src_to_dest)
            
            if pkg_costs:
                # The max single-package cost is a lower bound because the robot
                # must at least deliver that one package
                t2_nav_max = max(pkg_costs) if pkg_costs else 0
                
                # Also: total minimum travel. Since deliveries are serialized,
                # a better lower bound considers that after delivering one package,
                # the robot is at the destination and must travel to the next source.
                # We use a greedy nearest-neighbor TSP lower bound:
                # Sum of all src_to_dest distances is a lower bound on just the
                # loaded-travel portion (each package must travel src->dest while held)
                loaded_travel = 0
                for pkg, src_room, dest_room in misplaced:
                    if src_room is None:
                        loaded_travel += self.dist.get((robot_room, dest_room), 0)
                    else:
                        loaded_travel += self.dist.get((src_room, dest_room), 0)
                
                t2_nav = max(t2_nav_max, loaded_travel)
        
        # Term 3: Closed doors that need opening
        # Count closed doors on shortest paths for misplaced packages.
        # Each closed door needs at least 1 open action. But this is hard to compute exactly.
        # Simple version: just count currently closed doors (not all need opening though).
        # Better: for each misplaced package, check if any door on shortest path is closed.
        # For admissibility, we skip this complex computation and use a simpler bound.
        
        # Term 4: Task network based estimate
        # Count primitive tasks and abstract tasks in the network
        t4_tn = 0
        iop_init = self.model.iop_init
        iop_end = self.model.iop_end
        iabt_init = self.model.iabt_init
        iabt_end = self.model.iabt_end
        
        for task in task_network:
            gid = task.global_id
            if gid >= iop_init and gid <= iop_end:
                # Primitive task - costs 1
                t4_tn += 1
            elif gid >= iabt_init and gid <= iabt_end:
                # Abstract task - estimate minimum decomposition cost
                task_name = task.name
                if 'achieve-goals' in task_name:
                    # This could decompose to nothing (finished method) or to more work
                    # Don't count it since misplaced packages already counted
                    pass
                elif 'release' in task_name:
                    # Release involves putdown + achieve-goals, min cost = 1 (putdown)
                    pass
                elif 'pickup_abstract' in task_name:
                    t4_tn += 1  # At least 1 pickup action
                elif 'putdown_abstract' in task_name:
                    t4_tn += 1  # At least 1 putdown action
                elif 'move_abstract' in task_name:
                    t4_tn += 1  # At least 1 move action
                elif 'open_abstract' in task_name:
                    t4_tn += 1  # At least 1 open action
                else:
                    t4_tn += 1  # Generic abstract task
        
        # Combine: total cost >= serial_delivery_cost
        # total cost >= navigation_cost + serial_delivery_cost  
        # The navigation moves are ADDITIONAL to pickup/putdown actions
        
        # t1_serial counts pickups and putdowns
        # t2_nav counts move actions needed
        # These are disjoint action types, so their sum is admissible!
        combined_additive = t1_serial + t2_nav
        
        # Also use task network estimate
        result = max(combined_additive, t4_tn)
        
        return result

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
