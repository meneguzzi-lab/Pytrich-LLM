"""
Domain-dependent heuristic for the Robot planning domain.

Bottleneck Analysis:
The primary bottleneck is the robot's single arm and the serial nature of transporting packages. 
The robot can carry only one package at a time, meaning every misplaced package requires a 
distinct sequence of pickup, movement, and putdown actions. Additionally, navigation is 
constrained by doors that may need to be opened.

Admissible Lower Bounds:
1. Delivery Actions: Each package not at its goal room requires at least 2 actions (pickup + putdown).
   If the robot is currently holding the package, it only needs 1 action (putdown). 
   This is a strict lower bound on action count.
2. Movement Actions: Since the robot moves packages one by one, the total number of 'move' actions 
   must be at least the sum of distances from each package's current location to its target room.
   Additionally, the robot must move to the first package if it is not already holding one.
3. Opening Actions: Every closed door that lies on a required path between a package and its goal 
   must be opened at least once.

Complexity: O(N_packages) in the search loop after O(N_rooms^2) preprocessing.
"""

import time
from collections import deque, defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic3(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_3=None, name="RobotGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.package_goals = {}  # package -> goal_room
        self.packages = []
        self.rooms = []
        self.doors = []

        # Fact Maps
        self.rloc_fids = {}        # room -> fid
        self.in_fids = {}          # (package, room) -> fid
        self.holding_fids = {}     # package -> fid
        self.closed_fids = {}      # door -> fid
        self.armempty_fid = None

        # Static connectivity
        self.adj = defaultdict(list)
        self.room_dist = {}        # (r1, r2) -> distance
        self.room_path_doors = {}  # (r1, r2) -> set of doors

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse and classify facts
        pkgs = set()
        rms = set()
        drs = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "rloc":
                rms.add(args[0])
                self.rloc_fids[args[0]] = fact.global_id
            elif pred == "in":
                pkgs.add(args[0])
                rms.add(args[1])
                if args[0] not in self.in_fids: self.in_fids[args[0]] = {}
                self.in_fids[args[0]][args[1]] = fact.global_id
            elif pred == "holding":
                pkgs.add(args[0])
                self.holding_fids[args[0]] = fact.global_id
            elif pred == "closed":
                drs.add(args[0])
                self.closed_fids[args[0]] = fact.global_id
            elif pred == "armempty":
                self.armempty_fid = fact.global_id
            elif pred == "door":
                # door(r1, r2, d)
                r1, r2, d = args[0], args[1], args[2]
                self.adj[r1].append((r2, d))
                self.adj[r2].append((r1, d))
            elif pred == "goal_in":
                # goal_in(p, r) is used to mark destination in robot domain methods/actions
                # It appears in initial state
                if (initial_node.state >> fact.global_id) & 1:
                    self.package_goals[args[0]] = args[1]

        self.packages = list(pkgs)
        self.rooms = list(rms)
        self.doors = list(drs)

        # 2. Precompute room-to-room shortest paths (BFS)
        for start_room in self.rooms:
            self._compute_distances_from(start_room)

        h_initial = self._compute_h(initial_node.state)
        return super().initialize(model, h_initial)

    def _compute_distances_from(self, start):
        q = deque([start])
        visited = {start}
        self.room_dist[(start, start)] = 0
        self.room_path_doors[(start, start)] = set()

        while q:
            curr = q.popleft()
            curr_dist = self.room_dist[(start, curr)]
            curr_path = self.room_path_doors[(start, curr)]

            for neighbor, door_name in self.adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    self.room_dist[(start, neighbor)] = curr_dist + 1
                    self.room_path_doors[(start, neighbor)] = curr_path.union({door_name})
                    q.append(neighbor)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def _compute_h(self, state):
        # 1. Get current robot state
        robot_loc = None
        for r in self.rooms:
            if (state >> self.rloc_fids[r]) & 1:
                robot_loc = r
                break
        if not robot_loc: return 0

        holding_p = None
        for p in self.packages:
            if (state >> self.holding_fids[p]) & 1:
                holding_p = p
                break

        # 2. Identify misplaced packages and their current positions
        misplaced = []
        p_locs = {}
        for p in self.packages:
            # A package is misplaced if it's not at its goal_in location
            goal_r = self.package_goals.get(p)
            if not goal_r: continue

            curr_loc = None
            if p == holding_p:
                curr_loc = robot_loc
            else:
                for r, fid in self.in_fids.get(p, {}).items():
                    if (state >> fid) & 1:
                        curr_loc = r
                        break
            
            if curr_loc and (curr_loc != goal_r or p == holding_p):
                misplaced.append(p)
                p_locs[p] = curr_loc

        if not misplaced: return 0

        # 3. Calculate heuristic terms
        h_delivery = 0
        h_moves = 0
        required_doors = set()

        for p in misplaced:
            goal_r = self.package_goals[p]
            curr_r = p_locs[p]

            # Action costs: pickup + putdown
            h_delivery += (1 if p == holding_p else 2)

            # Movement costs: current location to goal room
            dist = self.room_dist.get((curr_r, goal_r), 0)
            h_moves += dist
            
            # Door costs
            required_doors.update(self.room_path_doors.get((curr_r, goal_r), set()))

        # If not holding, must move to the closest misplaced package
        if not holding_p:
            min_move_to_p = min(self.room_dist.get((robot_loc, p_locs[p]), 999) for p in misplaced)
            h_moves += min_move_to_p
            # Note: Doors for this initial segment are omitted for strict admissibility in edge cases

        # 4. Count unique closed doors on path
        h_open = 0
        for d in required_doors:
            fid = self.closed_fids.get(d)
            if fid is not None and (state >> fid) & 1:
                h_open += 1

        # Tie-breaker: prefer states where robot is closer to targets
        tie_breaker = 0.001 * sum(self.room_dist.get((robot_loc, p_locs[p]), 0) for p in misplaced)

        return int(h_delivery + h_moves + h_open) + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute_h(node.state)
        super().update_info(h_val)
        return h_val
