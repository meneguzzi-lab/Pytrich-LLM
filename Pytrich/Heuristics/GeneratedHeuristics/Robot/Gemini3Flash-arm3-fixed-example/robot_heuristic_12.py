"""
Domain-dependent heuristic for the Robot HDDL domain.

Bottleneck Analysis:
The primary bottleneck is the single-arm constraint: the robot can carry only one package at a time.
Every unsettled package requires a sequence of: move-to-package, pickup, move-to-goal, putdown.
Since these sequences must be performed mostly serially (one package at a time), the total cost 
is dominated by the sum of manipulation actions (pickup/putdown) and the necessary travel distances.

Admissibility Sketches:
1. Manipulation Cost: Each package not at its goal room must at least be picked up and put down. 
   If the package is not currently held, this requires 2 actions. If it is already held, it requires 1 (putdown).
   Cost = 2 * count(unsettled_not_held) + 1 * count(held).
2. Travel Cost: The robot must travel the shortest path distance (number of move actions) between 
   locations. For any package p, the distance from its current location to its goal location is 
   an unavoidable number of 'move' actions. Adding the distance to the first package to be handled 
   maintains admissibility because those moves must happen before any pickup.
"""

import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic12(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_12=None, name="RobotGemini3FlashHeuristic12"):
        super().__init__(name=name)
        self.package_goal = {}  # pkg -> goal_room
        self.pkg_in_fact_ids = {} # pkg -> {room: global_id}
        self.holding_fact_ids = {} # pkg -> global_id
        self.rloc_fact_ids = {} # room -> global_id
        self.room_dist = {} # room1 -> {room2: dist}
        self.rooms = []
        self.packages = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse all facts to identify static structure and map predicates
        adj = {}
        all_rooms = set()
        all_pkgs = set()

        # Extract goal_in (static target locations)
        # Note: goal_in facts are in the initial state as static info
        initial_state = initial_node.state

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue

            if pred == "goal_in" and len(args) == 2:
                # Check if this goal_in fact is true in the initial state
                if (initial_state >> fact.global_id) & 1:
                    self.package_goal[args[0]] = args[1]
                    all_pkgs.add(args[0])
                    all_rooms.add(args[1])

            elif pred == "door" and len(args) == 3:
                r1, r2, d = args[0], args[1], args[2]
                all_rooms.add(r1)
                all_rooms.add(r2)
                if r1 not in adj: adj[r1] = set()
                adj[r1].add(r2)

            elif pred == "in" and len(args) == 2:
                pkg, room = args[0], args[1]
                all_pkgs.add(pkg)
                all_rooms.add(room)
                if pkg not in self.pkg_in_fact_ids: self.pkg_in_fact_ids[pkg] = {}
                self.pkg_in_fact_ids[pkg][room] = fact.global_id

            elif pred == "holding" and len(args) == 1:
                pkg = args[0]
                all_pkgs.add(pkg)
                self.holding_fact_ids[pkg] = fact.global_id

            elif pred == "rloc" and len(args) == 1:
                room = args[0]
                all_rooms.add(room)
                self.rloc_fact_ids[room] = fact.global_id

        self.rooms = list(all_rooms)
        self.packages = list(all_pkgs)

        # 2. Precompute All-Pairs Shortest Paths (BFS for unit action cost)
        for start_node in self.rooms:
            self.room_dist[start_node] = {r: 1e9 for r in self.rooms}
            self.room_dist[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj.get(u, []):
                    if self.room_dist[start_node][v] == 1e9:
                        self.room_dist[start_node][v] = self.room_dist[start_node][u] + 1
                        queue.append(v)

        h_val = self._compute(initial_node.state)
        return super().initialize(model, h_val)

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

    def _compute(self, state: int) -> int:
        # Current Robot Location
        curr_rloc = None
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                curr_rloc = room
                break

        # Holding status
        held_pkg = None
        for pkg, fid in self.holding_fact_ids.items():
            if (state >> fid) & 1:
                held_pkg = pkg
                break

        unsettled = [] # List of (pkg, current_loc, target_loc)
        for pkg in self.packages:
            target = self.package_goal.get(pkg)
            if not target: continue

            if pkg == held_pkg:
                continue # Handled separately
            
            # Where is the package?
            curr_pkg_loc = None
            for room, fid in self.pkg_in_fact_ids.get(pkg, {}).items():
                if (state >> fid) & 1:
                    curr_pkg_loc = room
                    break
            
            if curr_pkg_loc != target:
                unsettled.append((pkg, curr_pkg_loc, target))

        # Admissible term 1: Manipulation Actions
        # Every unsettled package needs pickup + putdown (2 actions)
        # If held, needs putdown (1 action)
        h_manip = 2 * len(unsettled)
        if held_pkg:
            h_manip += 1

        # Admissible term 2: Travel Actions
        h_travel = 0
        
        # Every unsettled package must traverse the distance from its location to goal
        for _, loc, target in unsettled:
            if loc and target:
                d = self.room_dist.get(loc, {}).get(target, 0)
                h_travel += (d if d < 1e9 else 0)
        
        # Robot must travel to its first destination (either goal of held pkg or loc of an unsettled pkg)
        min_dist_to_first = 0
        if held_pkg:
            target = self.package_goal.get(held_pkg)
            if target and curr_rloc:
                min_dist_to_first = self.room_dist.get(curr_rloc, {}).get(target, 0)
                if min_dist_to_first >= 1e9: min_dist_to_first = 0
        elif unsettled and curr_rloc:
            dists = []
            for _, loc, _ in unsettled:
                if loc:
                    d = self.room_dist.get(curr_rloc, {}).get(loc, 0)
                    if d < 1e9: dists.append(d)
            if dists: min_dist_to_first = min(dists)
        
        h_travel += min_dist_to_first

        # Tie-breaker: prefer smaller task networks
        # (Not included in admissible h, just returned as part of the total)
        return int(h_manip + h_travel)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state)
        # Tie-breaking on depth or task network size is standard
        # but for simplicity we keep it close to our admissible lower bound
        super().update_info(h_val)
        return h_val
