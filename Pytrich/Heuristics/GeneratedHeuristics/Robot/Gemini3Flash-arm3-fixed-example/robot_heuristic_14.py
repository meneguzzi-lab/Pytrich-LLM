"""
Domain-dependent heuristic for the Robot planning domain.

Bottleneck Analysis:
The primary bottleneck in this domain is the robot's single-arm capacity. The robot can carry 
only one package at a time. This serializes all transport tasks: for every package not at 
its target location, the robot must travel to its current location, pick it up, travel to 
the goal location, and put it down. Closed doors further serialize movement, requiring 
an additional 'open' action before passing through.

Admissibility Sketches:
1. Transport Action Count: Every package 'p' not at its target room requires a 'putdown' action (1). 
   If the package is not currently held, it also requires a 'pickup' action (1). 
   Summing (2 actions per misplaced package, 1 if already held) is a lower bound on robot operations.
2. Travel Cost: To perform the first pickup or putdown, the robot must move to the target room. 
   The shortest-path distance (number of moves) from the current robot location to the 
   nearest package room or held-package goal is a lower bound on travel cost.
3. Combined: h = (Action Count) + (Travel to First Task). This is admissible as it ignores 
   the travel required between subsequent packages and the costs of opening closed doors.
"""

import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic14(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_14=None, name="RobotGemini3FlashHeuristic14"):
        super().__init__(name=name)
        self.pkg_to_goal = {}       # package -> room
        self.rloc_fact_ids = {}    # room -> global_id
        self.pkg_in_fact_ids = defaultdict(list) # package -> [(room, global_id)]
        self.holding_fact_ids = {} # package -> global_id
        self.armempty_id = None
        self.dist = {}             # room1 -> room2 -> int (moves)
        self.rooms = set()
        self.packages = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        initial_state = initial_node.state
        
        # 1. Parse all facts to map structure
        adj = defaultdict(list)
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "goal_in":
                # Check initial state to find static goal requirements
                if (initial_state >> fact.global_id) & 1:
                    self.pkg_to_goal[args[0]] = args[1]
            elif pred == "rloc":
                self.rooms.add(args[0])
                self.rloc_fact_ids[args[0]] = fact.global_id
            elif pred == "in":
                self.packages.add(args[0])
                self.rooms.add(args[1])
                self.pkg_in_fact_ids[args[0]].append((args[1], fact.global_id))
            elif pred == "holding":
                self.packages.add(args[0])
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "door":
                r1, r2, d = args[0], args[1], args[2]
                adj[r1].append(r2)
                self.rooms.add(r1)
                self.rooms.add(r2)

        # 2. Precompute room distances via BFS
        for start_node in self.rooms:
            self.dist[start_node] = {start_node: 0}
            queue = deque([(start_node, 0)])
            while queue:
                u, d = queue.popleft()
                for v in adj[u]:
                    if v not in self.dist[start_node]:
                        self.dist[start_node][v] = d + 1
                        queue.append((v, d + 1))

        h_value = self._calculate_heuristic(initial_state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _calculate_heuristic(self, state: int, task_network) -> int:
        # 1. Identify robot location
        curr_rloc = None
        for r, f_id in self.rloc_fact_ids.items():
            if (state >> f_id) & 1:
                curr_rloc = r
                break

        # 2. Identify held package and misplaced packages
        held_pkg = None
        for p in self.packages:
            f_id = self.holding_fact_ids.get(p)
            if f_id is not None and (state >> f_id) & 1:
                held_pkg = p
                break

        misplaced = set()
        current_locs = {}
        for p in self.packages:
            if p == held_pkg:
                misplaced.add(p)
                continue
            
            p_room = None
            for r, f_id in self.pkg_in_fact_ids.get(p, []):
                if (state >> f_id) & 1:
                    p_room = r
                    break
            
            goal_room = self.pkg_to_goal.get(p)
            if p_room != goal_room:
                misplaced.add(p)
                current_locs[p] = p_room

        if not misplaced:
            return 0

        # 3. Compute Admissible Costs
        # Term 1: Action lower bound (2 per package to be picked+dropped, 1 if already held)
        h_actions = 2 * len(misplaced)
        if held_pkg:
            h_actions -= 1

        # Term 2: Travel to the first actionable room
        # If holding, must travel to goal. If not, must travel to any misplaced package room.
        dist_to_task = 0
        if curr_rloc:
            if held_pkg:
                target_room = self.pkg_to_goal.get(held_pkg)
                if target_room:
                    dist_to_task = self.dist.get(curr_rloc, {}).get(target_room, 1)
            else:
                # Distance to nearest package to pick up
                min_d = 1000
                for p in misplaced:
                    loc = current_locs.get(p)
                    if loc:
                        d = self.dist.get(curr_rloc, {}).get(loc, 1)
                        if d < min_d: min_d = d
                dist_to_task = min_d if min_d != 1000 else 0

        # Tie-breaker: prefer shorter task networks
        tie_breaker = 0.001 * len(task_network)

        return int(h_actions + dist_to_task) + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
