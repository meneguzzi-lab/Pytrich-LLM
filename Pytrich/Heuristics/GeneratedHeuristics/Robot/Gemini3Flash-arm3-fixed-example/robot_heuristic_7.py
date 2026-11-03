"""
Domain-dependent heuristic for the Robot HDDL domain.

Bottleneck Analysis:
The primary bottleneck in this domain is the robot's single arm. The robot can only 
carry one package at a time. This serializes the delivery process: for every package, 
the robot must travel to its location, pick it up, travel to the goal location, 
and put it down. Movement between rooms may also be restricted by closed doors.

Admissibility Sketches:
1. Package Delivery Cost: For each package not at its goal, we count at least one pickup,
   one putdown, and the shortest-path distance (moves) between its current location and 
   its goal location. Because only one package can be carried at a time, these costs 
   are additive (admissible).
2. Travel to Package: If the robot is not holding a package, it must travel to the 
   location of the first package it intends to deliver. The minimum distance to 
   any unsettled package's location is an admissible lower bound for this initial travel.
3. Door Costs: While closed doors must be opened, adding '1' for every closed door 
   can be tricky to keep additive with movement costs. We use the shortest path distance 
   as the base move cost, which is always admissible.
"""

import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic7(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_7=None, name="RobotGemini3FlashHeuristic7"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Static domain maps
        self.rooms = []
        self.packages = []
        self.dist = defaultdict(lambda: defaultdict(lambda: 1000))
        self.pkg_goal_room = {}  # pkg -> target_room
        
        # Fact ID maps for fast bitwise checking
        self.rloc_facts = {}     # room -> fact_object
        self.in_facts = defaultdict(dict)  # pkg -> {room: fact_object}
        self.holding_facts = {}  # pkg -> fact_object
        self.armempty_fact = None

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # 1. Parse all facts and categorize them
        adj = defaultdict(list)
        pkgs_seen = set()
        rooms_seen = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "rloc":
                self.rloc_facts[args[0]] = fact
                rooms_seen.add(args[0])
            elif pred == "in":
                self.in_facts[args[0]][args[1]] = fact
                pkgs_seen.add(args[0])
                rooms_seen.add(args[1])
            elif pred == "holding":
                self.holding_facts[args[0]] = fact
                pkgs_seen.add(args[0])
            elif pred == "armempty":
                self.armempty_fact = fact
            elif pred == "door":
                adj[args[0]].append(args[1])
                rooms_seen.add(args[0])
                rooms_seen.add(args[1])
            elif pred == "goal_in":
                # Use the static goal_in facts defined in the state
                if (model.initial_state >> fact.global_id) & 1:
                    self.pkg_goal_room[args[0]] = args[1]
        
        self.rooms = list(rooms_seen)
        self.packages = list(pkgs_seen)

        # 2. Precompute shortest-path distance (moves) using BFS
        for start_node in self.rooms:
            self.dist[start_node][start_node] = 0
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in self.dist[start_node]:
                        self.dist[start_node][v] = self.dist[start_node][u] + 1
                        queue.append(v)

        self.preprocessing_time = time.time() - start_time
        h_val = self._compute(initial_node.state, initial_node.task_network)
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

    def _compute(self, state, task_network):
        # 1. Identify robot location
        current_rloc = None
        for room, fact in self.rloc_facts.items():
            if (state >> fact.global_id) & 1:
                current_rloc = room
                break
        
        if current_rloc is None: return 0

        # 2. Identify held package
        held_pkg = None
        for pkg, fact in self.holding_facts.items():
            if (state >> fact.global_id) & 1:
                held_pkg = pkg
                break
        
        h = 0
        unsettled_pkgs_locs = []

        # 3. Add delivery cost for currently held package
        if held_pkg:
            target = self.pkg_goal_room.get(held_pkg)
            if target:
                h += self.dist[current_rloc][target] + 1  # travel to goal + putdown
        
        # 4. Add delivery costs for all other packages not at goal
        for pkg, goal_room in self.pkg_goal_room.items():
            if pkg == held_pkg: continue
            
            # Find current location of package
            pkg_loc = None
            for room, fact in self.in_facts[pkg].items():
                if (state >> fact.global_id) & 1:
                    pkg_loc = room
                    break
            
            if pkg_loc and pkg_loc != goal_room:
                # Package needs delivery: pickup(1) + travel to goal(dist) + putdown(1)
                h += self.dist[pkg_loc][goal_room] + 2
                unsettled_pkgs_locs.append(pkg_loc)
            elif pkg_loc is None and pkg != held_pkg:
                # Package is neither in a room nor held? Should not happen in valid state
                pass

        # 5. Add cost to reach the next package if not currently holding one
        if not held_pkg and unsettled_pkgs_locs:
            min_travel = min(self.dist[current_rloc][loc] for loc in unsettled_pkgs_locs)
            h += min_travel

        # 6. Tie-breaker for task network progress
        h += 0.001 * len(task_network)

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value) if h_value > 10000 else round(h_value)
