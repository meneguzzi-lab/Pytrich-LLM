"""
Domain-dependent heuristic for the Robot delivery domain.

Bottleneck Analysis:
The bottleneck in this domain is the single robot arm. Because the robot can only carry one package at a time,
it must deliver packages serially. This means the total plan length is at least the sum of picking up, 
moving, and dropping off every undelivered package, plus the transit time for the robot to move 
between these locations.

Admissibility Sketches:
1. Package Delivery Actions: Every undelivered package requires at least one pickup and one putdown action (cost 2).
   If the package is already held, it requires only a putdown (cost 1). Since these actions are mandatory 
   and distinct for each package (due to the single arm), this is a lower bound.
2. Package Travel: Every package must be moved from its current location to its target location. 
   The number of 'move' actions required for a package is the shortest path distance in the room-door graph.
   Since the robot can only carry one item, these move sequences are disjoint for different packages.
3. Initial Transit: If the robot is not currently at a package's location or its destination (if holding one), 
   it must move there. The distance to the nearest required interaction point is a lower bound on initial transit.

These terms are combined additively because they represent distinct actions (pickup != putdown != move).
"""

import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic15(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_15=None, name="RobotGemini3FlashHeuristic15"):
        super().__init__(name=name)
        self.pkg_targets = {}  # pkg -> target_room
        self.pkg_in_facts = defaultdict(dict)  # pkg -> {room: fact_id}
        self.pkg_holding_facts = {}  # pkg -> fact_id
        self.rloc_facts = {}  # room -> fact_id
        self.armempty_id = -1
        self.adj = defaultdict(set)
        self.distances = {}
        self.rooms = set()
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # 1. Parse goals from model.goals bitmask to find targets
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "in" and len(args) == 2:
                self.pkg_targets[args[0]] = args[1]

        # 2. Parse facts for indexing
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "in" and len(args) == 2:
                self.pkg_in_facts[args[0]][args[1]] = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self.pkg_holding_facts[args[0]] = fact.global_id
            elif pred == "rloc" and len(args) == 1:
                self.rloc_facts[args[0]] = fact.global_id
                self.rooms.add(args[0])
            elif pred == "armempty":
                self.armempty_id = fact.global_id
            elif pred == "door" and len(args) == 3:
                u, v = args[0], args[1]
                self.adj[u].add(v)
                self.adj[v].add(u)

        # 3. Precompute all-pairs shortest paths on the room graph (BFS)
        self._compute_distances()

        self.preprocessing_time = time.time() - start_time
        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

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

    def _compute_distances(self):
        for start_room in self.rooms:
            self.distances[(start_room, start_room)] = 0
            queue = deque([(start_room, 0)])
            visited = {start_room}
            while queue:
                curr, d = queue.popleft()
                self.distances[(start_room, curr)] = d
                for neighbor in self.adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

    def _compute(self, state, task_network):
        # Determine robot location
        robot_room = None
        for r, fid in self.rloc_facts.items():
            if (state >> fid) & 1:
                robot_room = r
                break

        if not robot_room:
            return 0

        h = 0
        undelivered = []
        holding_pkg = None

        # Identify state of each package
        for pkg, target_room in self.pkg_targets.items():
            # Is robot holding it?
            held_fid = self.pkg_holding_facts.get(pkg)
            if held_fid is not None and (state >> held_fid) & 1:
                holding_pkg = pkg
                h += 1  # 1 putdown
                h += self.distances.get((robot_room, target_room), 0)  # travel to target
                continue

            # Is it at target?
            in_target_fid = self.pkg_in_facts[pkg].get(target_room)
            if in_target_fid is not None and (state >> in_target_fid) & 1:
                continue

            # Needs delivery
            undelivered.append(pkg)
            h += 2  # 1 pickup + 1 putdown
            
            # Find where it currently is
            pkg_room = None
            for r, fid in self.pkg_in_facts[pkg].items():
                if (state >> fid) & 1:
                    pkg_room = r
                    break
            
            if pkg_room:
                h += self.distances.get((pkg_room, target_room), 0)

        # Initial transit to first task
        if not holding_pkg and undelivered:
            min_transit = float('inf')
            for pkg in undelivered:
                pkg_room = None
                for r, fid in self.pkg_in_facts[pkg].items():
                    if (state >> fid) & 1:
                        pkg_room = r
                        break
                if pkg_room:
                    d = self.distances.get((robot_room, pkg_room), 0)
                    if d < min_transit:
                        min_transit = d
            if min_transit != float('inf'):
                h += min_transit

        # Task network overhead: ensure we account for abstract tasks if state h is low
        # but for this domain, state-based delivery cost is usually dominant and admissible.
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
