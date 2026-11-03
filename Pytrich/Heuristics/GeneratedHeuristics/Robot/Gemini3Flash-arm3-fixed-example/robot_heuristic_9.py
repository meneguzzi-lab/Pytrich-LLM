"""
Domain-dependent heuristic for the Robot planning domain.

Bottleneck Analysis:
The single-arm capacity of the robot is the primary bottleneck. Because the robot can only carry 
one package at a time, deliveries are strictly serialized. This means that for every undelivered 
package, the robot must perform a unique sequence of move, pickup, and putdown actions that 
cannot be shared with other packages.

Admissibility Sketch:
1. h_ops: Every misplaced package requires at least 1 pickup and 1 putdown (2 actions). 
   If already held, it needs 1 putdown. These are disjoint for each package due to the arm bottleneck.
2. h_moves: Every package must be moved from its current location to its goal. 
   Since only one package can be held, the distance 'dist(origin, goal)' for all misplaced 
   packages is a lower bound on mandatory moves. Additionally, the robot must move to the 
   first package it intends to pick up.
3. h_total = h_ops + h_moves is admissible as it counts distinct required physical actions.
"""

import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini3FlashHeuristic9(Heuristic):
    def __init__(self, robot_gemini3flash_heuristic_9=None, name="RobotGemini3FlashHeuristic9"):
        super().__init__(name=name)
        self.goals_map = {}  # pkg -> goal_room
        self.package_ids = []
        self.room_ids = []
        self.rloc_fact_ids = {} # room -> global_id
        self.holding_fact_ids = {} # pkg -> global_id
        self.in_fact_ids = {} # (pkg, room) -> global_id
        self.in_facts_by_package = defaultdict(list)
        self.dist_matrix = {}
        self.adj = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute_h(initial_node.state)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute_h(node.state)
        super().update_info(h_value)
        return h_value

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
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        packages = set()
        rooms = set()
        
        # 1. Map Goal Destinations (check goal state and initial state for goal_in facts)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_fact_names:
                pred, args = self._parse_fact_name(name)
                if pred in ["goal_in", "in"] and len(args) == 2:
                    self.goals_map[args[0]] = args[1]

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "goal_in" and (self.model.initial_state >> fact.global_id) & 1:
                self.goals_map[args[0]] = args[1]
            
            if pred == "rloc":
                rooms.add(args[0])
                self.rloc_fact_ids[args[0]] = fact.global_id
            elif pred == "holding":
                packages.add(args[0])
                self.holding_fact_ids[args[0]] = fact.global_id
            elif pred == "in":
                packages.add(args[0])
                rooms.add(args[1])
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
                self.in_facts_by_package[args[0]].append((fact.global_id, args[1]))
            elif pred == "door":
                self.adj.append((args[0], args[1]))

        self.package_ids = list(packages)
        self.room_ids = list(rooms)
        
        # 2. Precompute Room Distances (BFS)
        graph = defaultdict(list)
        for r1, r2 in self.adj: graph[r1].append(r2)
        
        for start_r in self.room_ids:
            dists = {start_r: 0}
            queue = deque([start_r])
            while queue:
                curr = queue.popleft()
                for neighbor in graph[curr]:
                    if neighbor not in dists:
                        dists[neighbor] = dists[curr] + 1
                        queue.append(neighbor)
            for target_r, d in dists.items():
                self.dist_matrix[(start_r, target_r)] = d

    def _compute_h(self, state: int) -> int:
        robot_loc = None
        for room, fid in self.rloc_fact_ids.items():
            if (state >> fid) & 1:
                robot_loc = room
                break
        if not robot_loc: return 0

        h_ops = 0
        h_moves = 0
        holding_pkg = None
        misplaced_locs = []

        # Check what is currently held
        for p in self.package_ids:
            if (state >> self.holding_fact_ids.get(p, -1)) & 1:
                holding_pkg = p
                break

        # Calculate costs for all packages needing delivery
        for p in self.package_ids:
            dest = self.goals_map.get(p)
            if not dest: continue

            if p == holding_pkg:
                h_ops += 1
                h_moves += self.dist_matrix.get((robot_loc, dest), 10)
            else:
                # Is it at its destination?
                in_dest_fid = self.in_fact_ids.get((p, dest), -1)
                at_goal = (state >> in_dest_fid) & 1 if in_dest_fid != -1 else False
                if not at_goal:
                    # Find its current location
                    p_loc = None
                    for fid, r in self.in_facts_by_package[p]:
                        if (state >> fid) & 1:
                            p_loc = r
                            break
                    if p_loc:
                        h_ops += 2
                        h_moves += self.dist_matrix.get((p_loc, dest), 10)
                        misplaced_locs.append(p_loc)

        # If robot is free, add distance to the nearest misplaced package
        if not holding_pkg and misplaced_locs:
            h_moves += min(self.dist_matrix.get((robot_loc, loc), 10) for loc in misplaced_locs)

        return h_ops + h_moves
