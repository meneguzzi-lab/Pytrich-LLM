from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque, defaultdict
import time

class RobotGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Robot domain.

    1. Bottleneck Analysis:
       The core bottleneck is the robot's single-arm capacity. Since the robot can only carry one
       package at a time, the process of delivering N packages is strictly serialized.

    2. Admissible Lower Bounds:
       Term A (Delivery Cost): Each package not at its target requires at least one 'pickup' and
       one 'putdown' action (Cost = 2). If the robot is already holding a package, that package
       only requires one 'putdown' (Cost = 1).

       Term B (Carriage Distance): Every misplaced package must be carried from its current
       location to its goal location. The number of 'move' actions is at least the shortest
       path distance in the room-door graph.

       Term C (Inter-task Travel): The robot must travel to the location of the first package
       and then between subsequent delivery points. The distance from the robot's current position
       to the next relevant location (package source or goal) is a mandatory lower bound.

    These terms are additive because the robot cannot carry multiple packages or move while
    performing other actions simultaneously.
    """

    def __init__(self, robot_gemini3flash_heuristic_1=None, name="RobotGemini3FlashHeuristic1"):
        super().__init__(name=name)
        self.pkg_goal = {} # package -> goal_room
        self.rooms = set()
        self.packages = set()
        self.adj = defaultdict(list)
        self.distances = defaultdict(dict)
        self.rloc_fact_ids = {} # room -> global_id
        self.in_fact_ids = defaultdict(dict) # pkg -> room -> global_id
        self.holding_fact_ids = {} # pkg -> global_id

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._calculate_heuristic(initial_node.state)
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

    def _preprocess(self):
        # 1. Map all facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "rloc":
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_ids[room] = fact.global_id
            elif pred == "in":
                pkg, room = args
                self.packages.add(pkg)
                self.rooms.add(room)
                self.in_fact_ids[pkg][room] = fact.global_id
            elif pred == "holding":
                pkg = args[0]
                self.packages.add(pkg)
                self.holding_fact_ids[pkg] = fact.global_id
            elif pred == "door":
                r1, r2, d = args
                self.adj[r1].append(r2)
                self.adj[r2].append(r1)
            elif pred == "goal_in":
                # Helper predicate in state
                pkg, room = args
                self.pkg_goal[pkg] = room

        # 2. Extract targets from model goals if goal_in wasn't sufficient
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "in" and len(args) == 2:
                self.pkg_goal[args[0]] = args[1]

        # 3. Precompute distances using BFS
        for start_room in self.rooms:
            self.distances[start_room] = {r: 1000 for r in self.rooms}
            self.distances[start_room][start_room] = 0
            queue = deque([(start_room, 0)])
            while queue:
                curr, d = queue.popleft()
                for neighbor in self.adj[curr]:
                    if self.distances[start_room][neighbor] == 1000:
                        self.distances[start_room][neighbor] = d + 1
                        queue.append((neighbor, d + 1))

    def _get_dist(self, r1, r2):
        if r1 == r2: return 0
        return self.distances.get(r1, {}).get(r2, 1000)

    def _calculate_heuristic(self, state: int) -> int:
        h = 0
        # Identify robot state
        robot_loc = None
        for room, f_id in self.rloc_fact_ids.items():
            if (state >> f_id) & 1:
                robot_loc = room
                break

        # Identify held package
        held_pkg = None
        for pkg in self.packages:
            f_id = self.holding_fact_ids.get(pkg)
            if f_id is not None and (state >> f_id) & 1:
                held_pkg = pkg
                break

        # Identify misplaced packages
        misplaced = [] # list of (pkg, current_room)
        for pkg in self.packages:
            if pkg == held_pkg: continue
            target_room = self.pkg_goal.get(pkg)
            if not target_room: continue
            
            # Find current room
            pkg_room = None
            for r, f_id in self.in_fact_ids[pkg].items():
                if (state >> f_id) & 1:
                    pkg_room = r
                    break
            
            if pkg_room and pkg_room != target_room:
                misplaced.append((pkg, pkg_room))

        # Calculation
        if held_pkg:
            target = self.pkg_goal.get(held_pkg)
            h += 1 + self._get_dist(robot_loc, target)
            for p, p_loc in misplaced:
                h += 2 + self._get_dist(p_loc, self.pkg_goal[p])
        elif misplaced:
            # Robot must move to a package first
            dist_to_pkg = min(self._get_dist(robot_loc, p_loc) for p, p_loc in misplaced)
            h += dist_to_pkg
            for p, p_loc in misplaced:
                h += 2 + self._get_dist(p_loc, self.pkg_goal[p])

        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
