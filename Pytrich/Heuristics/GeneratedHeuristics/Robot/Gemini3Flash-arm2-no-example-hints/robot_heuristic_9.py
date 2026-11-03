from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class RobotGemini3FlashHeuristic9(Heuristic):
    """
    Domain: Robot
    Bottleneck: Single-carry constraint. The robot can only hold one package at a time.
    This constraint serializes all deliveries into independent round-trips.

    Admissible Bounds:
    1. Actions: Every undelivered package requires at least a pickup and a putdown (2 actions).
       If the robot is already holding a package, it only needs a putdown (1 action).
    2. Travel: The robot must travel to the location of each undelivered package, pick it up, 
       and travel to its goal. The sum of these distances (number of move actions) is a 
       tight lower bound because deliveries cannot be combined.
    """
    def __init__(self, robot_gemini3flash_heuristic_9="robot_gemini3flash_heuristic_9"):
        super().__init__(robot_gemini3flash_heuristic_9)
        self.package_goals = {} # pkg -> goal_room
        self.holding_facts = {} # pkg -> fact_id
        self.in_facts = {}      # (pkg, room) -> fact_id
        self.rloc_facts = {}    # room -> fact_id
        self.all_packages = []
        self.all_rooms = []
        self.dists = collections.defaultdict(lambda: collections.defaultdict(lambda: 1000))

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
        room_pairs = []
        unique_rooms = set()
        unique_pkgs = set()

        for f_id in range(self.model.ifacts_init, self.model.ifacts_end + 1):
            f = self.model.facts[f_id]
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == "goal_in":
                self.package_goals[args[0]] = args[1]
                unique_pkgs.add(args[0])
                unique_rooms.add(args[1])
            elif pred == "holding":
                self.holding_facts[args[0]] = f.global_id
                unique_pkgs.add(args[0])
            elif pred == "in":
                self.in_facts[(args[0], args[1])] = f.global_id
                unique_pkgs.add(args[0])
                unique_rooms.add(args[1])
            elif pred == "rloc":
                self.rloc_facts[args[1] if len(args) > 1 else args[0]] = f.global_id
                unique_rooms.add(args[1] if len(args) > 1 else args[0])
            elif pred == "door":
                room_pairs.append((args[0], args[1]))
                unique_rooms.add(args[0])
                unique_rooms.add(args[1])

        self.all_packages = list(unique_pkgs)
        self.all_rooms = list(unique_rooms)

        # Standard BFS for all-pairs shortest paths
        adj = collections.defaultdict(list)
        for r1, r2 in room_pairs:
            adj[r1].append(r2)

        for start_node in self.all_rooms:
            self.dists[start_node][start_node] = 0
            queue = collections.deque([start_node])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in self.dists[start_node]:
                        self.dists[start_node][v] = self.dists[start_node][u] + 1
                        queue.append(v)

        robot_gemini3flash_heuristic_9_initial_h = self._compute(initial_node.state)
        return super().initialize(model, robot_gemini3flash_heuristic_9_initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        robot_gemini3flash_heuristic_9_h_val = self._compute(node.state)
        super().update_info(robot_gemini3flash_heuristic_9_h_val)
        return robot_gemini3flash_heuristic_9_h_val

    def _compute(self, state: int) -> int:
        robot_loc = None
        for r in self.all_rooms:
            if (state >> self.rloc_facts.get(r, 0)) & 1:
                robot_loc = r
                break
        if not robot_loc: return 0

        held_pkg = None
        for p in self.all_packages:
            if (state >> self.holding_facts.get(p, 0)) & 1:
                held_pkg = p
                break

        h = 0
        undelivered = []
        
        for p in self.all_packages:
            if p == held_pkg: continue
            target_room = self.package_goals.get(p)
            if not target_room: continue
            
            is_at_goal = (state >> self.in_facts.get((p, target_room), 0)) & 1
            if not is_at_goal:
                p_loc = None
                for r in self.all_rooms:
                    if (state >> self.in_facts.get((p, r), 0)) & 1:
                        p_loc = r
                        break
                if p_loc:
                    undelivered.append((p, p_loc, target_room))

        if held_pkg:
            target = self.package_goals.get(held_pkg)
            h += self.dists[robot_loc][target] + 1
            for p, p_loc, g_room in undelivered:
                h += self.dists[p_loc][g_room] + 2
        else:
            for p, p_loc, g_room in undelivered:
                h += self.dists[p_loc][g_room] + 2
            if undelivered:
                h += min(self.dists[robot_loc][p_loc] for p, p_loc, g_room in undelivered)

        return h
