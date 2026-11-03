from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic12(Heuristic):
    """
    Robot Domain Heuristic Analysis:
    
    1. Bottleneck Name: The single-arm transport constraint.
       The robot can only carry one package at a time, meaning all package transportation 
       tasks (pickup, move, putdown) must be executed sequentially.

    2. Admissible Lower Bounds:
       - LB1 (Action Count): Each package not at its goal requires at least a pickup and a putdown (2 actions).
         Packages currently being held require 1 action (putdown).
       - LB2 (Travel Cost): Each package not at its goal requires a certain number of move actions
         to be transported from its current location to its goal location. These move counts are 
         additive because the robot can only transport one package at a time.
       - LB3 (Entry Travel): If the robot is empty, it must travel to the nearest unplaced package.
         If the robot is holding a package, it must travel to its destination.

    3. Admissibility Sketch:
       Since the robot must handle each package one by one, the total number of actions is the sum of:
       (Pickup actions) + (Putdown actions) + (Moves while loaded) + (Moves while empty).
       My heuristic sums (pickup+putdown per package) + (min moves for that package's path),
       which is a strict lower bound on the actual action sequence.
    """

    def __init__(self, robot_gemini3flash_heuristic_12):
        super().__init__(name="robot_gemini3flash_heuristic_12")
        self.pkg_to_idx = {}
        self.room_to_idx = {}
        self.pkg_goal_room = {}  # pkg_idx -> room_idx
        self.pkg_in_fact_ids = []  # [pkg_idx][room_idx] -> global_id
        self.pkg_holding_fact_id = [] # [pkg_idx] -> global_id
        self.rloc_fact_ids = []  # [room_idx] -> global_id
        self.room_dist = []
        self.num_pkgs = 0
        self.num_rooms = 0

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        pkg_names = set()
        room_names = set()
        
        # First pass: collect unique objects
        for fact in model.facts:
            pred, args, _ = self._parse_fact_name(fact.name)
            if pred == "in":
                pkg_names.add(args[0])
                room_names.add(args[1])
            elif pred == "rloc":
                room_names.add(args[0])
            elif pred == "goal_in":
                pkg_names.add(args[0])
                room_names.add(args[1])

        sorted_pkgs = sorted(list(pkg_names))
        sorted_rooms = sorted(list(room_names))
        self.pkg_to_idx = {name: i for i, name in enumerate(sorted_pkgs)}
        self.room_to_idx = {name: i for i, name in enumerate(sorted_rooms)}
        self.num_pkgs = len(sorted_pkgs)
        self.num_rooms = len(sorted_rooms)

        self.pkg_in_fact_ids = [[-1] * self.num_rooms for _ in range(self.num_pkgs)]
        self.pkg_holding_fact_id = [-1] * self.num_pkgs
        self.rloc_fact_ids = [-1] * self.num_rooms
        adj = [[] for _ in range(self.num_rooms)]

        # Second pass: map facts and build graph
        for fact in model.facts:
            pred, args, prefix = self._parse_fact_name(fact.name)
            if pred == "in":
                p_idx = self.pkg_to_idx[args[0]]
                r_idx = self.room_to_idx[args[1]]
                self.pkg_in_fact_ids[p_idx][r_idx] = fact.global_id
            elif pred == "holding":
                p_idx = self.pkg_to_idx[args[0]]
                self.pkg_holding_fact_id[p_idx] = fact.global_id
            elif pred == "rloc":
                r_idx = self.room_to_idx[args[0]]
                self.rloc_fact_ids[r_idx] = fact.global_id
            elif pred == "goal_in":
                if (initial_node.state >> fact.global_id) & 1:
                    p_idx = self.pkg_to_idx[args[0]]
                    r_idx = self.room_to_idx[args[1]]
                    self.pkg_goal_room[p_idx] = r_idx
            elif pred == "door":
                # Pre-grounded door facts are always true if they exist in this domain
                r1 = self.room_to_idx[args[0]]
                r2 = self.room_to_idx[args[1]]
                adj[r1].append(r2)

        # BFS for all-pairs shortest paths
        self.room_dist = [[1000] * self.num_rooms for _ in range(self.num_rooms)]
        for i in range(self.num_rooms):
            self.room_dist[i][i] = 0
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if self.room_dist[i][v] == 1000:
                        self.room_dist[i][v] = self.room_dist[i][u] + 1
                        queue.append(v)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        h = 0
        curr_rloc = -1
        for r_idx in range(self.num_rooms):
            if (state >> self.rloc_fact_ids[r_idx]) & 1:
                curr_rloc = r_idx
                break

        min_entry_travel = 1000
        has_unplaced = False
        arm_empty = True # Default for calculation if no holding found

        for p_idx in range(self.num_pkgs):
            if p_idx not in self.pkg_goal_room:
                continue
            
            goal_r_idx = self.pkg_goal_room[p_idx]
            in_goal_f_id = self.pkg_in_fact_ids[p_idx][goal_r_idx]

            # If package is already at goal room
            if (state >> in_goal_f_id) & 1:
                continue

            # Package is not at goal
            has_unplaced = True
            holding_f_id = self.pkg_holding_fact_id[p_idx]
            
            if (state >> holding_f_id) & 1:
                # Robot is holding this package
                arm_empty = False
                h += 1 # putdown cost
                if curr_rloc != -1:
                    h += self.room_dist[curr_rloc][goal_r_idx]
            else:
                # Package is in some room
                curr_pkg_room = -1
                for r_idx in range(self.num_rooms):
                    f_id = self.pkg_in_fact_ids[p_idx][r_idx]
                    if f_id != -1 and (state >> f_id) & 1:
                        curr_pkg_room = r_idx
                        break
                
                if curr_pkg_room != -1:
                    h += 2 # pickup + putdown
                    h += self.room_dist[curr_pkg_room][goal_r_idx]
                    # Track distance for robot to get to this package
                    if curr_rloc != -1:
                        dist_to_pkg = self.room_dist[curr_rloc][curr_pkg_room]
                        if dist_to_pkg < min_entry_travel:
                            min_entry_travel = dist_to_pkg

        if has_unplaced and arm_empty and min_entry_travel != 1000:
            h += min_entry_travel

        # Tie-breakers for guidance
        h_weighted = h + 0.01 * len(task_network)
        
        return int(h_weighted)
