from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict, deque
import time

class RobotGemini3FlashHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.
    
    Bottleneck Analysis:
    The primary bottleneck in this domain is the single-arm constraint (armempty / holding).
    The robot can only carry one package at a time, meaning each package must be delivered 
    serially. This makes costs strictly additive across packages.

    Admissible Lower Bounds:
    1. Handling Cost: Each package not at its goal destination requires exactly one 'pickup' and 
       one 'putdown' action (+2). If a package is already being held, it only requires 'putdown' (+1).
    2. Travel Cost: The robot must move to each package's location (retrieval) and then to its 
       destination (delivery). Since packages are handled one-by-one, the 'move' actions required 
       to transport each package from its current location to its goal are disjoint and additive.
    3. Retrieval Cost: If no package is held, the robot must move to the nearest package that 
       still needs to be delivered.

    The final heuristic is: H = (Sum of Handling Costs) + (Sum of Delivery Distances) + (Min Retrieval Distance).
    """

    def __init__(self, robot_gemini3flash_heuristic_17=None, name="RobotGemini3FlashHeuristic17"):
        super().__init__(name=name)
        self.package_goals = {}     # package -> goal_room
        self.adj = defaultdict(set) # room -> set of neighbor rooms
        self.distances = {}         # room -> {room -> dist}
        self.in_facts = defaultdict(dict)  # pkg -> {room -> global_id}
        self.holding_facts = {}     # pkg -> global_id
        self.rloc_facts = {}        # room -> global_id
        self.packages = set()
        self.rooms = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse goals (where packages must end up)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "in" and len(args) == 2:
                self.package_goals[args[0]] = args[1]

        # 2. Extract static domain structure and fact IDs
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == "door":
                # door(loc1, loc2, d) - door facts are static in these problems
                r1, r2 = args[0], args[1]
                if (self.model.initial_state >> f.global_id) & 1:
                    self.adj[r1].add(r2)
                    self.rooms.add(r1)
                    self.rooms.add(r2)
            elif pred == "in":
                self.in_facts[args[0]][args[1]] = f.global_id
                self.packages.add(args[0])
                self.rooms.add(args[1])
            elif pred == "holding":
                self.holding_facts[args[0]] = f.global_id
                self.packages.add(args[0])
            elif pred == "rloc":
                self.rloc_facts[args[0]] = f.global_id
                self.rooms.add(args[0])

        # 3. Precompute all-pairs shortest paths (BFS)
        for start_room in self.rooms:
            self.distances[start_room] = {r: 999 for r in self.rooms}
            self.distances[start_room][start_room] = 0
            queue = deque([start_room])
            while queue:
                curr = queue.popleft()
                for neighbor in self.adj[curr]:
                    if self.distances[start_room][neighbor] == 999:
                        self.distances[start_room][neighbor] = self.distances[start_room][curr] + 1
                        queue.append(neighbor)

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
        # Determine robot location
        robot_loc = None
        for r, fid in self.rloc_facts.items():
            if (state >> fid) & 1:
                robot_loc = r
                break

        h = 0
        pkgs_to_retrieve = []
        held_pkg = None

        for p in self.packages:
            goal_r = self.package_goals.get(p)
            if not goal_r: continue

            # Check if held
            is_held = (state >> self.holding_facts.get(p, -1)) & 1 if p in self.holding_facts else False
            if is_held:
                held_pkg = p
                h += 1 # putdown cost
                if robot_loc:
                    h += self.distances[robot_loc].get(goal_r, 0)
                continue

            # Find current room
            curr_r = None
            for r, fid in self.in_facts[p].items():
                if (state >> fid) & 1:
                    curr_r = r
                    break
            
            if curr_r and curr_r != goal_r:
                h += 2 # pickup + putdown costs
                h += self.distances[curr_r].get(goal_r, 0) # delivery distance
                pkgs_to_retrieve.append(curr_r)

        # Add retrieval cost for the first package if not holding anything
        if not held_pkg and pkgs_to_retrieve and robot_loc:
            h += min(self.distances[robot_loc].get(r, 0) for r in pkgs_to_retrieve)

        # Tie-breaker: prefer smaller task networks to discourage infinite recursion
        h += 0.001 * len(task_network)
        
        return int(h) if h < 999 else 999

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
