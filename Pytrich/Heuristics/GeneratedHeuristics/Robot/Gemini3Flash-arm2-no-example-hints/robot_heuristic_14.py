from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini3FlashHeuristic14(Heuristic):
    """
    Robot Domain Heuristic
    
    Bottleneck: The single-carry constraint. The robot can only carry one package at a time.
    This means every package delivery is a discrete trip: Go to Package -> Pickup -> Go to Goal -> Putdown.
    
    Admissible Terms:
    1. Delivery Actions: Each undelivered package needs at least 1 pickup and 1 putdown (2 actions).
       If currently holding a package, that package needs 1 putdown (1 action).
    2. Travel Distance: Each delivery requires moving the package from its start to its goal.
       Sum of BFS distances between start/goal locations is an admissible lower bound on 'move' actions.
    3. Robot Inter-trip Travel: The robot must travel to the first package, and between subsequent delivery tasks.
       Minimum distance from robot to the nearest 'work' location (start or goal) is a simple admissible lower bound.
    """
    
    def __init__(self, robot_gemini3flash_heuristic_14="robot_heuristic_14"):
        super().__init__(name=robot_gemini3flash_heuristic_14)
        self.package_targets = {}  # pkg_name -> room_name
        self.rooms = []
        self.packages = []
        self.distances = {}  # (r1, r2) -> int
        
        # Global IDs for fast bitwise checking
        self.rloc_bits = {}      # room_name -> global_id
        self.in_bits = {}        # (pkg_name, room_name) -> global_id
        self.holding_bits = {}   # pkg_name -> global_id
        self.armempty_bit = None

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Facts
        adj = {}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc":
                self.rloc_bits[args[0]] = f.global_id
                if args[0] not in adj: adj[args[0]] = set()
                self.rooms.append(args[0])
            elif pred == "in":
                self.in_bits[(args[0], args[1])] = f.global_id
                if args[0] not in self.packages: self.packages.append(args[0])
            elif pred == "holding":
                self.holding_bits[args[0]] = f.global_id
            elif pred == "armempty":
                self.armempty_bit = f.global_id
            elif pred == "door":
                # args are loc1, loc2, door_id
                r1, r2 = args[0], args[1]
                if r1 not in adj: adj[r1] = set()
                if r2 not in adj: adj[r2] = set()
                adj[r1].add(r2)
                adj[r2].add(r1)

        # 2. Identify Goals from model.goals bitmask
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "in":
                self.package_targets[args[0]] = args[1]

        # 3. Precompute All-Pairs Shortest Paths (BFS)
        for start_r in self.rooms:
            self.distances[(start_r, start_r)] = 0
            queue = deque([(start_r, 0)])
            visited = {start_r}
            while queue:
                u, d = queue.popleft()
                self.distances[(start_r, u)] = d
                if u in adj:
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            queue.append((v, d + 1))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        # 1. Identify robot location
        robot_loc = None
        for r_name, r_bit in self.rloc_bits.items():
            if (state >> r_bit) & 1:
                robot_loc = r_name
                break
        
        # 2. Identify package status
        undelivered = []
        holding_pkg = None
        for pkg in self.packages:
            target = self.package_targets.get(pkg)
            if not target: continue
            
            # Check if held
            hold_bit = self.holding_bits.get(pkg)
            if hold_bit and (state >> hold_bit) & 1:
                holding_pkg = pkg
                continue
            
            # Check current location
            at_target = False
            target_bit = self.in_bits.get((pkg, target))
            if target_bit and (state >> target_bit) & 1:
                at_target = True
            
            if not at_target:
                # Find where it currently is
                current_loc = None
                for r_name in self.rooms:
                    in_bit = self.in_bits.get((pkg, r_name))
                    if in_bit and (state >> in_bit) & 1:
                        current_loc = r_name
                        break
                if current_loc:
                    undelivered.append((pkg, current_loc, target))

        # 3. Calculate Admissible Costs
        h_base = 0
        h_travel = 0
        
        # Penalty for each undelivered package
        h_base += 2 * len(undelivered)
        for _, start, goal in undelivered:
            h_travel += self.distances.get((start, goal), 0)
            
        if holding_pkg:
            h_base += 1 # Only putdown remains
            target = self.package_targets[holding_pkg]
            h_travel += self.distances.get((robot_loc, target), 0)
        elif undelivered:
            # Must travel to at least one package to start
            h_travel += min(self.distances.get((robot_loc, start), 0) for _, start, _ in undelivered)

        # Simple tie-breaker for HTN progress
        h_tie = 0.001 * len(task_network)
        
        return int(h_base + h_travel + h_tie)

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
