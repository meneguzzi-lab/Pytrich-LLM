from typing import List, Dict, Set, Tuple, Optional, Any
import collections
import heapq

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGemini3ProHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.
    
    This heuristic estimates the cost to achieve the goal state from the current state
    by analyzing the location of packages and the robot.
    
    It performs the following:
    1. Static Analysis: Builds a connectivity graph of rooms from static 'door' facts and
       precomputes all-pairs shortest paths (unweighted BFS, assuming open/close cost is manageable).
    2. Dynamic State Analysis:
       - Identifies the robot's current location.
       - Identifies the current location of all packages.
       - Identifies which packages are currently held.
       - Identifies goal locations from the model's goal bitmask.
    3. Cost Estimation:
       - Sums the movement and action costs for each misplaced package.
       - Cost = Distance(package_loc, goal_loc) + 2 (pickup + putdown).
       - If the robot is holding a package, adds Distance(robot_loc, goal_loc) + 1 (putdown).
       - Adds a greedy estimate for the robot to travel to the nearest pending package source.
    
    The heuristic is admissible if we relax the constraint of the robot returning to pick up
    subsequent packages (treating them independently) and ignore door opening costs in the distance metric.
    """

    def __init__(self, model, robot_gemini3pro_heuristic_6=None, **kwargs):
        super().__init__(model)
        self.model = model
        
        # --- 1. Parse Static Domain Information (Map Topology) ---
        self.rooms: Set[str] = set()
        self.doors: List[Tuple[str, str]] = []
        
        # Mappings for fast state lookups
        self.rloc_facts: List[Tuple[int, str]] = []     # (global_id, room)
        self.holding_facts: List[Tuple[int, str]] = []  # (global_id, obj)
        self.in_facts: Dict[str, List[Tuple[int, str]]] = {} # obj -> [(global_id, room)]
        self.goal_locs: Dict[str, str] = {} # obj -> room

        # Iterate over all grounded facts in the model to build indices and the map
        for fact in self.model.facts:
            name = fact.name
            predicate, args = self._parse_fact_name(name)
            
            if predicate == "door":
                # Expecting: door[loc1, loc2, d]
                if len(args) == 3:
                    r1, r2, _ = args
                    self.rooms.add(r1)
                    self.rooms.add(r2)
                    self.doors.append((r1, r2))
            
            elif predicate == "rloc":
                # Expecting: rloc[loc]
                if len(args) == 1:
                    loc = args[0]
                    self.rloc_facts.append((fact.global_id, loc))
                    self.rooms.add(loc)
            
            elif predicate == "holding":
                # Expecting: holding[obj]
                if len(args) == 1:
                    obj = args[0]
                    self.holding_facts.append((fact.global_id, obj))
            
            elif predicate == "in":
                # Expecting: in[obj, loc]
                if len(args) == 2:
                    obj, loc = args
                    if obj not in self.in_facts:
                        self.in_facts[obj] = []
                    self.in_facts[obj].append((fact.global_id, loc))
                    self.rooms.add(loc)

        # --- 2. Precompute Distances ---
        # We use unweighted BFS since 'move' cost is 1. 
        # Ignoring 'open' cost (which adds 1) keeps it admissible.
        self.dist_matrix = self._compute_all_pairs_shortest_paths(self.rooms, self.doors)

        # --- 3. Parse Goal State ---
        # model.goals is a bitwise integer representing the set of facts that must be true.
        # We map these back to specific target locations for packages.
        goal_mask = self.model.goals
        if goal_mask > 0:
            for obj, fact_list in self.in_facts.items():
                for fid, loc in fact_list:
                    if (goal_mask >> fid) & 1:
                        self.goal_locs[obj] = loc

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parses a fact string like '+in[o1,r1]' into ('in', ['o1', 'r1']).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return predicate, args
        return None, []

    def _compute_all_pairs_shortest_paths(self, rooms: Set[str], doors: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
        """
        Computes APSP using unweighted BFS on the room connectivity graph.
        """
        adj = collections.defaultdict(list)
        for r1, r2 in doors:
            adj[r1].append(r2)
            # Note: Explicit symmetric facts (door r2 r1 d) usually exist in HDDL.
            # If not, the graph is directed, which is correct.

        dists = {}
        for start_node in rooms:
            dists[start_node] = {}
            # BFS
            queue = collections.deque([(start_node, 0)])
            visited = {start_node}
            dists[start_node][start_node] = 0
            
            while queue:
                curr, d = queue.popleft()
                dists[start_node][curr] = d
                
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, d + 1))
        return dists

    def get_dist(self, r1: str, r2: str) -> int:
        if r1 == r2:
            return 0
        if r1 in self.dist_matrix and r2 in self.dist_matrix[r1]:
            return self.dist_matrix[r1][r2]
        return 99999  # Unreachable penalty

    def compute(self, node, **kwargs) -> float:
        # Handle both Node object and raw state integer
        state = node.state if hasattr(node, 'state') else node
        
        # 1. Determine Robot Location
        robot_loc = None
        for fid, room in self.rloc_facts:
            if (state >> fid) & 1:
                robot_loc = room
                break
        
        # If robot location is unknown (invalid state), return safe max
        if robot_loc is None:
            return float('inf')

        # 2. Determine Holding Status
        held_obj = None
        for fid, obj in self.holding_facts:
            if (state >> fid) & 1:
                held_obj = obj
                break

        h_val = 0
        pending_sources = []

        # 3. Calculate Cost for Goals
        for obj, target_loc in self.goal_locs.items():
            # Check if obj is already at target_loc in current state
            is_at_goal = False
            
            # We check the specific bit for in[obj, target_loc]
            # To do this fast, we scan the pre-cached indices for this object
            if obj in self.in_facts:
                for fid, loc in self.in_facts[obj]:
                    if loc == target_loc:
                        if (state >> fid) & 1:
                            is_at_goal = True
                        break
            
            if is_at_goal:
                continue

            # If not at goal, add cost
            if held_obj == obj:
                # Robot has it. Cost to deliver: dist(robot, target) + putdown(1)
                d = self.get_dist(robot_loc, target_loc)
                h_val += d + 1
                # This task is effectively "active".
            else:
                # Robot doesn't have it. Find where it is.
                current_loc = None
                if obj in self.in_facts:
                    for fid, loc in self.in_facts[obj]:
                        if (state >> fid) & 1:
                            current_loc = loc
                            break
                
                if current_loc:
                    # Cost: Move package from current to target
                    # Pickup(1) + Move(dist) + Putdown(1)
                    # We add dist + 2
                    d = self.get_dist(current_loc, target_loc)
                    h_val += d + 2
                    pending_sources.append(current_loc)
                else:
                    # Object location unknown (lost?) or parsing error.
                    # In a valid state, every package is somewhere or held.
                    pass

        # 4. Robot Travel Cost to Next Task
        # If robot is not holding anything, it must travel to a package source.
        # We add the minimum distance to any pending package source.
        if held_obj is None and pending_sources:
            min_travel = float('inf')
            for src in pending_sources:
                d = self.get_dist(robot_loc, src)
                if d < min_travel:
                    min_travel = d
            
            if min_travel != float('inf'):
                h_val += min_travel
        
        return float(h_val)
