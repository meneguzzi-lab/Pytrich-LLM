import time
import heapq
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic8(Heuristic):
    """
    A domain-dependent heuristic for the 'robot' HDDL domain.

    Rationale:
    This heuristic calculates an admissible estimate of the remaining cost (number of actions)
    to achieve all package delivery goals. It is based on the h_max heuristic, which computes
    the cost to achieve each individual unmet goal from the current state and takes the maximum
    of these costs. The maximum of costs for independent subproblems is an admissible lower
    bound on the cost to solve the combined problem.

    Key Components:
    1.  **Preprocessing (`initialize`)**: 
        - Builds a graph representation of the rooms and doors from the static predicates.
        - Caches the goal location for each package.
        - This one-time setup makes per-node evaluation faster.

    2.  **State-Dependent Calculation (`__call__`)**: 
        - The heuristic value is computed based on the current state of the world (`node.state`).
        - It parses the state to determine the robot's location, what it's holding, the locations of all packages, and which doors are closed.

    3.  **Shortest Path Calculation (`_shortest_path_cost`)**: 
        - It uses Dijkstra's algorithm to find the minimum cost to travel between any two rooms.
        - The cost is not just the number of moves but also includes the cost of opening closed doors. A move through an open door costs 1, while a move through a closed door costs 2 (1 for 'open' + 1 for 'move').

    4.  **Goal Cost Estimation (`_cost_to_achieve`)**: 
        - For each misplaced package, it calculates the full sequence of actions required to deliver it, considering three scenarios:
            a) The robot is already holding the package.
            b) The robot is holding a different package (must deliver it first).
            c) The robot's arm is empty.
        - This provides a realistic, admissible estimate for achieving a single goal.

    This state-based approach is more powerful than simple method-scoring heuristics as it
    provides a much more accurate estimate of the distance to the goal, leading to more
    informed search and better performance, especially on larger problem instances.
    """
    def __init__(self, robot_gemini25pro_heuristic_8="robot_gemini25pro_heuristic_8"):
        super().__init__(name=robot_gemini25pro_heuristic_8)
        self.model = None
        self.adj = {}
        self.door_map = {}
        self.goals = {}
        self.packages = set()
        self.rooms = set()
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by preprocessing the static information from the model.
        """
        start_time = time.time()
        self.model = model
        self.adj = {}
        self.door_map = {}
        self.goals = {}
        self.packages = set()
        self.rooms = set()

        for fact in self.model.facts:
            parts = fact.name.strip('()').split()
            predicate = parts[0]
            if predicate == 'door':
                r1, r2, d = parts[1], parts[2], parts[3]
                if r1 not in self.adj: self.adj[r1] = []
                self.adj[r1].append(r2)
                self.door_map[(r1, r2)] = d
                self.rooms.add(r1)
                self.rooms.add(r2)
            elif predicate == 'goal_in':
                pkg, room = parts[1], parts[2]
                self.goals[pkg] = room
                self.packages.add(pkg)
        
        self.preprocessing_time = time.time() - start_time

        initial_h_value = self.__call__(None, initial_node)
        return super().initialize(model, initial_h_value)

    def _parse_state(self, state_facts):
        """Parses a set of fact strings into a structured state dictionary."""
        state_info = {
            'robot_loc': None,
            'held_pkg': None,
            'pkg_locs': {},
            'closed_doors': set()
        }
        for fact_str in state_facts:
            parts = fact_str.strip('()').split()
            predicate = parts[0]
            if predicate == 'rloc':
                state_info['robot_loc'] = parts[1]
            elif predicate == 'holding':
                state_info['held_pkg'] = parts[1]
            elif predicate == 'in':
                # package, room
                state_info['pkg_locs'][parts[1]] = parts[2]
            elif predicate == 'closed':
                state_info['closed_doors'].add(parts[1])
        return state_info

    def _shortest_path_cost(self, start, end, closed_doors):
        """Calculates the shortest path cost between two rooms using Dijkstra's algorithm."""
        if start == end:
            return 0

        # Priority queue: (cost, current_room)
        pq = [(0, start)]
        visited_costs = {start: 0}

        while pq:
            cost, u = heapq.heappop(pq)

            if cost > visited_costs[u]:
                continue

            if u == end:
                return cost

            if u not in self.adj:
                continue

            for v in self.adj[u]:
                door = self.door_map.get((u, v))
                edge_cost = 2 if door in closed_doors else 1
                new_cost = cost + edge_cost

                if v not in visited_costs or new_cost < visited_costs[v]:
                    visited_costs[v] = new_cost
                    heapq.heappush(pq, (new_cost, v))
        
        return float('inf') # No path exists

    def _cost_to_achieve(self, goal_pkg, goal_loc, state_info):
        """Estimates the cost to deliver a single package to its goal location."""
        robot_loc = state_info['robot_loc']
        held_pkg = state_info['held_pkg']
        closed_doors = state_info['closed_doors']
        pkg_locs = state_info['pkg_locs']
        path_cost = self._shortest_path_cost

        # Case 1: The robot is already holding the target package.
        if held_pkg == goal_pkg:
            # Cost is to move to the goal and putdown.
            return path_cost(robot_loc, goal_loc, closed_doors) + 1
        
        # Case 2: The robot is holding a different package.
        elif held_pkg is not None:
            other_pkg_goal = self.goals[held_pkg]
            p_loc = pkg_locs[goal_pkg]
            
            # Cost to deliver the other package first.
            cost_deliver_other = path_cost(robot_loc, other_pkg_goal, closed_doors) + 1
            # Cost to then go get the target package.
            cost_get_target = path_cost(other_pkg_goal, p_loc, closed_doors) + 1
            # Cost to finally deliver the target package.
            cost_deliver_target = path_cost(p_loc, goal_loc, closed_doors) + 1
            return cost_deliver_other + cost_get_target + cost_deliver_target

        # Case 3: The robot's arm is empty.
        else:
            p_loc = pkg_locs.get(goal_pkg)
            if p_loc is None: # Should not happen if logic is correct
                return float('inf')
            
            # Cost to go to the package and pick it up.
            cost_get = path_cost(robot_loc, p_loc, closed_doors) + 1
            # Cost to deliver the package to its goal.
            cost_deliver = path_cost(p_loc, goal_loc, closed_doors) + 1
            return cost_get + cost_deliver

    def __call__(self, parent_node, node):
        """
        Calculates the h_max heuristic value for the given node's state.
        """
        if not self.model or not hasattr(node, 'state'):
            return 0

        # Assumption: node.state is a bitmask of facts.
        state_facts = {self.model.facts[i].name
                       for i in range(len(self.model.facts))
                       if (node.state >> i) & 1}

        state_info = self._parse_state(state_facts)

        if not state_info['robot_loc'] or not self.goals:
            return 0

        unachieved_goals = []
        for pkg, goal_loc in self.goals.items():
            current_loc = state_info['pkg_locs'].get(pkg)
            # A package's location is its goal if it's there, or if another package is held 
            # but this one is already at its goal (it can't be moved).
            if current_loc is not None and current_loc != goal_loc:
                 unachieved_goals.append((pkg, goal_loc))

        if not unachieved_goals:
            return 0
        
        h_value = 0
        for pkg, loc in unachieved_goals:
            cost = self._cost_to_achieve(pkg, loc, state_info)
            if cost > h_value:
                h_value = cost

        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic8(name={self.name})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic (h_max) info:\n"
            f"\tName: {self.name}\n"
            f"\tGoals tracked: {len(self.goals)}\n"
            f"\tRooms in map: {len(self.rooms)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
