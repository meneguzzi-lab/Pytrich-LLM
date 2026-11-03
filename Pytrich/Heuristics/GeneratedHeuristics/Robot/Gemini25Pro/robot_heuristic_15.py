import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import deque

class RobotGemini25ProHeuristic15(Heuristic):
    """
    A domain-dependent heuristic for the HDDL 'robot' domain.

    This heuristic estimates the cost to achieve goals by analyzing the
    methods being applied during the search. It provides an admissible lower-bound
    estimate of the actions required to complete the sub-plan initiated by a given method.

    Design Rationale:
    1.  **Pre-computation of Distances**: The main cost in this domain is movement.
        The heuristic pre-computes the shortest path (in terms of 'move' actions)
        between all pairs of rooms using a Breadth-First Search. This allows for
        quick lookups of travel costs during planning.

    2.  **Goal-Oriented Analysis**: It identifies package goal locations ('goal_in')
        during preprocessing. This information is crucial for estimating the effort
        required to deliver a package.

    3.  **Informed Method Scoring**: The heuristic's strength lies in its ability to
        provide a more accurate cost estimate for the most critical planning decision:
        picking up a package. For the 'achieve-goals-pickup' method, it calculates
        a heuristic value of `1 (pickup) + distance_to_goal + 1 (putdown)`.
        This is significantly more informed than a generic action count and helps the
        planner prioritize delivering packages that are closer to their destination.

    4.  **Admissibility**: The heuristic is designed to be admissible (h(n) <= h*(n)).
        It only considers the minimum necessary actions (pickup, move, putdown) and
        deliberately ignores potential additional costs like 'open' actions. This ensures
        that the estimate never overshoots the true minimum cost, making it suitable for
        optimal search algorithms like A*.

    5.  **Hierarchical Awareness**: The heuristic assigns different values to different
        methods, reflecting their role in the hierarchy. Goal-achieving methods like
        'release-putdown_abstract' are given precise low costs, while recursive
        movement methods are assigned a cost reflecting the single action they perform per step.
    """

    def __init__(self, robot_gemini25pro_heuristic_15):
        """
        Initializes the RobotGemini25ProHeuristic15.
        The parameter `robot_gemini25pro_heuristic_15` is unused but required by the naming convention.
        """
        super().__init__(name="robot_gemini25pro_heuristic_15")
        self.problem_model = None
        self.distances = {}
        self.package_goals = {}
        self.rooms = set()
        self.adj = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-processing the problem model.
        """
        start_time = time.time()
        self.problem_model = model
        self._preprocess_graph()
        self._preprocess_goals()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _preprocess_graph(self):
        """
        Builds a graph of room connectivity from '(door ...)' facts and computes
        all-pairs shortest paths using BFS.
        """
        for fact in self.problem_model.facts:
            parts = fact.name.strip("()").split()
            if parts and parts[0] == 'door':
                _, r1, r2, _ = parts
                self.rooms.add(r1)
                self.rooms.add(r2)
                if r1 not in self.adj: self.adj[r1] = []
                if r2 not in self.adj: self.adj[r2] = []
                self.adj[r1].append(r2)
                self.adj[r2].append(r1)

        for start_node in self.rooms:
            q = deque([(start_node, 0)])
            visited = {start_node: 0}
            self.distances[(start_node, start_node)] = 0
            
            while q:
                curr, dist = q.popleft()
                for neighbor in self.adj.get(curr, []):
                    if neighbor not in visited:
                        visited[neighbor] = dist + 1
                        self.distances[(start_node, neighbor)] = dist + 1
                        q.append((neighbor, dist + 1))

    def _preprocess_goals(self):
        """
        Parses and stores the goal location for each package from '(goal_in ...)' facts.
        """
        for fact in self.problem_model.facts:
            parts = fact.name.strip("()").split()
            if parts and parts[0] == 'goal_in':
                _, package, room = parts
                self.package_goals[package] = room

    def _get_params(self, name_string: str) -> list:
        """
        Helper to extract parameters from a grounded method/task name string.
        Example: "(method-name param1 param2)" -> ["param1", "param2"]
        """
        return name_string.strip("()").split()[1:]

    def _calculate_node_heuristic(self, node: HTNNode) -> int:
        """
        Calculates the heuristic value for a given search node.
        The value is an admissible estimate of the minimum cost to complete the plan
        from this node's decomposition.
        """
        if not node.method:
            return 1 if node.task else 0

        method_name = node.method.name
        
        if 'achieve-goals-pickup' in method_name:
            params = self._get_params(method_name)
            if len(params) == 2:
                loc, obj = params[0], params[1]
                goal_loc = self.package_goals.get(obj)
                if goal_loc:
                    dist = self.distances.get((loc, goal_loc), len(self.rooms))
                    return 1 + dist + 1
            return 2  

        if 'release-putdown_abstract' in method_name:
            return 1

        if 'achieve-goals-move' in method_name or \
           'achieve-goals-open' in method_name or \
           'release-move' in method_name or \
           'release-open' in method_name:
            return 1
        
        if 'finished' in method_name:
            return 0

        decomposed_task_name = node.method.compound_task.name
        if 'pickup_abstract' in decomposed_task_name or \
           'putdown_abstract' in decomposed_task_name or \
           'move_abstract' in decomposed_task_name or \
           'open_abstract' in decomposed_task_name:
            return 1
        
        return 1

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        The main entry point for the heuristic evaluation.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic15(rooms={len(self.rooms)}, goals={len(self.package_goals)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 15 Info:\n"
            f"\tName: {self.name}\n"
            f"\tRooms found: {len(self.rooms)}\n"
            f"\tPackage goals found: {len(self.package_goals)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
        )
