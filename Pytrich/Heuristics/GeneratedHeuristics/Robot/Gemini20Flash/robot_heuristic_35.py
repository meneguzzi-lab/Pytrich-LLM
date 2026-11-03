import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic35(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_35=1, name="robot_gemini20flash_heuristic_35"):
        super().__init__(name=name)
        self.name = name
        self.robot_gemini20flash_heuristic_35 = robot_gemini20flash_heuristic_35
        self.method_scores = {}
        self.fact_cost = {}

    def initialize(self, model, initial_node):
        self.model = model
        self._precompute_method_info()
        self._precompute_fact_info()
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_fact_info(self):
        """Assigns a cost to each fact, which can be used as a proxy for goal distance."""
        for fact in self.model.facts:
            fact_name = fact.name.lower()
            if 'armempty' in fact_name:
                self.fact_cost[fact.name] = 0.1 # Low cost, as it's a starting state
            elif 'rloc' in fact_name:
                self.fact_cost[fact.name] = 0.2 # Robot location is important but not high cost
            elif 'in' in fact_name:
                self.fact_cost[fact.name] = 0.5  # Packages in rooms have a moderate cost
            elif 'holding' in fact_name:
                self.fact_cost[fact.name] = 0.7 # Holding a package should have high cost, want to put it down
            elif 'closed' in fact_name:
                self.fact_cost[fact.name] = 0.3 # Cost to open a door.
            elif 'door' in fact_name: 
                self.fact_cost[fact.name] = 0 # ignore door
            elif 'goal_in' in fact_name:
                self.fact_cost[fact.name] = 1  # Goal predicates are most important
            else:
                self.fact_cost[fact.name] = 0 # ignore any other predicate


    def _precompute_method_info(self):
        """Analyzes the domain's methods to estimate their cost.
        Aims to provide a more informed heuristic for method selection.
        """
        if not self.model: return

        methods = self.model.decompositions # Changed to handle 'decompositions' instead of 'methods'
        for method in methods:
            method_name = method.name.lower()
            subtask_count = 0
            pickup_count = 0
            putdown_count = 0
            move_count = 0
            open_count = 0
            release_count = 0
            achieve_count = 0

            if hasattr(method, 'task_network'):
                subtasks = method.task_network
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = getattr(subtask, 'name', str(subtask)).lower() # handles operators and tasks

                    if 'pickup' in subtask_name: pickup_count += 1
                    elif 'putdown' in subtask_name: putdown_count += 1
                    elif 'move' in subtask_name: move_count += 1
                    elif 'open' in subtask_name: open_count += 1
                    elif 'release' in subtask_name: release_count += 1
                    elif 'achieve-goals' in subtask_name: achieve_count += 1

            self.method_scores[method.name] = {
                'subtask_count': subtask_count,
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'move_count': move_count,
                'open_count': open_count,
                'release_count':release_count,
                'achieve_count':achieve_count,
                'is_goal_method': 'achieve-goals' in method_name
            }


    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node.
        Considers the state of the world and the chosen decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, likely initial state
        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """Scores a method based on its characteristics.
        Lower scores are better (cost-based heuristic).
        """
        score = 0
        method_name = method.name
        method_data = self.method_scores.get(method_name, {})

        # Adjust score based on the types and number of subtasks
        score += method_data.get('pickup_count', 0) * 0.6  # High cost for pickups
        score += method_data.get('putdown_count', 0) * 0.5 # High cost for putdowns
        score += method_data.get('move_count', 0) * 0.4  # Moderate cost for moves
        score += method_data.get('open_count', 0) * 0.3 # Moderate cost for opening doors
        score += method_data.get('release_count', 0) * 0.2 # small cost for release
        score += method_data.get('achieve_count', 0) * 0.1 # very small cost for achieve

        #if method_data.get('is_goal_method'):
        #    score -= 0.5  # Encourage goal-achieving methods

        # Add state-based heuristic components
        state = node.state # get current state

        # Extract goal facts directly from model, rather than re-computing each time.
        goal_facts = [fact.name for fact in self.model.facts if 'goal_in' in fact.name.lower()]

        # State based fact heuristic calculations
        for goal in goal_facts:
            if goal not in self.model.state_explicit_repr(state):
              score += self.fact_cost.get(goal, 0.1) # Get cost if available

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(robot_gemini20flash_heuristic_35={self.robot_gemini20flash_heuristic_35})"

    def __str__(self):
        return f"{self.name}(robot_gemini20flash_heuristic_35={self.robot_gemini20flash_heuristic_35})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 35 Info:\n"
            f"\tName: {self.name}\n"
            f"\trobot_gemini20flash_heuristic_35: {self.robot_gemini20flash_heuristic_35}\n"
            f"\tMethod Scores: {len(self.method_scores)}\n"
            f"\tFact Costs: {len(self.fact_cost)}\n"
        )
