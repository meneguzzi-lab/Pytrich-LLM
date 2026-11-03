import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for Blocksworld domain
class BlocksworldGemini20FlashHeuristic19(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_19=1, name="BlocksworldGemini20FlashHeuristic19"):
        super().__init__(name=name)
        self.name = name
        self.move_penalty = 1  # Cost for moving a block
        self.wrong_location_penalty = 2  # Cost for a block being in the wrong location
        self.hand_empty_bonus = -1 # Small incentive to make sure hand is empty to finish
        self.method_scores = {}
        self.preprocessing_time = 0


    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)  # Evaluate the initial state

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes information about methods for faster scoring.
        """

        if not self.model:
            return

        methods = []

        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count subtasks for each method type
            put_on_count = 0
            unstack_count = 0
            stack_count = 0
            pick_up_count = 0
            put_down_count = 0
            subtask_count = 0

            # Get subtasks from method
            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = ''
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)


                    subtask_lower = subtask_name.lower()

                    if 'put-on' in subtask_lower or 'do_put_on' in subtask_lower:
                        put_on_count += 1
                    if 'unstack' in subtask_lower:
                        unstack_count += 1
                    if 'stack' in subtask_lower:
                        stack_count += 1
                    if 'pick-up' in subtask_lower:
                        pick_up_count += 1
                    if 'put-down' in subtask_lower:
                        put_down_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'unstack_count': unstack_count,
                'stack_count': stack_count,
                'pick_up_count': pick_up_count,
                'put_down_count': put_down_count,
                'subtask_count': subtask_count,
                'is_goal_method': self._is_goal_method(method_name),
                'complexity_score': put_on_count + unstack_count + stack_count + pick_up_count + put_down_count
            }


    def _is_goal_method(self, method_name):
        """Checks if the method helps achieve the goal (e.g., putting a block in place)."""
        return 'put_on' in method_name.lower() or 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node based on the current state and goal.
        """
        h_value = 0
        state = node.state # Current state bitwise
        model = self.model
        goal_literals = model.goals  # set of goal literals (bitwise IDs).

        # Get explicit state representation (names of the facts).
        current_state = model.state_explicit_repr(state)

        # Count blocks in wrong locations.
        wrong_location_count = 0

        for goal_id in goal_literals:
            goal = model.get_component(goal_id)
            goal_name = goal.name # Predicate (on b1 b4) etc

            if goal_name not in current_state:
                wrong_location_count += 1
        
        # Incentivize having handempty
        if "handempty" in current_state:
            h_value += self.hand_empty_bonus

        h_value += wrong_location_count * self.wrong_location_penalty  # Penalize misplaced blocks.

        # Adjust heuristic based on the decomposition method used (if any).
        if hasattr(node, 'method') and node.method:
            h_value += self._score_method(node.method, node)

        return max(h_value, 0)

    def _score_method(self, method, node):
        """Scores a method based on its potential to achieve the goal. Rewards goal-achieving methods.
        """
        score = 0  # Initialize score
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Reward goal achieving methods. These get a negative score i.e. are preferred
            if info['is_goal_method']:
                score -= 0.5  # Prefer methods directly related to achieving the goal.

            # Penalize complex methods (more subtasks).
            score += info['subtask_count'] * 0.1  # Slightly penalize methods with more subtasks.

        # Add small cost for actions related to manipulating blocks.
        if 'unstack' in method_lower or 'stack' in method_lower or 'pick-up' in method_lower or 'put-down' in method_lower:
            score += 0.2 # Small additional cost for block manipulation actions.

        return score

    def __call__(self, parent_node, node):
        """Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(move_penalty={self.move_penalty}, wrong_location_penalty={self.wrong_location_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(move_penalty={self.move_penalty}, wrong_location_penalty={self.wrong_location_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 19 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tWrong location penalty: {self.wrong_location_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
