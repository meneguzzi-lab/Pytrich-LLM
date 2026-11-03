import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for blocksworld domain
class BlocksworldGemini20FlashHeuristic4(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_4="blocksworld_gemini20flash_heuristic_4", misplaced_blocks_weight=1, ontable_blocks_weight=2):
        super().__init__(name=blocksworld_gemini20flash_heuristic_4)
        self.misplaced_blocks_weight = misplaced_blocks_weight
        self.ontable_blocks_weight = ontable_blocks_weight
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """

        if not self.model:
            return

        # Try different possible attributes for methods
        methods = []

        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    task_methods = task.decompositions
                    methods.extend(task_methods)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks
            put_on_count = 0
            subtask_count = 0

            # Try different possible attributes for subtasks
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
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'do_put_on' in subtask_lower:
                        put_on_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'subtask_count': subtask_count,
                'complexity_score': put_on_count + subtask_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on the number of misplaced blocks
        and blocks on the table that should be somewhere else.
        """
        if not hasattr(node, 'state') or node.state is None or self.model is None:
            return 0

        state = node.state
        goal_state = self.model.goals
        # Convert sets to lists for indexing (if necessary)
        if isinstance(state, set):
            state_list = list(state)
        else:
            state_list = [i for i in range(state.bit_length()) if (state & (1 << i))]#state.bit_length()#list(state)

        if isinstance(goal_state, set):
            goal_list = list(goal_state)
        else:
            goal_list = [i for i in range(goal_state.bit_length()) if (goal_state & (1 << i))]#goal_state.bit_length()#list(goal_state)


        misplaced_blocks = 0
        ontable_blocks = 0
        facts = self.model.facts

        # Determine facts present in goal
        goal_facts = [facts[idx].name for idx in goal_list if idx < len(facts)]
        # Determine facts present in current state
        current_state_facts = [facts[idx].name for idx in state_list if idx < len(facts)]
        # Compute misplaced blocks
        misplaced_blocks = len([fact for fact in current_state_facts if fact not in goal_facts and ('on ' in fact or 'ontable' in fact)])
        # Compute ontable blocks
        ontable_blocks = len([fact for fact in current_state_facts if 'ontable' in fact and fact not in goal_facts])

        # Use weights to determine the heuristic value
        h_value = (self.misplaced_blocks_weight * misplaced_blocks) + (self.ontable_blocks_weight * ontable_blocks)
        return h_value

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic4(misplaced_blocks_weight={self.misplaced_blocks_weight}, ontable_blocks_weight={self.ontable_blocks_weight})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic4(misplaced_blocks_weight={self.misplaced_blocks_weight}, ontable_blocks_weight={self.ontable_blocks_weight})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 4 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced blocks weight: {self.misplaced_blocks_weight}\n"
            f"\tOntable blocks weight: {self.ontable_blocks_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
