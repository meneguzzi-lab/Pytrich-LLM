import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask

class DepotGemini25ProHeuristic3(Heuristic):
    def __init__(self, depot_gemini25pro_heuristic_3=None, drive_penalty=3, manipulation_penalty=1, clear_penalty=0.5):
        """
        A domain-dependent heuristic for the Depot domain.
        This heuristic estimates the cost of a decomposition method by statically analyzing
        the actions it will likely lead to. It prioritizes solutions that minimize driving
        and complex crate manipulations.

        Args:
            drive_penalty (int): The cost assigned to a Drive action. Should be > manipulation_penalty.
            manipulation_penalty (int): The cost for actions like Lift, Drop, Load, Unload.
            clear_penalty (float): A small cost for tasks that might require clearing surfaces.
        """
        super().__init__(name="depot_gemini25pro_heuristic_3")
        self.drive_penalty = drive_penalty
        self.manipulation_penalty = manipulation_penalty
        self.clear_penalty = clear_penalty
        self.method_costs = {}
        self.task_costs = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing costs for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_costs()
        self.preprocessing_time = time.time() - start_time
        
        h_value = 0
        return super().initialize(model, h_value)

    def _precompute_costs(self):
        """
        Pre-computes the heuristic cost for each decomposition method and abstract task
        using a fixed-point iteration approach. This provides a more accurate and globally
        consistent estimation of the minimum action counts.
        """
        if not self.model:
            return

        for task in self.model.abstract_tasks:
            self.task_costs[task.name] = float('inf')

        changed = True
        iterations = 0
        max_iterations = len(self.model.abstract_tasks) + 5

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for method in self.model.decompositions:
                current_method_cost = 0
                for subtask in method.task_network:
                    if isinstance(subtask, Operator):
                        action_name = subtask.name.lower().split(' ')[0]
                        if 'drive' in action_name:
                            current_method_cost += self.drive_penalty
                        elif any(op in action_name for op in ['lift', 'drop', 'load', 'unload']):
                            current_method_cost += self.manipulation_penalty
                    elif isinstance(subtask, AbstractTask):
                        task_cost = self.task_costs.get(subtask.name, float('inf'))
                        if 'do_clear' in subtask.name.lower():
                             current_method_cost += self.clear_penalty
                        if task_cost != float('inf'):
                            current_method_cost += task_cost

                self.method_costs[method.name] = current_method_cost
            
            for task in self.model.abstract_tasks:
                min_cost = float('inf')
                for method in task.decompositions:
                    min_cost = min(min_cost, self.method_costs.get(method.name, float('inf')))
                
                if min_cost < self.task_costs.get(task.name, float('inf')):
                    self.task_costs[task.name] = min_cost
                    changed = True

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is based on the pre-computed cost of the method used for decomposition.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        cost = self.method_costs.get(method_name, 1.0)
        return float(cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Computes and returns the heuristic value for the current node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_costs)}\n"
            f"\tTasks analyzed: {len(self.task_costs)}\n"
            f"\tDrive Penalty: {self.drive_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
