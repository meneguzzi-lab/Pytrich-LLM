import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini25ProHeuristic4(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Depot domain.

    This heuristic evaluates the cost of a decomposition method based on the complexity
    of the subtasks it generates. The main drivers of cost in the Depot domain are:
    1.  **Transportation:** Moving trucks between locations ('Drive' action). This is the
        most expensive operation as it changes the global state significantly.
    2.  **Manipulation:** Using hoists to lift, drop, load, and unload crates. These
        actions occur within a single location but are necessary for any progress.
    3.  **Clearing Surfaces:** The 'do_clear' task can be simple (if the surface is
        already clear) or very complex (if it requires moving other crates out of the way,
        potentially involving trucks).

    The heuristic pre-computes a score for each method in the domain during initialization.
    This score is a weighted sum of the penalties associated with the types of subtasks
    in the method's decomposition. Methods that are simple "no-ops" (i.e., the
    precondition they achieve is already true) are assigned a score of 0.
    """
    def __init__(self, drive_penalty=0.5, manipulation_penalty=0.1, clear_penalty=0.2, depot_heuristic_4=None, name="depot_gemini25pro_heuristic_4"):
        super().__init__(name=name)
        self.drive_penalty = drive_penalty
        self.manipulation_penalty = manipulation_penalty
        self.clear_penalty = clear_penalty
        # This parameter is for compatibility as requested, but not used in the logic.
        self.depot_gemini25pro_heuristic_4 = depot_heuristic_4

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic, pre-computing scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a heuristic
        score to each based on its subtasks. This is done once at the beginning
        of the search for efficiency.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = 0.0

            # Base case methods that do nothing because the state is already achieved
            # have zero cost.
            # m0_do_put_on: crate is already on the target surface.
            # m4_do_clear: surface is already clear.
            # m6_do_get_truck: truck is already at the target place.
            if method.name in ["m0_do_put_on", "m4_do_clear", "m6_do_get_truck"]:
                 self.method_scores[method.name] = 0.0
                 continue

            if not hasattr(method, 'task_network'):
                self.method_scores[method.name] = score
                continue

            # Assign costs based on the types of subtasks (primitive actions or abstract tasks)
            for subtask in method.task_network:
                subtask_name_lower = subtask.name.lower()

                # Penalize driving, the most expensive action
                if "drive" in subtask_name_lower:
                    score += self.drive_penalty
                # Penalize tasks that might lead to driving
                elif "do_get_truck" in subtask_name_lower:
                    # This is a smaller penalty because the truck might already be there (m6)
                    score += self.drive_penalty * 0.5
                elif "do_load_truck" in subtask_name_lower or "do_unload_truck" in subtask_name_lower:
                    # These complex tasks involve both manipulation and truck usage
                    score += self.drive_penalty * 0.5 + self.manipulation_penalty

                # Penalize manipulation actions (Lift, Drop, Load, Unload)
                elif any(op in subtask_name_lower for op in ["lift", "drop", "load", "unload"]):
                    score += self.manipulation_penalty

                # Penalize clearing tasks, as they can be complex
                elif "do_clear" in subtask_name_lower:
                    score += self.clear_penalty

            # m5_do_clear is particularly complex, involving getting a truck just to
            # temporarily move a crate. Give it an additional penalty.
            if method.name == "m5_do_clear":
                 score += self.clear_penalty

            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given search node.
        The value is retrieved from the pre-computed scores.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        # Retrieve the pre-computed score for the method applied in this node.
        # Default to 0 if the method is somehow not in our map.
        return self.method_scores.get(node.method.name, 0.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        It calculates the heuristic value for the current node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return (f"{self.__class__.__name__}(drive={self.drive_penalty}, "
                f"manipulation={self.manipulation_penalty}, clear={self.clear_penalty})")

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tDrive Penalty: {self.drive_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
