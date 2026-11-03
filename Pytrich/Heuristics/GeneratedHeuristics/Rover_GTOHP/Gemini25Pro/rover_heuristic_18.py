import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini25ProHeuristic18(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Rover domain.

    This heuristic is primarily a goal-count heuristic, which is admissible for this
    domain. It estimates the remaining cost as the number of top-level goals
    (communicated data) that have not yet been achieved.

    Design Rationale:
    1.  Admissibility: The primary heuristic component is the count of unachieved
        goal predicates. In the Rover domain, each goal predicate like
        `(communicated_soil_data waypointX)` is achieved by a unique primitive
        communication action. Since no single action can satisfy multiple goals,
        the number of remaining goals is a lower bound on the number of actions
        needed to complete the plan, making the heuristic admissible.

    2.  Informed Tie-Breaking: To guide the search more effectively when multiple
        nodes have the same number of remaining goals, a small, static cost is
        added based on the method being decomposed. This cost is precomputed and
        estimates the minimum number of primitive actions a method will likely generate.
        For example, `get_image_data` is more complex and thus has a higher
        static cost than `get_soil_data`. This encourages the planner to explore
        simpler, more direct methods first.

    3.  Efficiency: All expensive computations are done once during the
        `initialize` phase. This includes identifying goal fact IDs and pre-calculating
        method costs. The per-node heuristic calculation in `__call__` is therefore
        very fast, involving only bitwise state checks and a dictionary lookup.

    4.  Domain Insights:
        - The key challenge is sequencing many actions (navigation, calibration,
          sampling, communication) to achieve each goal.
        - The final goal-achieving actions are the communication actions.
        - Tasks have varying complexity: imaging requires an extra calibration
          step, making it inherently more costly than soil or rock sampling.
          This is reflected in the method tie-breaking costs.
    """
    def __init__(self, rover_gemini25pro_heuristic_18={"action_cost_weight": 1.0, "method_complexity_weight": 0.1}):
        """
        Initializes the heuristic.

        :param rover_gemini25pro_heuristic_18: A configuration dictionary.
            - "action_cost_weight": Weight for the goal-count component. Should be >= 1 for admissibility.
            - "method_complexity_weight": Small weight for the tie-breaking component.
        """
        super().__init__(name="rover_gemini25pro_heuristic_18")
        self.config = rover_gemini25pro_heuristic_18
        self.action_cost_weight = self.config.get("action_cost_weight", 1.0)
        self.method_complexity_weight = self.config.get("method_complexity_weight", 0.1)

        # Precomputed data stores
        self.goal_fact_ids = set()
        self.method_costs = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Precomputes necessary information from the domain and problem model.
        This is called once before the search begins.
        """
        start_time = time.time()
        self.model = model

        # 1. Identify all goal fact IDs from the problem's goal specification.
        # The model.goals is a bitmask representing the set of goal facts.
        goal_bits = self.model.goals
        i = 0
        while goal_bits > 0:
            if (goal_bits & 1) == 1:
                self.goal_fact_ids.add(i)
            goal_bits >>= 1
            i += 1
        
        # 2. Pre-compute static costs for each method for efficient tie-breaking.
        self._precompute_method_costs()

        self.preprocessing_time = time.time() - start_time
        
        # Calculate and set the initial heuristic value for the root node.
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_costs(self):
        """
        Assigns a static, estimated cost to each method based on an analysis of
        the minimum number of primitive actions it is expected to generate. This cost
        is used as a small tie-breaker. The costs are underestimates to preserve admissibility.
        """
        # Base costs for simple methods (directly correspond to actions or nothing)
        costs = {
            'm0_do_navigate1': 0, 'm2_do_navigate2': 0, 'm5_empty_store': 0, # Nop methods
            'm3_do_navigate2': 1, 'm6_empty_store': 1, 'm9_send_soil_data': 1,
            'm12_send_rock_data': 1
        }

        # Estimated minimum costs for tasks based on their cheapest decomposition
        task_min_costs = {
            'do_navigate2': 0, # Can be achieved by m2_do_navigate2 with 0 actions
            'do_navigate1': 0, # Can be achieved by m0_do_navigate1 with 0 actions
            'empty_store': 0,  # Can be achieved by m5_empty_store with 0 actions
            'send_soil_data': costs['m9_send_soil_data'], # min cost is 1 action
            'send_rock_data': costs['m12_send_rock_data'], # min cost is 1 action
        }

        # Costs for more complex methods, built from task costs
        costs['m1_do_navigate1'] = task_min_costs['do_navigate2'] # visit/unvisit are meta-actions
        costs['m4_do_navigate2'] = 1 + task_min_costs['do_navigate2'] # navigate + subtask
        costs['m8_send_soil_data'] = task_min_costs['do_navigate1'] + 1 # navigate subtask + communicate action
        costs['m11_send_rock_data'] = task_min_costs['do_navigate1'] + 1 # navigate subtask + communicate action
        costs['m14_send_image_data'] = task_min_costs['do_navigate1'] + 1
        costs['m15_do_calibrate'] = task_min_costs['do_navigate1'] + 1
        task_min_costs['do_calibrate'] = costs['m15_do_calibrate']
        task_min_costs['send_image_data'] = costs['m14_send_image_data']

        # Costs for top-level tasks
        costs['m7_get_soil_data'] = task_min_costs['do_navigate1'] + task_min_costs['empty_store'] + 1 + task_min_costs['send_soil_data']
        costs['m10_get_rock_data'] = task_min_costs['do_navigate1'] + task_min_costs['empty_store'] + 1 + task_min_costs['send_rock_data']
        costs['m13_get_image_data'] = task_min_costs['do_calibrate'] + task_min_costs['do_navigate1'] + 1 + task_min_costs['send_image_data']

        self.method_costs = costs

    def _count_unachieved_goals(self, state_bits: int) -> int:
        """
        Counts how many goal predicates are not true in the given bitwise state.
        """
        unachieved_count = 0
        for goal_id in self.goal_fact_ids:
            # Check if the bit at the goal_id position is 0.
            if not (state_bits & (1 << goal_id)):
                unachieved_count += 1
        return unachieved_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        h(n) = w1 * (number of unachieved goals) + w2 * (tie-breaking cost for current method)
        """
        # 1. Main heuristic component: Count of remaining goals.
        num_unachieved_goals = self._count_unachieved_goals(node.state)
        h_value = num_unachieved_goals * self.action_cost_weight

        # 2. Tie-breaker component: Add a small, precomputed cost based on the
        # complexity of the method being applied at this node. This helps the
        # planner to prefer simpler decompositions when the goal count is equal.
        if node.method is not None:
            method_name = node.method.name
            tie_breaker_cost = self.method_costs.get(method_name, 1) # Default cost of 1 if not found
            h_value += tie_breaker_cost * self.method_complexity_weight

        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"RoverGemini25ProHeuristic18(action_weight={self.action_cost_weight}, method_weight={self.method_complexity_weight})"

    def __output__(self):
        return (
            f"Rover Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoals Tracked: {len(self.goal_fact_ids)}\n"
            f"\tMethod Costs Analyzed: {len(self.method_costs)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
