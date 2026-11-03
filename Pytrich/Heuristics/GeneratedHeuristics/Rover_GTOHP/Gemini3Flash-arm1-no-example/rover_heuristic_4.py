from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class RoverGTOHPGemini3FlashHeuristic4(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    Bottleneck Analysis:
    The primary bottleneck in the Rover domain is the sequence of navigation and store management. 
    A rover must navigate to a waypoint, ensure its store is empty, take a sample, navigate again 
    (possibly) to a waypoint visible from a lander, and then communicate the data. 
    Each goal (communicated data) represents a serialized chain of dependencies.

    Admissible Lower Bounds:
    1. Task-Network Complexity (h_tasks): Sum of the minimum number of primitive actions required 
        to fully decompose every task currently in the task network. This is admissible as these 
        actions are unavoidable to clear the network.
    2. Unfulfilled Goal Cost (h_goals): Each uncommunicated data goal requires at least a 
        communication action (1). If the rover does not yet have the sample in its store, 
        it additionally requires at least one sampling action (1). This provides a lower 
        bound on the physical actions remaining for the state to satisfy the goal predicates.
    """

    def __init__(self, rover_gemini3flash_heuristic_4="rover_gemini3flash_heuristic_4"):
        super().__init__(name=rover_gemini3flash_heuristic_4)
        self.goal_ids = []
        self.task_min_costs = {}
        self.sample_info = {} # mapping goal waypoints to their corresponding 'have_analysis' fact IDs
        self.empty_store_ids = []

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Facts
        soil_analysis_facts = {} # waypoint -> fact_id
        rock_analysis_facts = {} # waypoint -> fact_id
        image_facts = {} # (objective, mode) -> fact_id

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "have_soil_analysis":
                soil_analysis_facts[args[1]] = fact.global_id
            elif pred == "have_rock_analysis":
                rock_analysis_facts[args[1]] = fact.global_id
            elif pred == "have_image":
                image_facts[(args[1], args[2])] = fact.global_id
            elif pred == "empty":
                self.empty_store_ids.append(fact.global_id)

        # 2. Map Goals and associate them with sample requirements
        goal_names = model.state_explicit_repr(model.goals)
        for fact_name in goal_names:
            pred, args = self._parse_fact_name(fact_name)
            # Find the actual Fact object to get global_id
            target_id = -1
            for f in model.facts:
                if f.name == fact_name:
                    target_id = f.global_id
                    break
            
            if target_id != -1:
                self.goal_ids.append(target_id)
                # Associate goal with the fact that represents having the sample collected
                if pred == "communicated_soil_data":
                    if args[0] in soil_analysis_facts:
                        self.sample_info[target_id] = soil_analysis_facts[args[0]]
                elif pred == "communicated_rock_data":
                    if args[0] in rock_analysis_facts:
                        self.sample_info[target_id] = rock_analysis_facts[args[0]]
                elif pred == "communicated_image_data":
                    if (args[0], args[1]) in image_facts:
                        self.sample_info[target_id] = image_facts[(args[0], args[1])]

        # 3. Precompute Task Costs (Minimum primitive actions required to decompose)
        # Primitive actions cost 1. Abstract tasks cost sum of subtasks.
        for op in model.operators:
            self.task_min_costs[op.global_id] = 1 if "nop" not in op.name else 0

        # Iterative estimation for Abstract Tasks (simple lower bound)
        for _ in range(5):
            for abt in model.abstract_tasks:
                best_decomp_cost = float('inf')
                for decomp in abt.decompositions:
                    d_cost = 0
                    for subtask in decomp.task_network:
                        if subtask.global_id in self.task_min_costs:
                            d_cost += self.task_min_costs[subtask.global_id]
                        else:
                            d_cost += 1 # Default assumption for uncomputed tasks
                    best_decomp_cost = min(best_decomp_cost, d_cost)
                self.task_min_costs[abt.global_id] = best_decomp_cost

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        # Term 1: Task Network Cost
        h_tasks = 0
        for task in task_network:
            h_tasks += self.task_min_costs.get(task.global_id, 1)

        # Term 2: Goal Progress Cost
        h_goals = 0
        for g_id in self.goal_ids:
            # If goal not reached
            if not ((state >> g_id) & 1):
                h_goals += 1 # Communication required
                # If sample not even in possession, need to sample too
                sample_possession_id = self.sample_info.get(g_id)
                if sample_possession_id is not None:
                    if not ((state >> sample_possession_id) & 1):
                        h_goals += 1

        # Term 3: Resource Constraint Penalty (Tie-breaker)
        # If stores are full and we need to sample (indicated by tasks), add a small penalty
        penalty = 0
        if any(not ((state >> s_id) & 1) for s_id in self.empty_store_ids):
            penalty = 0.5

        return max(h_tasks, h_goals) + penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
