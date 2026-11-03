"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck: Each turn flips a center cell (1 action) plus propagates flips along
4 directions. A single turn can flip many cells, but each turn costs at least 1
action for the center flip, plus 1 action per propagated cell, plus method
precondition checks. The goal is to transform the grid from current to target config.

Admissible lower bounds:
1. Goal mismatch bound: Count cells differing from goal. Each flip action fixes
   exactly one cell's mismatch, so #mismatches is a lower bound on flip actions needed.
2. Task network bound: Each 'turn' task requires at least 1 flip action (the center).
   Each 'play' with pending turns needs at least 1 turn. We count minimum operators
   needed from the task network structure.
3. Combined: max(mismatch_bound, task_network_bound)
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsAnthropicOpus46Heuristic17(Heuristic):
    """
    Heuristic for the Lamps domain that combines:
    - Goal mismatch counting (each mismatched cell needs at least 1 flip)
    - Task network analysis (each turn needs at least 1 flip + method preconditions)
    """

    def __init__(self, lamps_anthropicopus46_heuristic_17=None, name="LampsAnthropicOpus46Heuristic17"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Maps for efficient state checking
        self.on_facts = {}  # (x, y, s) -> global_id
        self.goal_on_specs = []  # list of (x, y, s, global_id) for goal facts
        self.goal_mismatch_fact_ids = []  # list of (goal_fact_global_id, is_positive)
        
        # For task network analysis
        self.turn_task_ids = set()  # global_ids of abstract 'turn' tasks
        self.play_task_ids = set()  # global_ids of abstract 'play' tasks
        self.propagate_task_ids = set()  # global_ids of propagate_* tasks
        self.flip_op_ids = set()  # global_ids of flip operators
        self.method_precondition_op_ids = set()  # global_ids of __method_precondition operators
        self.noop_op_ids = set()  # global_ids of noop operators
        
        # Precomputed goal state info
        self.goal_bits = 0
        self.num_goal_facts = 0
        
        # For each on[x,y,s] fact, store the complementary fact on[x,y,s'] 
        self.complement_fact = {}  # global_id -> global_id of complement
        
        # Decomposition analysis
        self.min_ops_per_turn = 1  # at minimum 1 flip per turn
        self.min_ops_per_propagate = 0  # propagate can be empty

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _preprocess(self):
        """Extract domain structure from grounded facts and operators."""
        
        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts[(x, y, s)] = fact.global_id
        
        # Build complement map: for each on[x,y,Y] find on[x,y,N] and vice versa
        for (x, y, s), gid in self.on_facts.items():
            comp_s = "N" if s == "Y" else "Y"
            comp_key = (x, y, comp_s)
            if comp_key in self.on_facts:
                self.complement_fact[gid] = self.on_facts[comp_key]
        
        # Parse goal facts
        self.goal_bits = self.model.goals
        if self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, s = args[0], args[1], args[2]
                    gid = self.on_facts.get((x, y, s))
                    if gid is not None:
                        self.goal_on_specs.append((x, y, s, gid))
                        self.num_goal_facts += 1
        
        # Classify operators
        for op in self.model.operators:
            name_lower = op.name.lower()
            if "flip" in name_lower and "__method" not in name_lower:
                self.flip_op_ids.add(op.global_id)
            elif "__method_precondition" in name_lower:
                self.method_precondition_op_ids.add(op.global_id)
            elif "noop" in name_lower or "no-op" in name_lower or "no_op" in name_lower:
                self.noop_op_ids.add(op.global_id)
        
        # Classify abstract tasks
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if "turn" in name_lower and "propagate" not in name_lower:
                self.turn_task_ids.add(at.global_id)
            elif "play" in name_lower:
                self.play_task_ids.add(at.global_id)
            elif "propagate" in name_lower:
                self.propagate_task_ids.add(at.global_id)
        
        # Analyze decompositions to find minimum cost per turn
        # A turn decomposes into: flip + propagate_up + propagate_down + propagate_left + propagate_right
        # But method preconditions also add cost
        # Count minimum operators in each decomposition
        self._analyze_decomposition_costs()

    def _analyze_decomposition_costs(self):
        """
        Analyze decompositions to find minimum primitive actions per task type.
        
        For 'turn': the do_turn method has flip + 4 propagations.
        The flip is 1 action. Each propagation can end immediately (0 actions)
        or continue (1+ actions). But method preconditions add cost.
        
        We count the minimum number of operators (including method_precondition)
        in the task network of each decomposition.
        """
        # For each turn decomposition, count minimum operators
        min_turn_cost = float('inf')
        for at_id in self.turn_task_ids:
            at = self.model.get_component(at_id)
            for dec in at.decompositions:
                # Count operators directly in the task network
                op_count = 0
                for task in dec.task_network:
                    if hasattr(task, 'cost'):  # It's an operator
                        op_count += 1
                if op_count > 0:
                    min_turn_cost = min(min_turn_cost, op_count)
        
        if min_turn_cost < float('inf'):
            self.min_ops_per_turn = min_turn_cost
        else:
            self.min_ops_per_turn = 1  # at least the flip

    def _count_goal_mismatches(self, state_bitwise: int) -> int:
        """
        Count the number of goal facts not satisfied in the current state.
        
        Each unsatisfied goal on[x,y,s] means the cell (x,y) has the wrong status.
        Fixing it requires at least 1 flip action on that cell.
        
        Admissibility: each flip action changes exactly one cell's status,
        so #mismatches <= #flip_actions_needed.
        """
        if not self.goal_bits:
            return 0
        
        # Check which goal facts are not satisfied
        # goal_bits has all goal facts set. We check which are missing from state.
        # A goal fact is satisfied if (state >> fact.global_id) & 1 == 1
        mismatches = 0
        for (x, y, s, gid) in self.goal_on_specs:
            if not ((state_bitwise >> gid) & 1):
                mismatches += 1
        
        return mismatches

    def _task_network_lower_bound(self, task_network) -> int:
        """
        Estimate minimum primitive actions from the task network.
        
        For each element in the task network:
        - Primitive operator: costs 1
        - 'turn' abstract task: costs at least min_ops_per_turn (1 flip minimum)
        - 'play' abstract task: costs 0 (can decompose to end_game with 0 subtasks)
        - 'propagate_*' abstract task: costs 0 (can end immediately)
        
        Admissibility: we use the minimum possible decomposition cost for each task.
        """
        if not task_network:
            return 0
        
        total = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                # Abstract task
                gid = task.global_id
                if gid in self.turn_task_ids:
                    # A turn requires at least 1 flip action
                    total += self.min_ops_per_turn
                elif gid in self.play_task_ids:
                    # play can be end_game (0 cost) - but if goals aren't met,
                    # it must do at least one turn
                    # We handle this conservatively: 0
                    pass
                elif gid in self.propagate_task_ids:
                    # propagate can end immediately (0 cost)
                    pass
                else:
                    # Unknown abstract task - conservative
                    pass
            else:
                # Primitive operator - costs 1
                total += 1
        
        return total

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Compute the heuristic value as max of multiple admissible bounds.
        """
        # If no tasks left and goals satisfied, we're done
        if not task_network:
            if self.model.goal_reached(state_bitwise, task_network):
                return 0
            else:
                # No tasks but goals not met - shouldn't happen in valid plans
                return float('inf')
        
        # Term 1: Goal mismatch count
        # Each mismatched cell needs at least 1 flip to fix
        mismatch_count = self._count_goal_mismatches(state_bitwise)
        
        # Term 2: Task network lower bound
        tn_bound = self._task_network_lower_bound(task_network)
        
        # Term 3: Combined reasoning
        # If there are mismatches, we need at least ceil(mismatches / max_flips_per_turn) turns
        # But each turn's propagation can flip many cells, so this is hard to bound tightly.
        # However, each mismatch still needs its own flip action regardless of batching.
        # So mismatch_count is already a valid lower bound on total flip actions.
        
        # The task network bound counts pending operators + minimum per abstract task.
        # The mismatch bound counts minimum flips needed for goal satisfaction.
        # These are somewhat independent views.
        
        h_value = max(mismatch_count, tn_bound)
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsAnthropicOpus46Heuristic17(goals={self.num_goal_facts})"

    def __str__(self):
        return f"LampsAnthropicOpus46Heuristic17 with {self.num_goal_facts} goal facts"

    def __output__(self):
        return (
            f"Lamps Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Facts: {self.num_goal_facts}\n"
            f"\tOn Facts: {len(self.on_facts)}\n"
            f"\tMin Ops Per Turn: {self.min_ops_per_turn}\n"
            f"\tFlip Operators: {len(self.flip_op_ids)}\n"
            f"\tMethod Precondition Ops: {len(self.method_precondition_op_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
