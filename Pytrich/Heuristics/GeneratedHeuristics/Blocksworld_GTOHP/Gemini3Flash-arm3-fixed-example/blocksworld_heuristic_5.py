"""
Domain-dependent heuristic for the Blocksworld_GTOHP domain.

Bottleneck analysis:
The primary bottleneck is the hand-empty constraint and stack clearing. 
Almost all meaningful methods (m1, m2, m4, m5, m7) require a handempty condition. 
Since only one block can be held at a time, actions must be serialized to clear blocks 
and move them to their target positions.

Admissibility sketches:
1. Task Network Cost (Term 1): Each task in the task network requires a minimum number 
   of primitive operators to decompose and execute. For example, 'do_move' always 
   requires 2 operators (pick-up/unstack + stack). Summing these minimum costs 
   is admissible as each task must be resolved.
2. Hand-Empty Constraint (Term 2): If handempty is false, the robot is holding a block. 
   At least one operator (put-down or stack) is required to free the hand and proceed 
   with other tasks that require handempty. This adds 1 to the lower bound.
3. State-Goal Discrepancy (Tie-breaker): A sub-unit penalty for goal facts not currently 
   satisfied helps guide the search toward goals when task network costs are tied.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class BlocksworldGTOHPGemini3FlashHeuristic5(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_5=None, name="BlocksworldGTOHPGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.on_facts = defaultdict(dict)
        self.ontable_id = {}
        self.clear_id = {}
        self.holding_id = {}
        self.handempty_id = None
        self.goal_on = {}
        self.goal_ontable = set()
        self.task_info_cache = {}
        self.initialized = False

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        # 1. Map predicates to global IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_facts[args[0]][args[1]] = fact.global_id
            elif pred == "ontable":
                self.ontable_id[args[0]] = fact.global_id
            elif pred == "clear":
                self.clear_id[args[0]] = fact.global_id
            elif pred == "holding":
                self.holding_id[args[0]] = fact.global_id
            elif pred == "handempty":
                self.handempty_id = fact.global_id

        # 2. Extract goals from model
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            pred, args = self._parse_fact_name(goal_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]
            elif pred == "ontable":
                self.goal_ontable.add(args[0])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        self.initialized = True
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
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _get_task_info(self, task_name):
        if task_name not in self.task_info_cache:
            bracket_idx = task_name.find("[")
            if bracket_idx != -1:
                t_type = task_name[:bracket_idx]
                args = task_name[bracket_idx+1:-1].split(",")
            else:
                t_type = task_name
                args = []
            self.task_info_cache[task_name] = (t_type, args)
        return self.task_info_cache[task_name]

    def _compute(self, state, task_network):
        h = 0
        # Term 1: Task Network Cost
        for task in task_network:
            t_type, args = self._get_task_info(task.name)
            if t_type == "do_put_on":
                x, y = args[0], args[1]
                # cost is 0 if already on(x,y), else 2 + clearing costs
                fid = self.on_facts.get(x, {}).get(y)
                if fid is not None and not ((state >> fid) & 1):
                    h += 2
                    # Cost of do_clear x and y
                    if x in self.clear_id and not ((state >> self.clear_id[x]) & 1): h += 2
                    if y in self.clear_id and not ((state >> self.clear_id[y]) & 1): h += 2
            elif t_type == "do_clear":
                x = args[0]
                if x in self.clear_id and not ((state >> self.clear_id[x]) & 1): h += 2
            elif t_type == "do_on_table":
                x = args[0]
                if x in self.clear_id and not ((state >> self.clear_id[x]) & 1): h += 2
            elif t_type == "do_move":
                h += 2
            elif t_type in ["pick-up", "put-down", "stack", "unstack"]:
                h += 1

        # Term 2: Hand-Empty Constraint
        if self.handempty_id is not None and not ((state >> self.handempty_id) & 1):
            h += 1

        # Term 3: Goal Tie-breaker (Sub-unit)
        penalty = 0
        for x, y in self.goal_on.items():
            fid = self.on_facts.get(x, {}).get(y)
            if fid is not None and not ((state >> fid) & 1):
                penalty += 0.001
        for x in self.goal_ontable:
            fid = self.ontable_id.get(x)
            if fid is not None and not ((state >> fid) & 1):
                penalty += 0.001

        return int(h) + penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
