"""
Domain-dependent heuristic for the Blocksworld_GTOHP (HTN) domain.

Bottleneck Identification:
The single serializing constraint is the robot hand (holding only one block) 
and the stack-based structure where moving a block requires clearing all 
blocks above it. 

Admissibility Sketches:
1. TaskNetworkCost: Sum of minimum actions to fulfill each task in the 
   current agenda. do_put_on(x,y) requires clearing x (2 actions per block above), 
   clearing y (2 actions per block above), and moving x to y (2 actions).
2. GoalCost: For every goal on(x,y) not satisfied, we need at least one stack(x,y) 
   and one pick-up/unstack(x). Blocks in hand or hand-empty status are also factored.
3. Both are lower bounds because they count unavoidable physical actions (pick-up, unstack, etc.)
   required to satisfy either the HTN tasks or the goal predicates.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGTOHPGemini3FlashHeuristic18(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_18=None, name="BlocksworldGTOHPGemini3FlashHeuristic18"):
        super().__init__(name=name)
        self.on_id = {}          # x -> y -> global_id
        self.on_ids_by_target = {} # y -> list of (x, global_id)
        self.clear_id = {}       # x -> global_id
        self.ontable_id = {}     # x -> global_id
        self.holding_id = {}     # x -> global_id
        self.handempty_id = None
        self.goal_on = {}        # x -> y (from model.goals)
        self.blocks = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        all_blocks = set()
        
        # Map facts
        for fact in model.facts:
            pred, args = self._parse_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                x, y = args[0], args[1]
                if x not in self.on_id: self.on_id[x] = {}
                self.on_id[x][y] = fact.global_id
                if y not in self.on_ids_by_target: self.on_ids_by_target[y] = []
                self.on_ids_by_target[y].append((x, fact.global_id))
                all_blocks.add(x); all_blocks.add(y)
            elif pred == "clear":
                self.clear_id[args[0]] = fact.global_id
                all_blocks.add(args[0])
            elif pred == "ontable":
                self.ontable_id[args[0]] = fact.global_id
                all_blocks.add(args[0])
            elif pred == "holding":
                self.holding_id[args[0]] = fact.global_id
                all_blocks.add(args[0])
            elif pred == "handempty":
                self.handempty_id = fact.global_id

        self.blocks = list(all_blocks)
        for b in self.blocks:
            if b not in self.on_ids_by_target: self.on_ids_by_target[b] = []

        # Map goals
        if hasattr(model, 'goals') and model.goals:
            goal_names = model.state_explicit_repr(model.goals)
            for gn in goal_names:
                pred, args = self._parse_name(gn)
                if pred == "on":
                    self.goal_on[args[0]] = args[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _parse_name(self, name):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b_idx = name.find("[")
            if b_idx > 0 and name.endswith("]"):
                pred = name[:b_idx]
                args = name[b_idx+1:-1].split(",")
                return pred, args
        if name.startswith("(") and name.endswith(")"):
            name = name[1:-1]
        parts = name.split()
        return (parts[0], parts[1:]) if parts else (None, [])

    def _check(self, state, fid):
        return (state >> fid) & 1 if fid is not None else False

    def _count_above(self, block, state):
        count = 0
        curr = block
        while True:
            found_next = False
            for y, fid in self.on_ids_by_target.get(curr, []):
                if (state >> fid) & 1:
                    count += 1
                    curr = y
                    found_next = True
                    break
            if not found_next: break
        return count

    def _compute(self, state, task_network):
        # 1. Task Network Cost
        tn_cost = 0
        for task in task_network:
            pred, args = self._parse_name(task.name)
            if not pred: continue
            if pred == "do_put_on":
                x, y = args[0], args[1]
                if not self._check(state, self.on_id.get(x, {}).get(y)):
                    tn_cost += 2 + 2*self._count_above(x, state) + 2*self._count_above(y, state)
            elif pred == "do_clear":
                tn_cost += 2 * self._count_above(args[0], state)
            elif pred == "do_move":
                tn_cost += 2
            elif pred == "do_on_table":
                if not self._check(state, self.ontable_id.get(args[0])):
                    tn_cost += 2 + 2 * self._count_above(args[0], state)
            elif pred in ["pick-up", "unstack", "stack", "put-down"]:
                tn_cost += 1

        # 2. Goal Cost (admissible lower bound based on unsatisfied goal predicates)
        goal_cost = 0
        if not self._check(state, self.handempty_id): goal_cost += 1
        for x, y in self.goal_on.items():
            if not self._check(state, self.on_id.get(x, {}).get(y)):
                goal_cost += 1 # for stack
                if not self._check(state, self.holding_id.get(x)): goal_cost += 1 # for pickup
        
        return max(tn_cost, goal_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
