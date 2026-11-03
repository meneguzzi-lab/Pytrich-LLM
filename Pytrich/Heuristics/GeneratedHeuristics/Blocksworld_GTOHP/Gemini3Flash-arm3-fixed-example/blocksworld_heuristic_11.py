import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Blocksworld_GTOHP HDDL domain.

1. Bottleneck Analysis:
   The vertical dependency of block stacks is the primary bottleneck. Clearing a block x
   requires moving all blocks currently on top of it. In GTOHP, the methods for do_put_on(x, y)
   are particularly rigid, often requiring x and y to be cleared and y to be placed on the table
   before stacking x onto y.

2. Admissible Lower Bounds:
   - Task Network Decomposition Cost (Term 1): Each abstract task (do_put_on, do_clear, etc.)
     is assigned a cost based on the minimum number of actions needed to satisfy its requirements
     in the current state. Clearing an obstruction costs 2 actions (unstack + put-down/stack).
   - Goal Misplacement Cost (Term 2): Each goal on(x, y) not currently in the state requires
     at least one move (2 actions) and potentially clearing obstructions.
   - Max-Combination: The heuristic returns the maximum of the task network cost and goal cost,
     ensuring admissibility while capturing both planned work and outstanding goals.

3. Preprocessing:
   Maps for all predicates (on, clear, holding, ontable, handempty) are built in initialize().
   Task parsing is cached to handle large problems like p30 (1000 blocks) efficiently.
"""

class BlocksworldGTOHPGemini3FlashHeuristic11(Heuristic):
    def __init__(self, blocksworld_gemini3flash_heuristic_11=None, name="BlocksworldGTOHPGemini3FlashHeuristic11"):
        super().__init__(name=name)
        self.block_names = []
        self.block_to_idx = {}
        self.num_blocks = 0
        
        self.on_ids_by_top_bottom = []
        self.on_ids_by_bottom = []
        self.clear_ids = []
        self.ontable_ids = []
        self.holding_ids = []
        self.handempty_id = -1
        
        self.goal_on_ids = {}
        self.task_cache = {}
        self._above_cache = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _preprocess(self):
        block_set = set()
        for f in self.model.facts:
            pred, args = self._parse_name(f.name)
            for a in args: block_set.add(a)
        
        self.block_names = sorted(list(block_set))
        self.block_to_idx = {name: i for i, name in enumerate(self.block_names)}
        self.num_blocks = len(self.block_names)
        
        self.on_ids_by_top_bottom = [[-1 for _ in range(self.num_blocks)] for _ in range(self.num_blocks)]
        self.on_ids_by_bottom = [[] for _ in range(self.num_blocks)]
        self.clear_ids = [-1] * self.num_blocks
        self.ontable_ids = [-1] * self.num_blocks
        self.holding_ids = [-1] * self.num_blocks
        
        for f in self.model.facts:
            pred, args = self._parse_name(f.name)
            if not pred: continue
            if pred == "on" and len(args) == 2:
                u, v = self.block_to_idx.get(args[0]), self.block_to_idx.get(args[1])
                if u is not None and v is not None:
                    self.on_ids_by_top_bottom[u][v] = f.global_id
                    self.on_ids_by_bottom[v].append((u, f.global_id))
            elif pred == "clear" and len(args) == 1:
                u = self.block_to_idx.get(args[0])
                if u is not None: self.clear_ids[u] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                u = self.block_to_idx.get(args[0])
                if u is not None: self.ontable_ids[u] = f.global_id
            elif pred == "holding" and len(args) == 1:
                u = self.block_to_idx.get(args[0])
                if u is not None: self.holding_ids[u] = f.global_id
            elif pred == "handempty":
                self.handempty_id = f.global_id
                
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_names:
                pred, args = self._parse_name(g_name)
                if pred == "on" and len(args) == 2:
                    u, v = self.block_to_idx.get(args[0]), self.block_to_idx.get(args[1])
                    if u is not None and v is not None:
                        self.goal_on_ids[(u, v)] = self.on_ids_by_top_bottom[u][v]

    def _parse_name(self, name):
        name = name.strip()
        if name.startswith(("+", "-")): name = name[1:]
        name = name.replace("(", "").replace(")", "").replace(",", " ")
        if "[" in name:
            pred = name[:name.find("[")]
            args = name[name.find("[")+1:-1].split()
            return pred, args
        parts = name.split()
        return (parts[0], parts[1:]) if parts else (None, [])

    def _get_above(self, block_idx, state):
        if block_idx in self._above_cache: return self._above_cache[block_idx]
        if (state >> self.clear_ids[block_idx]) & 1: return 0
        if (state >> self.holding_ids[block_idx]) & 1: return 0
        count, curr = 0, block_idx
        for _ in range(self.num_blocks):
            found = False
            for top_idx, f_id in self.on_ids_by_bottom[curr]:
                if (state >> f_id) & 1:
                    count += 1
                    curr = top_idx
                    found = True
                    break
            if not found: break
        self._above_cache[block_idx] = count
        return count

    def _compute(self, state, task_network):
        self._above_cache = {}
        h_tasks = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                h_tasks += task.cost
            else:
                h_tasks += self._estimate_abstract_task(task.name, state)
        
        h_goals = 0
        for (u, v), goal_id in self.goal_on_ids.items():
            if not ((state >> goal_id) & 1):
                h_goals += 2 + 2 * self._get_above(u, state)
        
        total_h = max(h_tasks, h_goals)
        if self.handempty_id != -1 and not ((state >> self.handempty_id) & 1):
            total_h += 1
        return total_h + 0.001 * len(task_network)

    def _estimate_abstract_task(self, task_key, state):
        if task_key not in self.task_cache:
            name, args = self._parse_name(task_key)
            arg_indices = tuple(self.block_to_idx.get(a) for a in args)
            self.task_cache[task_key] = (name, arg_indices)
        
        name, indices = self.task_cache[task_key]
        if name == "do_put_on":
            u, v = indices
            if (state >> self.on_ids_by_top_bottom[u][v]) & 1: return 0
            return 2*self._get_above(u, state) + 2*self._get_above(v, state) + (2 if not ((state >> self.ontable_ids[v]) & 1) else 0) + 2
        elif name == "do_clear":
            return 2 * self._get_above(indices[0], state)
        elif name == "do_on_table":
            u = indices[0]
            if (state >> self.ontable_ids[u]) & 1 and (state >> self.clear_ids[u]) & 1: return 0
            return 2 * self._get_above(u, state) + 2
        elif name == "do_move":
            return 2 * self._get_above(indices[0], state) + 2
        return 0

    def __call__(self, parent_node, node):
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
