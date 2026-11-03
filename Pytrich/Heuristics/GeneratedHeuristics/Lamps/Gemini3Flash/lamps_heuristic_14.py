import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
The bottleneck in this domain is the serial nature of turn-based state changes.
Each cell status change (flip) is an unavoidable primitive cost. 
Each flip is strictly tied to an HTN method decomposition (either a turn or a propagation),
meaning every state change requires at least 2 units of cost (1 for the flip action 
and 1 for the method's precondition checker).

Admissibility Sketch:
1. Mismatched Cells: Each cell whose status differs from the goal MUST be flipped at least once.
   In Pytrich, a flip costs 1 and its triggering method's precondition costs 1. 
   Thus, 2 * number_of_mismatched_cells is a lower bound on cost to fix the grid.
2. HTN Structural Cost: The domain forces every turn to have 4 propagation subtasks.
   Even if a direction flips 0 cells, it still incurs at least 1 unit of cost for the 
   precondition of its 'end' method. Additionally, the 'play' loop and 'end_game'
   task add mandatory overhead actions. 
3. Composite Bound: By combining the 2M mismatch cost with the fixed overhead derived 
   from the current task network state (play/turn/propagate), we get a tight lower bound.
"""

class LampsGemini3FlashHeuristic14(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_14=None, name="LampsGemini3FlashHeuristic14"):
        super().__init__(name=name)
        self.on_fact_ids = {}     # (x, y, status) -> global_id
        self.goal_fact_ids = {}   # (x, y) -> global_id of goal status
        self.play_task_id = -1
        self.turn_task_id = -1
        self.propagate_task_ids = set()
        self.flip_task_id = -1

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute_h(initial_node.state, initial_node.task_network)
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
        # Map all 'on' facts to coordinate-status tuples
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_fact_ids[(x, y, s)] = fact.global_id
        
        # Map goal statuses from the bitwise model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.goal_fact_ids[(x, y)] = self.on_fact_ids.get((x, y, s))

        # Identify Task IDs for structural cost estimation
        for abtask in self.model.abstract_tasks:
            name = abtask.name.lower()
            if "play" in name: self.play_task_id = abtask.global_id
            elif "turn" in name: self.turn_task_id = abtask.global_id
            elif "propagate" in name: self.propagate_task_ids.add(abtask.global_id)
        
        for op in self.model.operators:
            if "flip" in op.name.lower(): self.flip_task_id = op.global_id

    def _compute_h(self, state, task_network):
        # Count mismatched cells based on bitwise comparison with goal IDs
        mismatches = 0
        for (x, y), g_id in self.goal_fact_ids.items():
            if g_id is not None and not ((state >> g_id) & 1):
                mismatches += 1
        
        # Core cost: every mismatched cell requires 2 actions (precon + flip)
        h = 2 * mismatches
        
        # Analyze current task network to identify structural overhead
        has_play = False
        has_turn = False
        has_flip = False
        prop_count = 0
        
        for task in task_network:
            tid = task.global_id
            if tid == self.play_task_id: has_play = True
            elif tid == self.turn_task_id: has_turn = True
            elif tid == self.flip_task_id: has_flip = True
            elif tid in self.propagate_task_ids: prop_count += 1

        # HTN overhead calculation based on current decomposition state
        # If play is in network, we eventually incur do_play, do_turn, 4 props, and end_game.
        if has_flip:
            # If flip is already in network, do_turn precon is paid, but props are pending
            h += 4
        elif has_turn:
            # Turn precon is pending, along with 4 props and end_game
            h += 5
        elif has_play:
            # do_play precon is pending, then turn, then 4 props, then end_game
            if mismatches > 0:
                h += 6
            else:
                h = 1 # Just end_game
        elif prop_count > 0:
            # Each propagation in the network will result in at least an 'end' method action
            h += prop_count + 1
        elif mismatches > 0:
            # If tasks are empty but goals aren't met, search will eventually backtrack/fail
            # This is a large value to discourage goal-incomplete leaf nodes
            h += 1000

        return float(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute_h(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
