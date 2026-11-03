from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic7(Heuristic):
    """
    Bottleneck Analysis:
    The central bottleneck in the Towers (Hanoi) domain is the stack-ordering constraint. 
    A ring can only move if it is at the top of its current stack and the destination is valid 
    (empty or larger ring). This creates a recursive dependency where moving a bottom ring 
    requires moving all rings above it first.

    Admissible Lower Bounds:
    1. Misplaced Rings (Well-Placed Heuristic): A ring is only 'well-placed' if it is on its goal target 
       and that target (if a ring) is also well-placed. Any ring not well-placed must move at least once.
       Admissibility: Every 'move' action changes the 'on' predicate for exactly one ring. If a ring 
       is not in its final stable position, it requires at least one move to get there.
    2. Task Network moves: Each 'move_abstract' task or 'move' operator in the current task network 
       represents a physical move that must be accounted for.
    """

    def __init__(self, towers_gemini3flash_heuristic_7="towers_gemini3flash_heuristic_7"):
        super().__init__(name=towers_gemini3flash_heuristic_7)
        self.rings = []
        self.goal_on = {}  # ring -> target_obj
        self.on_ids = {}   # ring -> {obj -> global_id}
        self.move_abstract_id = -1
        self.move_op_ids = set()
        self.ring_set = set()

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
        self.model = model
        
        # 1. Map all 'on' facts and identify rings
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                ring, obj = args[0], args[1]
                if ring not in self.on_ids:
                    self.on_ids[ring] = {}
                self.on_ids[ring][obj] = f.global_id
                self.ring_set.add(ring)
        
        self.rings = list(self.ring_set)
        
        # 2. Extract Goal State from model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # 3. Identify Task/Operator IDs for network counting
        for abt in model.abstract_tasks:
            if "move_abstract" in abt.name.lower():
                self.move_abstract_id = abt.global_id
        for op in model.operators:
            if "move" == op.name.lower() or "move[" in op.name.lower():
                self.move_op_ids.add(op.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise, task_network):
        # Term 1: Count rings that are not in a stable goal position
        # A ring is stable if it's on its goal and its goal-support is also stable.
        well_placed = set()
        
        # Foundation iteration: Towers are essentially foundations that are always stable.
        changed = True
        while changed:
            changed = False
            for r in self.rings:
                if r in well_placed:
                    continue
                
                target = self.goal_on.get(r)
                if not target:
                    # If a ring has no specified goal_on, we can't determine stability easily
                    # In this domain, all rings usually have a goal_on.
                    continue
                
                # Check bitwise if on(r, target) is true
                fact_id = self.on_ids.get(r, {}).get(target)
                if fact_id is not None and (state_bitwise >> fact_id) & 1:
                    # Stable if on goal tower OR on a well-placed ring
                    if target not in self.ring_set or target in well_placed:
                        well_placed.add(r)
                        changed = True
        
        misplaced_count = len(self.rings) - len(well_placed)

        # Term 2: Tasks explicitly in the network that imply moves
        network_moves = 0
        for task in task_network:
            # Check if task is move_abstract or move action
            if hasattr(task, 'global_id'):
                if task.global_id == self.move_abstract_id or task.global_id in self.move_op_ids:
                    network_moves += 1
        
        # Admissible combine: max (Each misplaced ring requires at least 1 move overall)
        h = max(misplaced_count, network_moves)
        
        # Optional: very small tie-breaker for task network length to favor decomposition
        return h + (0.01 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
