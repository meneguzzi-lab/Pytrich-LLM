from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic17(Heuristic):
    """
    Bottleneck Analysis: 
    The single most serializing constraint in the Towers domain is the nested dependency of rings. 
    To move a ring R to its goal target, all rings currently on top of R must be removed, 
    and all rings currently on the destination that are smaller than R must also be removed.

    Admissibility Sketches:
    1. Misplaced Rings (h_state): A ring is defined as 'well-placed' if it is on its goal target AND that 
       target (if it is a ring) is also well-placed. Any ring that is not well-placed MUST be moved at least once.
       Count of rings - Count of well-placed rings is a safe lower bound.
    2. Task Network Cost (h_tasks): Every task currently in the frontier (move_abstract, shiftTower, rotateTower, 
       exchange) represents at least one concrete action (move) that must be performed in the future. 
       A simple count of these tasks is a safe lower bound.
    3. Final H: max(h_state, h_tasks) is admissible because both terms are independent lower bounds.
    """

    def __init__(self, towers_gemini3flash_heuristic_17="towers_gemini3flash_heuristic_17"):
        super().__init__(name=towers_gemini3flash_heuristic_17)
        self.on_ids = {}
        self.goals_dict = {}
        self.rings = set()
        self.towers = set()

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
        
        # Parse Goal State
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                self.goals_dict[args[0]] = args[1]

        # Parse All Possible Facts to build object maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.rings.add(args[0])
            elif pred == "towerTop":
                if len(args) == 2:
                    self.towers.add(args[1])
                    
        # Clean up ring/tower sets based on domain knowledge
        self.rings = self.rings - self.towers
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise, task_network):
        # Term 1: State-based misplaced count
        # A ring is well-placed if it is on its goal and its base is well-placed
        well_placed = set()
        # Rings with no goals are considered well-placed by default
        for r in self.rings:
            if r not in self.goals_dict:
                well_placed.add(r)

        changed = True
        while changed:
            changed = False
            for r in self.rings:
                if r in well_placed: continue
                target = self.goals_dict.get(r)
                if target:
                    fid = self.on_ids.get((r, target))
                    # Bitwise check if ring is on its goal target
                    if fid is not None and (state_bitwise >> fid) & 1:
                        # Target must be a tower or a ring that is already well-placed
                        if target in self.towers or target in well_placed:
                            well_placed.add(r)
                            changed = True
        
        h_state = len(self.rings) - len(well_placed)

        # Term 2: Task-based count
        h_tasks = 0
        for task in task_network:
            name = task.name.lower()
            # Abstract tasks and move operators all imply at least one concrete action
            if "move" in name or "shifttower" in name or "rotatetower" in name or "exchange" in name:
                h_tasks += 1
            # selectDirection is a search/navigation task that might result in 0 moves if already at base

        # Return max for admissibility, plus tiny tie-breaker for progress
        return max(h_state, h_tasks) + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
