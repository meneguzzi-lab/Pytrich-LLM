from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic16(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Towers domain (a Tower of Hanoi variant) is the recursive dependency of rings. 
    A ring cannot be placed on its target position until the target position is clear and all larger rings 
    below it are correctly placed. This creates a serializing constraint where progress is measured 
    by the depth of the 'stable' stack being built on the target tower.

    Admissibility Sketches:
    1. Misplaced/Unstable Ring Count: Any ring that is not currently in its goal position, or is on a 
       base that itself is not stable, must be moved at least once. This provides an admissible 
       lower bound on the number of 'move' actions required.
    2. Task Network Size: Each task in the decomposition (shiftTower, selectDirection, rotateTower, 
       move_abstract) represents a necessary step in the hierarchical plan. While a single task 
       might expand into many actions, the count of tasks in the network serves as a proxy for 
       remaining structural work.
    """

    def __init__(self, towers_gemini3flash_heuristic_16="towers_gemini3flash_heuristic_16"):
        super().__init__(name=towers_gemini3flash_heuristic_16)
        self.goal_on_map = {}  # ring -> target_obj
        self.on_fact_ids = {}  # (ring, obj) -> global_id
        self.ring_ids = set()
        self.tower_ids = set()
        self.all_rings_list = []

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
        all_objs = set()
        
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.ring_ids.add(args[0])
                all_objs.add(args[0])
                all_objs.add(args[1])
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "smallerThan":
                all_objs.add(args[0])
                all_objs.add(args[1])
            elif pred == "towerTop":
                all_objs.add(args[0])
                all_objs.add(args[1])

        self.tower_ids = all_objs - self.ring_ids
        self.all_rings_list = list(self.ring_ids)

        # Parse goal state from model.goals bitmask
        for fact_id in range(model.ifacts_init, model.ifacts_end + 1):
            if (model.goals >> fact_id) & 1:
                fact = model.get_component(fact_id)
                pred, args = self._parse_fact_name(fact.name)
                if pred == "on":
                    self.goal_on_map[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _check_stability(self, obj_id, state_bitwise, stable_cache):
        if obj_id in self.tower_ids:
            return True
        if obj_id in stable_cache:
            return stable_cache[obj_id]
        
        target_obj = self.goal_on_map.get(obj_id)
        if target_obj is None:
            # If no goal specified for this ring, assume it doesn't need moving
            stable_cache[obj_id] = True
            return True
            
        on_fact_id = self.on_fact_ids.get((obj_id, target_obj))
        if on_fact_id is None or not ((state_bitwise >> on_fact_id) & 1):
            stable_cache[obj_id] = False
            return False
            
        is_base_stable = self._check_stability(target_obj, state_bitwise, stable_cache)
        stable_cache[obj_id] = is_base_stable
        return is_base_stable

    def _compute(self, state_bitwise, task_network):
        stable_cache = {}
        unstable_count = 0
        
        for ring in self.all_rings_list:
            if not self._check_stability(ring, state_bitwise, stable_cache):
                unstable_count += 1
        
        # Task network size weight - ensures we prefer deeper progress in decomposition
        tn_cost = len(task_network)
        
        # Each unstable ring must be moved (at least 1 action).
        # Each task in TN represents at least one potential action or future state change.
        return unstable_count + tn_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
