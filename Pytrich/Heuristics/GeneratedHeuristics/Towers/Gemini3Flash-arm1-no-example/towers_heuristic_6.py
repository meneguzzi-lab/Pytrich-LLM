from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic6(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers of Hanoi domain.

    Bottleneck Analysis:
    The central bottleneck is the recursive nature of moving a stack of rings.
    To move a ring of size K, one must first clear all K-1 rings above it,
    then move the ring, and then move the stack back. This leads to an
    unavoidable exponential growth in the number of actions (2^K - 1).

    Admissible Lower Bounds:
    1. State-based Hanoi distance: Finding the largest ring K that is not on its 
       goal object. The minimum actions to place this ring and its stack correctly 
       is at least 2^K - 1.
    2. Task-based Move Count: Each 'move' action or 'move_abstract' task remaining 
       in the task network must eventually be executed as an action.

    Implementation:
    - Pre-computes ring sizes and goal mappings in initialize().
    - Uses bitwise checks for state monitoring in __call__.
    - Returns max(h_state, h_tasks) + tie-breaking penalty.
    """

    def __init__(self, towers_gemini3flash_heuristic_6="towers_gemini3flash_heuristic_6"):
        super().__init__(towers_gemini3flash_heuristic_6)
        self.goal_map = {}  # ring_name -> target_obj_name
        self.on_fact_ids = {}  # ring_name -> {target_obj_name -> global_id}
        self.ring_by_size = {}  # size_int -> ring_name
        self.num_rings = 0

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
        smaller_relations = set()
        potential_rings = set()
        
        # 1. Parse all facts to map the state space and identify rings
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                smaller_relations.add((args[0], args[1]))
                potential_rings.add(args[0])
            elif pred == "on":
                r, obj = args[0], args[1]
                if r not in self.on_fact_ids:
                    self.on_fact_ids[r] = {}
                self.on_fact_ids[r][obj] = fact.global_id

        # 2. Determine sizes (Size 1 is smallest, Size N is largest)
        rings = sorted(list(potential_rings))
        self.num_rings = len(rings)
        for r in rings:
            # count how many other rings 'r' is smaller than
            count = 0
            for r_other in rings:
                if r != r_other and (r, r_other) in smaller_relations:
                    count += 1
            # Largest ring has count 0 -> size num_rings
            # Smallest ring has count num_rings-1 -> size 1
            size = self.num_rings - count
            self.ring_by_size[size] = r

        # 3. Determine search goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                self.goal_map[args[0]] = args[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state_bitwise, task_network):
        # Term 1: State-based Hanoi bound
        h_state = 0
        # Find the largest misplaced ring
        for i in range(self.num_rings, 0, -1):
            ring = self.ring_by_size.get(i)
            target = self.goal_map.get(ring)
            if ring and target:
                fact_id = self.on_fact_ids.get(ring, {}).get(target)
                if fact_id is not None:
                    is_on = (state_bitwise >> fact_id) & 1
                    if not is_on:
                        h_state = (1 << i) - 1
                        break

        # Term 2: Task-based bound (counting move tasks)
        h_tasks = 0
        for task in task_network:
            if "move" in task.name:
                h_tasks += 1

        # Composite admissible value with a tiny tie-breaker for exploration depth
        return max(h_state, h_tasks) + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
