from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic19(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Towers (Towers of Hanoi) domain is the strict ring-on-ring size constraint
    and the fact that only the top object of a tower can be moved. This forces a serialized, recursive
    structure where larger rings cannot move until all smaller rings are placed on a specific auxiliary tower.

    Admissibility Sketch:
    1. h_state: For each ring, we check its 'stability'. A ring is stable if it is at its goal position
       and the object it rests on is also stable (towers are always stable). 
       - If a ring is not at its goal tower, it requires at least 1 move (admissible).
       - If a ring is at its goal tower but is not stable (meaning a ring below it is incorrect),
         it must be moved off to another tower and eventually moved back. This requires at least 2 moves (admissible).
    2. h_tn: Every task in the task network represents future work. Since every 'move_abstract' 
       corresponds to an action and compound tasks decompose into at least one action (eventually),
       the length of the task network is a lower bound on the remaining effort.
    """

    def __init__(self, towers_gemini3flash_heuristic_19="towers_gemini3flash_heuristic_19"):
        super().__init__(towers_gemini3flash_heuristic_19)
        self.towers_gemini3flash_heuristic_19_on_ids = {} # ring -> list of (obj, global_id)
        self.towers_gemini3flash_heuristic_19_goal_on_map = {} # ring -> target_obj
        self.towers_gemini3flash_heuristic_19_sorted_rings = []
        self.towers_gemini3flash_heuristic_19_tower_names = set()

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
        all_objects = set()
        rings = set()
        smaller_than_counts = {}

        # 1. Collect objects and ring-size relationships
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            for a in args: all_objects.add(a)
            if pred == "smallerThan":
                r_small, r_large = args[0], args[1]
                rings.add(r_small)
                smaller_than_counts[r_small] = smaller_than_counts.get(r_small, 0) + 1

        # Towers are objects that are not rings
        self.towers_gemini3flash_heuristic_19_tower_names = all_objects - rings

        # 2. Sort rings by size: largest to smallest
        # Largest ring is smaller than the fewest other things.
        self.towers_gemini3flash_heuristic_19_sorted_rings = sorted(
            list(rings), key=lambda r: smaller_than_counts.get(r, 0)
        )

        # 3. Build goal map and current 'on' fact map
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.towers_gemini3flash_heuristic_19_goal_on_map[args[0]] = args[1]

        # If goal is missing in model.goals, check initial state for goal_on hints
        init_fact_names = self.model.state_explicit_repr(initial_node.state)
        for f_name in init_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == "goal_on" and args[0] not in self.towers_gemini3flash_heuristic_19_goal_on_map:
                self.towers_gemini3flash_heuristic_19_goal_on_map[args[0]] = args[1]

        # Map fact IDs for fast access to 'on' predicates
        self.towers_gemini3flash_heuristic_19_on_fact_ids = {r: [] for r in rings}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and args[0] in rings:
                self.towers_gemini3flash_heuristic_19_on_fact_ids[args[0]].append((args[1], f.global_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        h_state = 0
        # Track which objects are stable from bottom up
        stable_objects = set(self.towers_gemini3flash_heuristic_19_tower_names)
        
        # Iterate rings from largest to smallest
        for r_name in self.towers_gemini3flash_heuristic_19_sorted_rings:
            target_name = self.towers_gemini3flash_heuristic_19_goal_on_map.get(r_name)
            if not target_name:
                continue

            # Find current object r_name is on
            current_on_name = None
            for obj_name, fact_id in self.towers_gemini3flash_heuristic_19_on_fact_ids[r_name]:
                if (state >> fact_id) & 1:
                    current_on_name = obj_name
                    break
            
            if current_on_name == target_name:
                if target_name in stable_objects:
                    stable_objects.add(r_name)
                else:
                    # On goal tower but not stable: must move out and back (2 moves)
                    h_state += 2
            else:
                # Not on goal tower: must move at least once
                h_state += 1
        
        # len(task_network) is a lower bound for remaining work
        h_tn = len(task_network)

        # Sub-unit tie-breaker to favor deeper/more decomposed states
        return max(h_state, h_tn) + (0.001 * h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
