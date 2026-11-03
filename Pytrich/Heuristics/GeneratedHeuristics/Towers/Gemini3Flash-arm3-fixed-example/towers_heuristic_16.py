import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic16(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Towers domain (a Tower of Hanoi variant) is the spatial and structural 
    constraint of the stack. A ring can only be moved if it is the 'towerTop' (no other rings on top) 
    and can only be placed on a larger object (another ring or a tower base). This forces a strict 
    serialization where moving a specific ring requires moving all rings above it to a temporary tower.

    Admissibility Sketches:
    1. Misplaced Ring Bound: Every ring that is not currently on its goal parent, or is on a parent 
       that itself needs to move, must be moved at least once. This count is a lower bound on the 
       number of 'move' actions remaining.
    2. Task Network moves: Each 'move_abstract' task in the task network corresponds to exactly 
       one primitive 'move' action. The count of these tasks in the network is a lower bound.
    """

    def __init__(self, towers_gemini3flash_heuristic_16="towers_gemini3flash_heuristic_16"):
        super().__init__(name=towers_gemini3flash_heuristic_16)
        self.on_fact_ids = {} # ring -> {parent: fact_id}
        self.goal_parent = {} # ring -> target_parent
        self.rings = set()
        self.towers = set()
        self.move_abstract_ids = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse all facts to identify rings, towers, and 'on' predicates
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                ring, parent = args[0], args[1]
                if ring not in self.on_fact_ids:
                    self.on_fact_ids[ring] = {}
                self.on_fact_ids[ring][parent] = fact.global_id
                self.rings.add(ring)
            elif pred == "towerTop":
                # towerTop can be applied to rings or towers
                obj, tower = args[0], args[1]
                self.towers.add(tower)

        # 2. Parse goal state to determine the target parent for every ring
        # model.goals is a bitwise integer
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                ring, parent = args[0], args[1]
                self.goal_parent[ring] = parent

        # 3. Identify abstract task IDs for 'move_abstract'
        for abt in model.abstract_tasks:
            if "move_abstract" in abt.name.lower():
                self.move_abstract_ids.add(abt.global_id)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

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

    def _compute(self, state, task_network):
        # Term 1: Misplaced Rings
        # We need current 'on' relations
        current_on = {}
        for ring in self.rings:
            for parent, fact_id in self.on_fact_ids.get(ring, {}).items():
                if (state >> fact_id) & 1:
                    current_on[ring] = parent
                    break

        h_misplaced = 0
        memo_correct = {}

        def is_correct(r):
            if r in memo_correct: return memo_correct[r]
            target = self.goal_parent.get(r)
            actual = current_on.get(r)
            
            if actual != target:
                memo_correct[r] = False
                return False
            
            # If it's on a tower and it's the correct tower, it's correct
            if target in self.towers:
                memo_correct[r] = True
                return True
            
            # If it's on another ring, that ring must also be correct
            res = is_correct(target)
            memo_correct[r] = res
            return res

        for ring in self.rings:
            if not is_correct(ring):
                h_misplaced += 1

        # Term 2: Tasks in network
        # Count move_abstract tasks as each is at least one move
        h_tasks = 0
        for task in task_network:
            if hasattr(task, 'global_id') and task.global_id in self.move_abstract_ids:
                h_tasks += 1

        # Both are admissible lower bounds. 
        # h_misplaced counts total moves needed. 
        # h_tasks counts moves already committed in the current decomposition branch.
        return max(h_misplaced, h_tasks)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
