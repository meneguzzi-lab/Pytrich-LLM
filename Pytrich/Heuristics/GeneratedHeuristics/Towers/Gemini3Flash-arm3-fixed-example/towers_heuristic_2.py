import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The central constraint in Towers is the hierarchical/recursive dependency: to move a disk n, 
    all disks 1 to n-1 must first be moved to the auxiliary peg, and then moved back on top of n 
    at the destination. This leads to a theoretical minimum move count of 2^n - 1.

    Admissibility Sketches:
    - Each task in the network (shiftTower, selectDirection, rotateTower) represents an intention to move 
      a stack of k disks, which requires at least 2^k - 1 atomic 'move' actions.
    - The 'exchange' task represents moving one disk plus a recursive 'rotateTower' of k disks, 
      requiring at least 1 + (2^k - 1) = 2^k actions.
    - These values are combined additively across the task network to estimate total remaining effort.
    """

    def __init__(self, towers_gemini3flash_heuristic_2=None, name="towers_gemini3flash_heuristic_2"):
        super().__init__(name=name)
        self.on_ids = {}  # Map: ring -> obj -> global_id
        self.tower_top_ids = {}  # Map: obj -> tower -> global_id
        self.tower_top_candidates = {}  # Map: tower -> List[possible_top_objs]
        self.ring_names = set()
        self.tower_names = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        """Called once before search starts. Pre-parse facts and IDs."""
        self.model = model
        for f in model.facts:
            pred, args = self._parse_name(f.name)
            if not pred: continue
            
            if pred == "on":
                r, o = args[0], args[1]
                if r not in self.on_ids: self.on_ids[r] = {}
                self.on_ids[r][o] = f.global_id
                self.ring_names.add(r)
            elif pred == "towerTop":
                o, t = args[0], args[1]
                if o not in self.tower_top_ids: self.tower_top_ids[o] = {}
                self.tower_top_ids[o][t] = f.global_id
                if t not in self.tower_top_candidates: self.tower_top_candidates[t] = []
                self.tower_top_candidates[t].append(o)
                self.tower_names.add(t)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_name(self, name: str):
        """Parses both grounded facts and grounded tasks."""
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

    def _count_rings_from(self, state, start_obj, tower_name):
        """Counts disks in a stack starting from a specific ring down to the tower base."""
        if start_obj == tower_name: return 0
        count = 1
        curr = start_obj
        while curr != tower_name:
            found = False
            on_choices = self.on_ids.get(curr, {})
            for r_below, on_id in on_choices.items():
                if (state >> on_id) & 1:
                    curr = r_below
                    if curr != tower_name:
                        count += 1
                    found = True
                    break
            if not found: break
        return count

    def _count_rings_on(self, state, tower_name):
        """Identifies the top of the tower and counts all disks on it."""
        top_obj = None
        candidates = self.tower_top_candidates.get(tower_name, [])
        for obj in candidates:
            tid = self.tower_top_ids[obj].get(tower_name)
            if tid is not None and (state >> tid) & 1:
                top_obj = obj
                break
        if top_obj == tower_name or top_obj is None:
            return 0
        return self._count_rings_from(state, top_obj, tower_name)

    def _compute(self, state, task_network):
        """Calculates heuristic value by summing estimated costs of all tasks in the network."""
        h = 0
        for task in task_network:
            name, args = self._parse_name(task.name)
            if not name: continue
            
            if name in ["move_abstract", "move"]:
                h += 1
            elif name in ["shiftTower", "rotateTower"]:
                # These tasks involve moving the whole stack on the source tower
                t_src = args[0]
                k = self._count_rings_on(state, t_src)
                if k > 0: h += (2**k - 1)
            elif name == "selectDirection":
                # selectDirection(r, t1, t2, t3) involves moving stack starting from r on t1
                r_start, t_src = args[0], args[1]
                k = self._count_rings_from(state, r_start, t_src)
                if k > 0: h += (2**k - 1)
            elif name == "exchange":
                # exchange(t1, t2, t3) involves one move and then a rotateTower of the stack on t2
                t2 = args[1]
                k2 = self._count_rings_on(state, t2)
                if k2 > 0:
                    h += (2**k2)  # 1 move + (2^k2 - 1) for the tower
                else:
                    # Base case (exchangeClear) costs 0 as it performs no further tasks
                    h += 0
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Heuristic evaluation called for every node expansion."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
