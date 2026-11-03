from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class TowersGemini3FlashHeuristic13(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Towers domain is the recursive stack dependency. 
    A ring cannot be moved to its goal position unless the target peg (or disk) is clear of any smaller rings, 
    and the ring itself is clear of any rings on top of it. This serializes the work such that 
    the largest ring must be placed in its final position before the next largest can be permanently placed.

    Admissibility Sketches:
    1. Misplaced Ring Count (h1): Every ring that is not in its "stable" goal position must be moved at least once. 
       A ring is stable if it is on its goal parent AND its goal parent is also stable (or is the target tower).
    2. Task Network Size (h2): Every task in the HTN network represents at least one future action or decomposition 
       that must occur to reach the goal state (where the task network is empty).

    Final Heuristic: h = h1 + h2. While strictly speaking h1 + h2 might slightly over-estimate if a misplaced 
    ring is handled by multiple tasks in the network, in practice, this provides much stronger guidance for 
    the recursive decomposition required by Towers.
    """
    def __init__(self, towers_gemini3flash_heuristic_13=1, name="towers_gemini3flash_heuristic_13"):
        super().__init__(name=name)
        self.weight = towers_gemini3flash_heuristic_13
        self.on_fact_ids = {}
        self.goal_on_map_rev = {}
        self.towers = set()
        self.rings = set()
        self.num_rings = 0
        self.initialized = False

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        all_objs = set()
        ring_objs = set()

        # Parse all facts to map the 'on' predicate and identify objects
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                ring_objs.add(args[0])
                all_objs.add(args[0])
                all_objs.add(args[1])
                self.on_fact_ids[(args[0], args[1])] = f.global_id

        self.rings = ring_objs
        self.towers = all_objs - ring_objs
        self.num_rings = len(self.rings)

        # Parse goals to build the target structure
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        goal_on_map = {}
        for f_name in goal_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred in ["on", "goal_on"]:
                goal_on_map[args[0]] = args[1]
        
        # Reverse map for bottom-up checking: parent -> child
        self.goal_on_map_rev = {v: k for k, v in goal_on_map.items()}
        self.initialized = True

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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

    def _get_stable_count(self, state_bitwise):
        """
        Counts how many rings are correctly placed in their goal stack starting from the tower upwards.
        A ring is stable if it's on the correct object and that object is also stable.
        """
        stable_count = 0
        for tower in self.towers:
            current_base = tower
            while current_base in self.goal_on_map_rev:
                target_ring = self.goal_on_map_rev[current_base]
                fid = self.on_fact_ids.get((target_ring, current_base))
                if fid is not None and (state_bitwise >> fid) & 1:
                    stable_count += 1
                    current_base = target_ring
                else:
                    break
        return stable_count

    def _compute(self, state, task_network):
        if not self.initialized:
            return 0
        
        # Term 1: Misplaced rings (rings not part of the stable goal stack)
        h1 = self.num_rings - self._get_stable_count(state)
        
        # Term 2: Complexity of the remaining task network
        h2 = len(task_network)
        
        return h1 + h2

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
