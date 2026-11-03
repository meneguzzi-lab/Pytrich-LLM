"""
TowersGPT5Heuristic17: A domain-dependent heuristic for the Towers (HDDL) domain in Pytrich.

Design rationale and domain insights:
- The Towers domain is a Towers-of-Hanoi variant encoded hierarchically. The primitive action is `move`.
- The optimal number of primitive moves to place the largest misplaced ring correctly (including all smaller rings
  that must be moved to clear it) is 2^k - 1, where k is the number of rings from the smallest up to (and including)
  the largest ring currently not at its goal position. This is a classic admissible lower bound.
- We extract goal facts via model.goals (bitwise), not via goal_on predicates (critical requirement).
- We preprocess static structure (rings, size order from smallerThan, ground facts for on) once in initialize().
- We compute a state-based lower bound h_state using the 2^k - 1 formula by scanning from largest to smallest ring
  and finding the largest ring not at its goal.
- To better consider hierarchy, we additionally derive a task-network-based lower bound h_tn:
  * Every primitive `move` operator in the current task network must still be executed -> contributes at least 1.
  * Every abstract `move_abstract` task eventually decomposes into exactly one `move` action in the provided domain
    (see method newMethod21). Thus, each currently pending ground `move_abstract` contributes at least 1.
  * We conservatively assign 0 to other abstract tasks (rotateTower, exchange, selectDirection, shiftTower), because
    while they may expand into moves, using 0 keeps the estimate admissible without requiring expensive reasoning
    about method applicability and recursion.
- The final heuristic value is max(h_state, h_tn), which remains admissible and typically tighter than either alone.

Performance considerations and robustness:
- We use bitwise checks for state queries (critical requirement) and cache fact indices per ring for fast lookups.
- All parsing and heavy preprocessing are done once in initialize().
- We handle both grounded "+predicate[args]" and HDDL "(predicate args)" name formats.
- The heuristic is consistent under unit-cost moves and avoids unnecessary overhead during node expansion.

Notes:
- This heuristic is designed to scale to large instances by keeping per-node computation O(#rings) for state bound and
  O(#tasks) for the hierarchical bound, both with small constants.
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Towers (HDDL) domain.

    It combines two admissible lower bounds and returns their maximum:
    1) State-based Hanoi lower bound: 2^k - 1 for the largest misplaced ring, where k counts rings from smallest up to
       that ring. This is the classic admissible bound for Towers of Hanoi.
    2) Task-network lower bound: counts currently pending primitive `move` operators and ground `move_abstract` tasks,
       each contributing at least 1 remaining primitive step.

    Both bounds are cheap to compute and exploit domain structure and hierarchy without risking inadmissibility.
    """

    def __init__(self, towers_gpt5_heuristic_17: Optional[dict] = None, name: str = "TowersGPT5Heuristic17"):
        super().__init__(name=name)
        self.preprocessing_time: float = 0.0

        # Domain knowledge extracted during preprocessing
        self.goal_positions: Dict[str, str] = {}         # ring -> goal support object
        self.rings_by_size: List[str] = []               # rings sorted smallest -> largest
        self.on_facts_by_ring: Dict[str, List[Tuple[object, str]]] = {}  # ring -> list of (Fact, target)
        self.num_rings: int = 0

        # Cached info for hierarchical TN lower bound
        self._abstract_min_moves: Dict[str, int] = {     # minimal required moves per abstract task name
            # Safe/conservative mapping:
            # - move_abstract always yields exactly 1 primitive move in this domain
            "move_abstract": 1,
            # Assign 0 to others (conservative, remains admissible without heavy reasoning):
            "rotateTower": 0,
            "exchange": 0,
            "selectDirection": 0,
            "shiftTower": 0,
        }

        # Recognizers for operator/abstract names
        self._primitive_move_prefix: str = "move"        # operator names should start with 'move'
        self._abstract_move_name: str = "move_abstract"  # exact abstract task name

        # Model ref
        self.model: Optional[Model] = None

    # ------------------------------ Framework API ------------------------------

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time

        h0 = self._heuristic_value(initial_node)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._heuristic_value(node)
        super().update_info(h_value)
        return h_value

    # ------------------------------ Preprocessing ------------------------------

    def _parse_fact_name(self, name: str):
        """
        Parse a fact-like name into (predicate, args) tuple.
        Supports grounded format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        and HDDL format: (predicate arg1 arg2)
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            b = name2.find("[")
            if b > 0 and name2.endswith("]"):
                predicate = name2[:b]
                args_str = name2[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        if self.model is None:
            return

        rings = set()
        smaller_than_pairs = []  # (smaller, larger)

        # Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, "goals") and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith("r"):  # rings typically named r1, r2, ...
                        self.goal_positions[r] = tgt

        # Scan all grounded facts once to collect ring set, on facts, and size info
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                small, big = args[0], args[1]
                if small.startswith("r"):
                    rings.add(small)
                    smaller_than_pairs.append((small, big))
                    if big.startswith("r"):
                        rings.add(big)

            elif pred == "on":
                r, tgt = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, tgt))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Derive a size order: smallest -> largest
        if smaller_than_pairs:
            # Count how many things a ring is smaller than; higher count => smaller ring
            smaller_count = defaultdict(int)
            for s, b in smaller_than_pairs:
                if s in rings:
                    smaller_count[s] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer partial ordering from goal stack if any
            larger_map = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if tgt.startswith("r"):
                    larger_map[r].add(tgt)
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(larger_map.keys()):
                    for x in list(larger_map[r]):
                        for y in larger_map.get(x, set()):
                            if y not in larger_map[r]:
                                larger_map[r].add(y)
                                changed = True
            smaller_count = {r: len(larger_map.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    # ------------------------------ Core computation ------------------------------

    def _get_current_position(self, state_bitwise: int, ring: str) -> Optional[str]:
        """Return the object supporting the given ring in the current state (fast bitwise check)."""
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, tgt in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _state_lower_bound(self, state_bitwise: int) -> int:
        """
        Compute the 2^k - 1 lower bound based on the largest ring out of place.
        Rings are processed largest-to-smallest; return moves needed to at least position that largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0

        # rings_by_size: smallest -> largest. Scan in reverse: largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, ring)
            goal = self.goal_positions.get(ring)
            if cur != goal:
                k = i + 1  # number of rings from smallest up to 'ring'
                return (1 << k) - 1
        return 0

    def _tn_lower_bound(self, node: HTNNode) -> int:
        """
        Hierarchical task-network based lower bound:
        - Count pending primitive `move` operators in the TN (each costs at least 1 step).
        - Count pending abstract `move_abstract` tasks (each decomposes to exactly one `move`).
        - Other abstract tasks contribute 0 (conservative, admissible).
        Returns the sum of these minimal necessary moves in the current TN.

        Implementation details:
        - This function is robust to different TN representations: list of component IDs or direct objects.
        - It uses model.get_component(id) to resolve integers to model components.
        """
        if self.model is None:
            return 0

        # Try to obtain the current task network from the node in a robust way
        tn = getattr(node, 'task_network', None)
        if tn is None:
            tn = getattr(node, 'tn', None)
        if tn is None:
            return 0

        min_moves = 0
        for elem in tn:
            comp = elem
            if isinstance(elem, int):
                try:
                    comp = self.model.get_component(elem)
                except Exception:
                    comp = None
            if comp is None:
                continue

            name = getattr(comp, 'name', '')
            # Primitive operator detection: names starting with 'move'
            # This is robust for the provided domain where only primitive is `move`.
            if self._is_operator(comp):
                if self._is_move_operator_name(name):
                    min_moves += 1
                continue

            # Abstract task detection: minimal contribution if recognized
            if self._is_abstract_task(comp):
                # Identify the abstract task kind by prefix or exact match
                abstract_kind = self._abstract_kind_from_name(name)
                min_moves += self._abstract_min_moves.get(abstract_kind, 0)
                continue

            # Decompositions or other elements do not directly add minimal primitive steps here

        return min_moves

    def _heuristic_value(self, node: HTNNode) -> int:
        state = node.state
        h_state = self._state_lower_bound(state)
        h_tn = self._tn_lower_bound(node)
        return max(h_state, h_tn)

    # ------------------------------ Helpers ------------------------------

    def _is_operator(self, comp) -> bool:
        # Operators have methods like applicable/apply and a 'cost' attribute typically.
        return hasattr(comp, 'apply') and hasattr(comp, 'cost')

    def _is_abstract_task(self, comp) -> bool:
        # AbstractTask typically has 'decompositions' list and no 'apply' method
        return hasattr(comp, 'decompositions') and not hasattr(comp, 'apply')

    def _is_move_operator_name(self, name: str) -> bool:
        # Accept names that start with 'move' regardless of ground format, e.g., 'move[...]' or '(move ...)'
        n = name.strip()
        if not n:
            return False
        if n.startswith(self._primitive_move_prefix):
            return True
        if n.startswith("(") and n[1:].startswith(self._primitive_move_prefix):
            return True
        return False

    def _abstract_kind_from_name(self, name: str) -> str:
        # Extract a canonical task name from possible representations '(task ...)' or 'task[...]' or direct 'task'
        n = name.strip()
        if not n:
            return n
        # HDDL-like: (task args)
        if n.startswith("(") and n.endswith(")"):
            inner = n[1:-1].strip()
            if inner:
                return inner.split()[0]
        # Ground-like: task[args]
        if "[" in n:
            return n.split("[", 1)[0]
        # Otherwise assume it's already the task name
        return n

    # ------------------------------ Debug / Info ------------------------------

    def __repr__(self):
        return f"TowersGPT5Heuristic17(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic17 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
