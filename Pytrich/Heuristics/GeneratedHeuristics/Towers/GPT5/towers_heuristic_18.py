import time
from collections import defaultdict
from typing import List, Union, Optional

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Key ideas and rationale:
    - The domain encodes a Towers-of-Hanoi style problem over rings (RING) stacked on
      rings and towers (OBJ/TOWER), with a single primitive operator 'move' and a set
      of hierarchical tasks (shiftTower, selectDirection, rotateTower, exchange,
      move_abstract). The hierarchy mirrors the classic recursive solution.
    - The minimum number of moves to relocate the largest misplaced ring (and all
      smaller ones above it) is 2^k - 1, where k is the number of rings from the
      smallest up to (and including) that largest misplaced one. This is a tight and
      admissible lower bound (the classic Hanoi bound).
    - We combine two admissible lower bounds using max():
        1) Hanoi-bound (state-based): derived from ring positions vs. goals.
        2) Task-network bound (hierarchy-aware): a cheap lower bound computed from the
           current task network, counting at least one primitive 'move' for each
           outstanding 'move_abstract' and for each 'rotateTower' (since it always
           includes a 'move_abstract' before 'exchange'). This leverages the task
           structure but stays conservative (admissible) by ignoring other tasks that
           could decompose to zero moves (e.g., exchangeClear).
    - To ensure scalability and efficiency:
        * All state checks use fast bitwise operations.
        * Goals are extracted from model.goals (bitset) via model.state_explicit_repr.
        * Grounded fact names are parsed in the "+predicate[arg1,arg2]" format.

    The final heuristic is:
        h(node) = max( HanoiBound(state), TaskNetworkBound(node.tn) )

    Both parts are admissible individually; taking max preserves admissibility and
    typically reduces node expansions significantly.
    """

    def __init__(self, towers_gpt5_heuristic_18: Optional[object] = None, name: str = "TowersGPT5Heuristic18"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}       # ring -> target object from goals (e.g., { 'r1': 't3' })
        self.rings_by_size: List[str] = []  # rings sorted smallest -> largest
        self.on_facts_by_ring = {}     # ring -> list of (Fact, target_obj)
        self.num_rings = 0

        # For hierarchy-aware bound: map component global_id -> base name (operator/abstract task)
        self.comp_base_name = {}  # int -> str (e.g., {op_id: 'move', at_id: 'rotateTower'})

    # --------------------------- Heuristic Lifecycle ---------------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h0 = self._hanoi_bound(initial_node.state)
        h0 = max(h0, self._task_network_bound(initial_node))

        return super().initialize(model, h0)

    # ------------------------------ Preprocessing ------------------------------
    def _parse_fact_name(self, name: str):
        """
        Parse a fact name in grounded format: "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]".
        Returns: (predicate, [arg1, arg2, ...]) or (None, []) if parsing fails.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _base_name(self, raw_name: str) -> str:
        """
        Extract the base symbol of a (possibly grounded) operator/abstract task name.
        Handles likely formats: "name(...)", "name[...]", or space-separated tokens.
        """
        s = raw_name.strip()
        if not s:
            return s
        # Strip parentheses/brackets with arguments
        for open_c, close_c in (("(", ")"), ("[", "]")):
            if open_c in s and s.endswith(close_c):
                return s.split(open_c, 1)[0]
        # Fallback: split by space and take first token
        return s.split()[0]

    def _preprocess(self):
        # 1) Extract goal positions from model.goals (bitwise goal state)
        if hasattr(self.model, "goals") and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g in goal_fact_names:
                predicate, args = self._parse_fact_name(g)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2) Extract static info and dynamic 'on' fact indexing
        rings = set()
        smaller_than_pairs = []  # (smaller_ring, larger_obj)

        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller, larger = args[0], args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_pairs.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self.num_rings = len(rings)

        # 3) Sort rings by size (smallest -> largest)
        # Use smallerThan counts as a proxy: smaller rings appear as 'smaller' more often
        if self.num_rings > 0:
            smaller_count = defaultdict(int)
            for smaller, _ in smaller_than_pairs:
                if smaller in rings:
                    smaller_count[smaller] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

        # 4) Preprocess component base names to support hierarchy-aware bound
        try:
            # Operators
            for op in self.model.operators:
                self.comp_base_name[op.global_id] = self._base_name(op.name)
            # Abstract tasks
            for at in self.model.abstract_tasks:
                self.comp_base_name[at.global_id] = self._base_name(at.name)
        except Exception:
            # Be robust in case the model structure is different
            pass

    # ---------------------------- Utility functions ----------------------------
    def _get_current_position(self, state_bitwise: int, ring: str) -> Optional[str]:
        """
        Return the object the given ring is currently 'on' by checking which grounded
        '+on[ring,obj]' fact is true in the current state (bitwise check).
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, target in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ----------------------------- Lower bounds (h) ----------------------------
    def _hanoi_bound(self, state_bitwise: int) -> int:
        """
        Classic admissible bound for Towers of Hanoi:
        - Scan rings from largest to smallest;
        - Find the largest ring not at goal position;
        - If its index among the sorted rings is i (0-based, smallest first), then
          at least 2^(i+1) - 1 moves are needed to move all rings up to that one.
        """
        if not self.rings_by_size:
            return 0

        for i in range(len(self.rings_by_size) - 1, -1, -1):  # largest -> smallest
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # If no explicit goal for this ring, skip it
                continue
            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                k = i + 1  # number of rings from smallest (index 0) to this one
                return (1 << k) - 1  # 2^k - 1
        return 0

    def _get_task_network(self, node: HTNNode) -> Optional[List[Union[int, object]]]:
        """
        Best-effort retrieval of the current task network from the node.
        Returns a list of component IDs or component objects, or None if unavailable.
        """
        for attr in ("tn", "task_network", "tasks", "open_tasks"):
            if hasattr(node, attr):
                tn = getattr(node, attr)
                if isinstance(tn, list):
                    return tn
        return None

    def _task_network_bound(self, node: HTNNode) -> int:
        """
        Hierarchy-aware admissible bound from the current task network:
        - Each 'move_abstract' requires at least 1 primitive 'move'.
        - Each 'rotateTower' has at least one leading 'move_abstract', so also >= 1 move.
        - Other abstract tasks (shiftTower, selectDirection, exchange) can decompose to
          zero primitive moves in best-case variants (e.g., exchangeClear), so we do
          not count them to keep the bound admissible.

        We count only what's guaranteed regardless of state: conservative but useful.
        """
        tn = self._get_task_network(node)
        if not tn:
            return 0

        count = 0
        for item in tn:
            comp = None
            base = ""
            try:
                if isinstance(item, int):
                    comp = self.model.get_component(item)
                    base = self.comp_base_name.get(getattr(comp, "global_id", -1), self._base_name(getattr(comp, "name", "")))
                else:
                    comp = item
                    gid = getattr(comp, "global_id", -1)
                    nm = getattr(comp, "name", "")
                    base = self.comp_base_name.get(gid, self._base_name(nm))
            except Exception:
                # Fallback: try to use str(item)
                base = self._base_name(str(item))

            if base == "move_abstract" or base == "rotateTower":
                count += 1
            elif base == "move":
                # If a primitive move is already in the TN, it's at least 1 action
                count += 1
            # else: do not add (conservative)
        return count

    # ---------------------------------- API -----------------------------------
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_hanoi = self._hanoi_bound(node.state)
        h_tn = self._task_network_bound(node)
        h_value = max(h_hanoi, h_tn)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT5Heuristic18(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic18 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic18 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
