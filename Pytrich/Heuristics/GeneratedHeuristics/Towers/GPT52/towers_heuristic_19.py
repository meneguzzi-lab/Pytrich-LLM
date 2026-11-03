"""towers_heuristic_19.py

Domain-dependent heuristic for the HDDL "towers" domain (Towers of Hanoi) for the Pytrich planner.

Design rationale (domain insights)
-------------------------------
The provided HDDL encodes a Hanoi-like process with hierarchical tasks:
  shiftTower -> selectDirection -> rotateTower -> (move_abstract, exchange)
  move_abstract -> move (primitive)

The only primitive operator is:
  move(r, o1, t1, o2, t2)
which corresponds to moving the top ring from tower t1 onto the top object of tower t2.

Key challenge: Moving a larger ring requires first clearing all smaller rings above it.
In classic Hanoi, the optimal cost to move a stack of k rings is 2^k - 1.

Heuristic strategy
------------------
We implement an admissible (lower-bound) heuristic that is tight for Hanoi:
1) Extract goal ring positions from model.goals (bitwise goal facts).
2) Infer ring size order (smallest -> largest) from static smallerThan facts.
3) For a given state, find the *largest* ring not currently at its goal position.
4) If that ring is at index i (0-based in smallest->largest list), then at least
   k=i+1 rings (that ring + all smaller ones) must be moved. Return 2^k - 1.

This is admissible because moving that largest misplaced ring requires relocating all
smaller rings at least as in the optimal subproblem.

Efficiency considerations
-------------------------
- Uses bitwise checks against fact.global_id (fast).
- Precomputes per-ring candidate on-facts for O(#facts for ring) position lookup.
- Precomputes ring ordering once in initialize().

Robustness
----------
- Handles cases where some goal facts are missing: returns 0 if no usable goal info.
- Falls back to a safe (very weak) heuristic if ring ordering cannot be inferred.

"""

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic19(Heuristic):
    """Admissible domain-dependent heuristic for the Towers (Hanoi) domain."""

    def __init__(self, towers_gpt52_heuristic_19=None, name: str = "TowersGPT52Heuristic19"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Filled in initialize/_preprocess
        self.model: Optional[Model] = None

        # ring -> target object in the goal (from goal +on[ring,target])
        self.goal_positions: Dict[str, str] = {}

        # rings sorted smallest -> largest
        self.rings_by_size: List[str] = []

        # ring -> list of (Fact, target_obj) for all possible on facts of that ring
        self.on_facts_by_ring: Dict[str, List[Tuple[object, str]]] = {}

        self.num_rings: int = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """Parse grounded fact name into (predicate, [args])."""
        name = name.strip()
        # Grounded format: +pred[a,b] or -pred[a,b]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                pred = name[:b]
                args_str = name[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
        # Optional HDDL-like: (pred a b)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            if inner:
                parts = inner.split()
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self) -> None:
        """Extract ring ordering, goal positions, and per-ring on-facts."""
        assert self.model is not None

        rings = set()
        smaller_edges: List[Tuple[str, str]] = []  # (smaller_ring, larger_obj)

        # 1) Extract goal positions from bitwise model.goals (goal state facts).
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2) Scan all grounded facts to build ordering and on-fact lookup tables.
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    rings.add(a)
                    smaller_edges.append((a, b))
                    if b.startswith("r"):
                        rings.add(b)

            elif pred == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Infer a total order smallest->largest.
        # We use: for rings, the one that is "smallerThan" many others is smaller.
        if smaller_edges:
            smaller_count = defaultdict(int)
            for s, larger in smaller_edges:
                if s in rings:
                    smaller_count[s] += 1
            # highest smaller_count => smallest
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: try to infer from goal stacking: if on(r_i, r_j) in goal, r_i < r_j
            is_smaller = defaultdict(set)
            for r, target in self.goal_positions.items():
                if target.startswith("r"):
                    is_smaller[r].add(target)
            # transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller.keys()):
                    grow = set()
                    for mid in is_smaller[r]:
                        grow |= is_smaller.get(mid, set())
                    if not grow.issubset(is_smaller[r]):
                        is_smaller[r] |= grow
                        changed = True
            score = {r: len(is_smaller.get(r, set())) for r in rings}
            self.rings_by_size = sorted(rings, key=lambda r: -score.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> Optional[str]:
        """Return the object this ring is currently on, using bitwise checks."""
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, target in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """Compute the 2^k-1 lower bound based on largest misplaced ring."""
        if not self.rings_by_size or not self.goal_positions:
            return 0

        # iterate from largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # if goal doesn't mention this ring, ignore it safely
                continue
            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                k = i + 1  # ring plus all smaller
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self) -> str:
        return f"TowersGPT52Heuristic19(rings={self.num_rings})"

    def __str__(self) -> str:
        return f"TowersGPT52Heuristic19 with {self.num_rings} rings"

    def __output__(self) -> str:
        return (
            f"TowersGPT52Heuristic19 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions (subset): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}