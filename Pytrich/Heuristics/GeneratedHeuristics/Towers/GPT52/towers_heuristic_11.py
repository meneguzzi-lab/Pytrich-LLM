"""\
Domain-dependent heuristic for the Towers (HTN Towers of Hanoi) domain.

Key domain insights leveraged
----------------------------
1) Only primitive action is `move`, which moves the current top ring of a tower
   onto the current top object of another tower, respecting `smallerThan`.
2) The training instances encode a classic Hanoi stack: r1 smallest ... rn largest.
3) The goal is a full stack on the target tower (t3 in training), meaning every
   ring must end up on its designated support.
4) In Towers of Hanoi, the optimal number of moves required to correctly place
   the smallest k rings (so that ring k is placeable) is at least 2^k - 1.

Heuristic design
----------------
We implement the classic *tight admissible* lower bound for Hanoi:
- Determine the largest ring that is not currently in its goal position.
- Let k be the number of rings from the smallest up to that ring.
- Return 2^k - 1.

Why this works (admissibility)
------------------------------
To move a misplaced ring of size k (k-th smallest), all k-1 smaller rings must be
moved out of the way at least once, and arranging them to allow that move requires
at least 2^(k-1) - 1 moves; then the ring move itself adds 1, plus additional
moves are needed afterward. Thus 2^k - 1 is a valid lower bound on the remaining
optimal cost.

Efficiency considerations
-------------------------
- All expensive work is done once in initialize(): parsing facts, building
  ring ordering, indexing `on` facts by ring, extracting goal positions.
- Per node, we only do O(#rings * avg_on_facts_per_ring) bit checks, with fast
  bitwise access: (state >> fact.global_id) & 1.

Pytrich integration details
---------------------------
- Goals are accessed via model.goals (bitwise) and decoded with
  model.state_explicit_repr(model.goals).
- Facts are parsed in grounded format: +predicate[arg1,arg2].

Class naming requirements
-------------------------
- Class name: TowersGPT52Heuristic11
- __init__ parameter name: towers_gpt52_heuristic_11 (all lower case)

"""

import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic11(Heuristic):
    """Tight admissible Hanoi heuristic: return 2^k - 1 for largest misplaced ring."""

    def __init__(self, towers_gpt52_heuristic_11=None, name="TowersGPT52Heuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Filled in during initialize()
        self.model: Model = None

        # ring -> target object in goal (from goal state's +on facts)
        self.goal_positions = {}

        # ring -> list of (Fact, target_obj) for all possible +on[ring,target]
        self.on_facts_by_ring = {}

        # Rings sorted smallest -> largest
        self.rings_by_size = []
        self.num_rings = 0

    # ---------- Parsing helpers ----------

    def _parse_fact_name(self, name: str):
        """Parse "+pred[a,b]" / "-pred[a,b]" or "(pred a b)" into (pred, [args])."""
        name = name.strip()

        # Grounded format: +predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
            return None, []

        # HDDL-like format: (predicate arg1 arg2)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            if not inner:
                return None, []
            parts = inner.split()
            return parts[0], parts[1:]

        return None, []

    # ---------- Preprocessing ----------

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess_static()
        self.preprocessing_time = time.time() - start

        h0 = self._h_from_state(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess_static(self):
        """Extract ring set, ordering, goal positions, and on-fact indices."""
        rings = set()
        smaller_pairs = []  # (smaller, larger)

        # 1) Extract goal positions from *goal state bitset* (not from goal_on)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = tgt
                        rings.add(r)
                        if tgt.startswith("r"):
                            rings.add(tgt)

        # 2) Index all facts relevant to ring order and position
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "smallerThan" and len(args) >= 2:
                a, b = args[0], args[1]
                if a.startswith("r"):
                    rings.add(a)
                    smaller_pairs.append((a, b))
                if b.startswith("r"):
                    rings.add(b)

            elif pred == "on" and len(args) >= 2:
                r, tgt = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, tgt))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine size ordering smallest -> largest.
        # Preferred: use smallerThan transitive signal via counts.
        if smaller_pairs:
            # Count how many things each ring is smaller than (within ring set).
            smaller_count = defaultdict(int)
            for a, b in smaller_pairs:
                if a in rings and b in rings:
                    smaller_count[a] += 1
            # Smaller rings tend to be smallerThan many other rings.
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer from goal stack relations: if on[r_i, r_j] in goal, then r_i < r_j.
            is_smaller = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if tgt.startswith("r"):
                    is_smaller[r].add(tgt)

            changed = True
            while changed:
                changed = False
                for r in list(is_smaller.keys()):
                    closure = set(is_smaller[r])
                    for mid in list(closure):
                        for bigger in is_smaller.get(mid, set()):
                            if bigger not in is_smaller[r]:
                                is_smaller[r].add(bigger)
                                changed = True

            smaller_count = {r: len(is_smaller.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    # ---------- Fast state queries ----------

    def _current_support(self, state: int, ring: str):
        """Return the object the ring is currently on, by checking indexed +on facts."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, tgt in lst:
            if (state >> fact.global_id) & 1:
                return tgt
        return None

    # ---------- Heuristic computation ----------

    def _h_from_state(self, state: int) -> int:
        if not self.rings_by_size:
            return 0

        # Iterate from largest -> smallest to find largest misplaced ring.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_tgt = self.goal_positions.get(ring)
            if goal_tgt is None:
                # If no explicit goal position, ignore the ring.
                continue
            cur_tgt = self._current_support(state, ring)
            if cur_tgt != goal_tgt:
                k = i + 1  # number of rings from smallest..this ring
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._h_from_state(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic11(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic11 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings: {self.num_rings}\n"
            f"\tRings smallest->largest: {self.rings_by_size}\n"
            f"\tGoal positions (from goal on-facts): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}