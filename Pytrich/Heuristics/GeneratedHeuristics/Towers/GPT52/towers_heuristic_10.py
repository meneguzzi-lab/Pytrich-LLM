"""\
Domain-dependent heuristic for the HDDL Towers (Towers of Hanoi) domain.

Heuristic idea (domain insights)
-------------------------------
This domain encodes classic Towers of Hanoi with:
- Primitive action: move(r, o1, t1, o2, t2)
- HTN tasks: shiftTower -> selectDirection -> rotateTower -> exchange -> move_abstract -> move

Key challenge:
- Even though the HTN controls legality, the true cost-to-go is dominated by the
  well-known exponential structure: moving the largest misplaced ring requires
  repositioning all smaller rings.

Heuristic design
----------------
We combine TWO admissible lower bounds and take their maximum:

(1) Classic Hanoi lower bound (tight and typically optimal):
    Find the largest ring that is not currently on its goal support.
    If that ring is the k-th smallest (i.e., there are k rings up to it), then
    at least 2^k - 1 moves are necessary.

(2) Goal-facts unsatisfied count (very cheap, also admissible):
    Count how many goal on-facts are currently false. Each primitive move can
    make at most one 'on[ring, obj]' goal fact become true (and may break others),
    so this is a lower bound.

Taking max keeps admissibility while providing stronger guidance in some
intermediate states.

Implementation notes for Pytrich
--------------------------------
- Goals are stored as a bitwise integer in model.goals.
- State checking MUST be bitwise: (state >> fact.global_id) & 1
- We preprocess:
  * goal_on relations by reading model.goals explicit representation and parsing '+on[...]'
  * mapping ring -> all grounded 'on' facts for fast current-position lookup
  * ring size order using smallerThan facts

Class name and init parameter naming follow the task requirements:
- Class: TowersGPT52Heuristic10
- __init__ parameter: towers_gpt52_heuristic_10
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic10(Heuristic):
    def __init__(self, towers_gpt52_heuristic_10=None, name="TowersGPT52Heuristic10"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Set in initialize
        self.model: Model | None = None

        # Preprocessed knowledge
        self.goal_positions: dict[str, str] = {}          # ring -> goal support (tower or ring)
        self.goal_on_facts: list = []                     # list of Fact objects that are goal +on[...]
        self.on_facts_by_ring: dict[str, list] = {}       # ring -> list[(Fact, target_obj)]

        self.rings: set[str] = set()
        self.rings_by_size: list[str] = []               # smallest -> largest
        self.ring_index: dict[str, int] = {}             # ring -> index in rings_by_size

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._heuristic(initial_node.state)
        return super().initialize(model, h0)

    # ---------- Parsing utilities ----------
    def _parse_fact_name(self, name: str):
        """Parse grounded fact names of the form '+pred[a,b]' or '-pred[a,b]'."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                pred = name[:b]
                args_str = name[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
        return None, []

    # ---------- Preprocessing ----------
    def _preprocess(self):
        assert self.model is not None

        # 1) Extract goal on-facts from model.goals (bitwise)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals) if getattr(self.model, "goals", 0) else []
        goal_on_pairs = []  # list[(ring, target)]
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                ring, target = args
                if ring.startswith("r"):
                    self.goal_positions[ring] = target
                    goal_on_pairs.append((ring, target))

        # 2) Scan all grounded facts once: build on-fact index and smallerThan ordering data
        smaller_edges = []  # (smaller, larger)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "on" and len(args) == 2:
                ring, target = args
                if ring.startswith("r"):
                    self.rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

            elif pred == "smallerThan" and len(args) == 2:
                a, b = args
                # In this domain, rings are named r1..rn in instances, but be robust.
                if a.startswith("r"):
                    self.rings.add(a)
                if b.startswith("r"):
                    self.rings.add(b)
                if a.startswith("r") and b:
                    smaller_edges.append((a, b))

        # 3) Map goal on-facts to Fact objects (for fast unsatisfied-goal counting)
        #    We avoid constructing names and searching in a dict each call.
        if goal_on_pairs:
            # Build quick lookup from (pred,arg1,arg2) to Fact
            fact_lookup = {}
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if pred and len(args) == 2:
                    fact_lookup[(pred, args[0], args[1])] = fact

            for ring, target in goal_on_pairs:
                f = fact_lookup.get(("on", ring, target))
                if f is not None:
                    self.goal_on_facts.append(f)

        # 4) Derive ring size order (smallest -> largest)
        #    Use counts of "is smaller than" relations: smallest has the most things larger than it.
        if self.rings:
            smaller_count = defaultdict(int)
            for a, b in smaller_edges:
                if a in self.rings:
                    # count edges a < b (b can be ring or tower)
                    smaller_count[a] += 1
            self.rings_by_size = sorted(self.rings, key=lambda r: -smaller_count.get(r, 0))
            self.ring_index = {r: i for i, r in enumerate(self.rings_by_size)}

    # ---------- Fast state queries ----------
    def _get_current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object (ring or tower) the given ring is currently on."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ---------- Heuristic computation ----------
    def _h_hanoi(self, state_bitwise: int) -> int:
        """Classic admissible Hanoi lower bound: 2^k - 1 for largest misplaced ring."""
        if not self.rings_by_size or not self.goal_positions:
            return 0

        # iterate largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                continue
            cur_support = self._get_current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1  # number of rings up to and including this one
                return (1 << k) - 1
        return 0

    def _h_unsat_goals(self, state_bitwise: int) -> int:
        """Admissible bound: number of unsatisfied goal on-facts."""
        if not self.goal_on_facts:
            return 0
        unsat = 0
        for f in self.goal_on_facts:
            if ((state_bitwise >> f.global_id) & 1) == 0:
                unsat += 1
        return unsat

    def _heuristic(self, state_bitwise: int) -> int:
        return max(self._h_hanoi(state_bitwise), self._h_unsat_goals(state_bitwise))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic10(rings={len(self.rings_by_size)})"

    def __str__(self):
        return "TowersGPT52Heuristic10 (Hanoi max unsat-goals)"

    def __output__(self):
        return (
            "TowersGPT52Heuristic10 Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings: {len(self.rings_by_size)}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal positions known for: {len(self.goal_positions)} rings\n"
            f"\tGoal on-facts tracked: {len(self.goal_on_facts)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}