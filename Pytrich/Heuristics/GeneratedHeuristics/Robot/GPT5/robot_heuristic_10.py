from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Core idea (admissible lower bound):
    - Every object not yet at its goal room must be put down at least once -> contributes 1.
    - If that object is not currently held, it must be picked up first -> contributes another 1.
    - We deliberately ignore movement and door opening costs in the estimate to preserve admissibility.

    Hence, for each goal fact in[o,g]:
      - If in[o,g] is already true in the state: contributes 0.
      - Else if holding[o] is true: contributes 1 (only putdown remains; moves/opens ignored for LB).
      - Else: contributes 2 (pickup + putdown; moves/opens ignored for LB).

    This yields an admissible and fast-to-compute heuristic: h = 2 * (#unsatisfied goals) - (#of those currently held).

    Rationale and domain insights:
    - In this domain, objects only change rooms via the sequence: pickup(o, r_obj), move/open as needed, putdown(o, r_goal).
    - Ignoring movement and opening (which are required in practice) produces a safe lower bound across all instances, including the
      largest ones with many rooms and closed doors, while still being informative because it counts compelled primitive actions
      due to carrying capacity (single arm) and object transfer semantics.
    - Hierarchical structure is respected: regardless of decomposition choices (achieve-goals, release, pickup_abstract, etc.),
      the primitive actions needed to relocate an object are at least one pickup and one putdown (except when already held or already placed),
      so the count above is a decomposition-independent lower bound on the remaining primitive cost.

    Implementation details:
    - We precompute goal in-fact identifiers and holding-fact identifiers per object for O(1) bitwise checks at runtime.
    - We use bitwise membership tests on the state's integer encoding for speed (per framework guidelines).

    Notes:
    - Parameter name robot_gpt5_heuristic_10 is included in __init__ per the specification; it's unused but kept for compatibility
      with potential configuration loaders.
    """

    def __init__(self, model, robot_gpt5_heuristic_10: Optional[bool] = None, **kwargs):
        try:
            super().__init__(model)
        except Exception:
            self.model = model

        # Internal caches
        self._facts_by_name: Dict[str, int] = {}              # '+pred[a,b]' -> fact.global_id
        self._holding_fact_id: Dict[str, int] = {}            # obj -> fact id of '+holding[obj]'
        self._in_fact_id: Dict[Tuple[str, str], int] = {}     # (obj, room) -> fact id of '+in[obj,room]'
        self._goal_in_facts: List[Tuple[str, str, int]] = []  # list of (obj, room, fact_id) for goal facts

        self._index_facts()
        self._extract_goals()

        # Keep the provided parameter to satisfy loader requirements (even if unused directly)
        self.robot_gpt5_heuristic_10 = robot_gpt5_heuristic_10

    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.evaluate(state_bitwise, task_network)

    def evaluate(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        # Count unsatisfied goals (in[o,g] not true) and among them, how many are currently held
        unsatisfied = 0
        held_among_unsatisfied = 0

        for obj, _room, in_fact_id in self._goal_in_facts:
            if not self._bit_is_true(state_bitwise, in_fact_id):
                unsatisfied += 1
                hold_id = self._holding_fact_id.get(obj, -1)
                if hold_id >= 0 and self._bit_is_true(state_bitwise, hold_id):
                    held_among_unsatisfied += 1

        # Lower bound on remaining primitive actions
        # Each unsatisfied goal needs >=1 putdown; if not held it additionally needs >=1 pickup
        h_val = 2 * unsatisfied - held_among_unsatisfied
        if h_val < 0:
            h_val = 0
        return h_val

    # ------------------------- Internal helpers -------------------------

    def _index_facts(self):
        # Build mappings from fact names to ids, and index holding/in facts for fast lookup
        for fact in self.model.facts:
            name = fact.name
            self._facts_by_name[name] = fact.global_id
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            if pred == 'holding' and len(args) == 1:
                self._holding_fact_id[args[0]] = fact.global_id
            elif pred == 'in' and len(args) == 2:
                self._in_fact_id[(args[0], args[1])] = fact.global_id

    def _extract_goals(self):
        # Extract goal facts as a static list of (obj, room, fact_id) for in-predicate goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'in' and len(args) == 2:
                obj, room = args
                fid = self._in_fact_id.get((obj, room), self._facts_by_name.get(gname, -1))
                if fid >= 0:
                    self._goal_in_facts.append((obj, room, fid))

    @staticmethod
    def _bit_is_true(state_bitwise: int, fact_id: int) -> bool:
        if fact_id < 0:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name in the Pytrich explicit string representation:
          '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'
        Returns (predicate, [args]) or (None, []).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
            bracket_idx = s.find('[')
            if bracket_idx > 0 and s.endswith(']'):
                predicate = s[:bracket_idx]
                args_str = s[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []


# Backwards-compatible alias (if framework expects a function to build the heuristic)
def build(robot_gpt5_heuristic_10=None):
    def _builder(model):
        return RobotGPT5Heuristic10(model, robot_gpt5_heuristic_10=robot_gpt5_heuristic_10)
    return _builder
