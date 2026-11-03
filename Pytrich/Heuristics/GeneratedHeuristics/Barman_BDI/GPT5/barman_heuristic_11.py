from typing import Dict, Tuple, List, Optional, Union, DefaultDict
from collections import defaultdict

try:
    # Standard import path in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimal base to allow static checking/running outside the framework
    class Heuristic:
        def __init__(self, model=None):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
            return 0


class BarmanGPT5Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design rationale (lower-bound, domain-specific, hierarchical-aware):
    - Primary top-level tasks are AchieveContainsShotCocktail(shot, cocktail).
    - Minimal primitive actions to produce one new cocktail from a clean-empty shaker:
        * 2 x fill-shot (one per ingredient)
        * 2 x pour-shot-to-... (first into clean-empty shaker, then into used/unshaked shaker)
        * 1 x shake
        * 1 x pour-shaker-to-shot
      -> Minimal 6 actions (ignoring grasp/hand exchanges) as admissible lower bound.
    - Reuse across multiple shots of the same cocktail: Once a cocktail is in the shaker (and shaked), you can pour to m shots with m pours. We group tasks by cocktail and compute: make_cost(c) + number_of_unsatisfied_shots(c). This avoids overestimating when multiple shots of the same cocktail are needed.
    - Make cost refinement using current shaker contents/state for tighter lower bound:
        * If any shaker already contains cocktail c and is shaked: make_cost(c) = 0
        * Else if any shaker is unshaked and already contains both required ingredients for c: make_cost(c) = 1 (just shake)
        * Else if any shaker is unshaked and already contains one required ingredient for c: make_cost(c) = 3 (fill + pour + shake)
        * Else: make_cost(c) = 5 (2 fills + 2 pours + 1 shake, excluding final pour)
      We take the minimum over available shakers (usually only one), ignoring resets/cleaning, which keeps it admissible.
    - Other abstract tasks (when the TN is partially decomposed) are given conservative lower bounds:
        * AchieveContainsShakerIngredient(s, i): 0 if already true; else 2 (fill + pour) as a domain LB.
        * AchieveContainsShotIngredient(shot, i): 0 if already true; else 1 (fill-shot) LB.
        * AchieveContainsShakerCocktail(s, c): as make_cost(c) but for the specific shaker s (no final pour).
        * DoPourShakerToShot: 1 if not already satisfied; else 0.
        * AchieveCleanShot/AchieveCleanShaker: 0 if already clean; else 1.
        * AchieveHolding: 0 if already holding; else 1.
        * AchieveHandEmpty: 0 if that hand is empty; else 1.
        * AchieveOnTable: 0 if ontable; else 1.
    - Primitive operators already present in the TN are counted by summing their (declared) costs. This is a safe LB on the remaining cost since they must be executed.

    This heuristic follows the HTN structure (counts cost-to-refine tasks) and uses domain knowledge while keeping computations lightweight (bitwise checks, pre-indexed facts).
    """

    def __init__(self, model, barman_gpt5_heuristic_11: bool = True):
        super().__init__(model)
        self.barman_gpt5_heuristic_11 = barman_gpt5_heuristic_11

        # Index all positive facts by predicate and arguments for O(1) bitwise checks
        self.fact_index: Dict[str, Dict[Tuple[str, ...], object]] = defaultdict(dict)
        self._build_fact_index()

        # Cocktail -> (ing1, ing2)
        self.cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._build_cocktail_parts()

        # Identify shakers from shaker-related facts
        self.shakers: List[str] = self._collect_shakers()

        # Cache of all cocktails/ingredients known (from cocktail parts)
        self.all_cocktails = set(self.cocktail_parts.keys())
        self.all_ingredients = set()
        for c, (p1, p2) in self.cocktail_parts.items():
            if p1:
                self.all_ingredients.add(p1)
            if p2:
                self.all_ingredients.add(p2)

    # ---------------------- Core Heuristic API ----------------------
    def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
        if not task_network:
            return 0

        h_val = 0

        # Group AchieveContainsShotCocktail tasks by cocktail (ignoring those already satisfied)
        cocktail_to_shots: DefaultDict[str, List[str]] = defaultdict(list)

        for node in task_network:
            kind = self._node_kind(node)
            if kind == 'operator':
                # Sum remaining primitive costs directly (strong LB)
                try:
                    cost = getattr(node, 'cost', 1)
                except Exception:
                    cost = 1
                h_val += int(cost)
                continue

            if kind != 'task':
                # Unknown object in TN; be safe and ignore (LB 0)
                continue

            name = getattr(node, 'name', '')
            head, args = self._parse_atom(name)

            # Handle key tasks with domain-specific LBs
            if head == 'AchieveContainsShotCocktail' and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                # If already satisfied, contributes 0
                if self._is_true_fact(state_bitwise, 'contains', (shot, cocktail)):
                    continue
                cocktail_to_shots[cocktail].append(shot)

            elif head == 'AchieveContainsShakerCocktail' and len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                h_val += self._make_cost_for_cocktail_with_shaker(state_bitwise, cocktail, shaker)

            elif head == 'AchieveContainsShakerIngredient' and len(args) >= 2:
                shaker, ingredient = args[0], args[1]
                if not self._is_true_fact(state_bitwise, 'contains', (shaker, ingredient)):
                    # At least: fill-shot + pour-to-... = 2
                    h_val += 2

            elif head == 'DoPourShakerToShot' and len(args) >= 3:
                # Parameters: shaker, shot, cocktail (per domain file)
                # Minimal LB: 1 pour if not already in the shot
                shot, cocktail = args[1], args[2]
                if not self._is_true_fact(state_bitwise, 'contains', (shot, cocktail)):
                    h_val += 1

            elif head == 'AchieveContainsShotIngredient' and len(args) >= 2:
                shot, ingredient = args[0], args[1]
                if not self._is_true_fact(state_bitwise, 'contains', (shot, ingredient)):
                    h_val += 1  # at least one fill-shot

            elif head == 'AchieveCleanShot' and len(args) >= 1:
                shot = args[0]
                if not self._is_true_fact(state_bitwise, 'clean', (shot,)):
                    h_val += 1  # at least one clean-shot action

            elif head == 'AchieveCleanShaker' and len(args) >= 1:
                shaker = args[0]
                if not self._is_true_fact(state_bitwise, 'clean', (shaker,)):
                    h_val += 1

            elif head == 'AchieveHolding' and len(args) >= 2:
                hand, container = args[0], args[1]
                if not self._is_true_fact(state_bitwise, 'holding', (hand, container)):
                    h_val += 1  # at least a grasp

            elif head == 'AchieveHandEmpty' and len(args) >= 1:
                hand = args[0]
                if not self._is_true_fact(state_bitwise, 'handEmpty', (hand,)):
                    h_val += 1  # at least a drop

            elif head == 'AchieveOnTable' and len(args) >= 1:
                container = args[0]
                if not self._is_true_fact(state_bitwise, 'ontable', (container,)):
                    h_val += 1  # at least one drop

            # Unrecognized tasks contribute 0 as a safe LB

        # Now add grouped lower bounds for remaining AchieveContainsShotCocktail tasks
        for cocktail, shots in cocktail_to_shots.items():
            # Shots we need to fill with that cocktail (we already filtered those already satisfied)
            pours_needed = len(shots)
            make_cost = self._make_cost_for_cocktail_best_shaker(state_bitwise, cocktail)
            h_val += make_cost + pours_needed

        return int(h_val)

    # ---------------------- Domain-specific helpers ----------------------
    def _make_cost_for_cocktail_best_shaker(self, state_bitwise: int, cocktail: str) -> int:
        """
        Returns a lower bound on the number of primitive actions to get some shaker to contain 'cocktail'.
        This version chooses the best (cheapest) shaker, which is admissible.
        """
        if not self.shakers:
            return 5  # very conservative default
        best = 10**9
        for s in self.shakers:
            cand = self._make_cost_for_cocktail_with_shaker(state_bitwise, cocktail, s)
            if cand < best:
                best = cand
                if best == 0:
                    break
        return best if best < 10**9 else 5

    def _make_cost_for_cocktail_with_shaker(self, state_bitwise: int, cocktail: str, shaker: str) -> int:
        """
        Lower bound for making 'cocktail' specifically in 'shaker'.
        Cases:
        - If shaker already contains the cocktail and is shaked: 0
        - Else if shaker is unshaked and already has both required ingredients: 1 (shake)
        - Else if shaker is unshaked and has one of the required ingredients: 3 (fill + pour + shake)
        - Else: 5 (two fills + two pours + shake)
        """
        parts = self.cocktail_parts.get(cocktail, (None, None))
        ing1, ing2 = parts[0], parts[1]

        # Check if shaker already holds the final cocktail and is shaked
        if self._is_true_fact(state_bitwise, 'contains', (shaker, cocktail)) and \
           self._is_true_fact(state_bitwise, 'shaked', (shaker,)):
            return 0

        # If we don't know the parts, use conservative LB
        if not ing1 or not ing2:
            return 5

        unshaked = self._is_true_fact(state_bitwise, 'unshaked', (shaker,))
        has1 = self._is_true_fact(state_bitwise, 'contains', (shaker, ing1))
        has2 = self._is_true_fact(state_bitwise, 'contains', (shaker, ing2))

        if unshaked and has1 and has2:
            return 1  # just shake
        if unshaked and (has1 or has2):
            return 3  # add one ingredient + shake
        return 5      # add both ingredients + shake

    # ---------------------- Fact indexing and checks ----------------------
    def _build_fact_index(self):
        """
        Build index: predicate -> { (args...): Fact }
        Only index positive facts (names starting with '+') for fast bitwise truth checks.
        """
        self.fact_index.clear()
        for f in getattr(self.model, 'facts', []):
            name = getattr(f, 'name', '')
            if not name:
                continue
            sign = name[0] if name and name[0] in '+-' else '+'
            if sign != '+':
                continue
            pred, args = self._parse_atom(name)
            if not pred:
                continue
            key = tuple(args)
            self.fact_index[pred][key] = f

    def _build_cocktail_parts(self):
        """
        Read cocktailPart1/2 facts to build cocktail -> (ing1, ing2) mapping.
        """
        # Initialize with empty parts
        for (pred_name, idx) in [('cocktailPart1', 1), ('cocktailPart2', 2)]:
            if pred_name in self.fact_index:
                for args, f in self.fact_index[pred_name].items():
                    if len(args) != 2:
                        continue
                    cocktail, ingr = args[0], args[1]
                    if cocktail not in self.cocktail_parts:
                        self.cocktail_parts[cocktail] = (None, None)
                    p1, p2 = self.cocktail_parts[cocktail]
                    if pred_name == 'cocktailPart1':
                        self.cocktail_parts[cocktail] = (ingr, p2)
                    else:
                        self.cocktail_parts[cocktail] = (p1, ingr)

    def _collect_shakers(self) -> List[str]:
        shakers = set()
        for pred in ('shakerLevel', 'shakerEmptyLevel', 'shaked', 'unshaked'):
            if pred in self.fact_index:
                for args in self.fact_index[pred].keys():
                    if args:
                        shakers.add(args[0])
        return list(shakers)

    def _is_true_fact(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        facts_with_pred = self.fact_index.get(predicate)
        if not facts_with_pred:
            return False
        f = facts_with_pred.get(tuple(args))
        if f is None:
            return False
        # Fast bitwise membership check
        return bool((state_bitwise >> f.global_id) & 1)

    # ---------------------- Generic parsing and TN utilities ----------------------
    def _parse_atom(self, name: str) -> Tuple[str, List[str]]:
        """
        Generic parser for names like:
          '+pred[arg1,arg2]', '-pred[arg1,arg2]', 'pred[arg1,arg2]', 'TaskName(arg1,arg2)'
        Returns (head, [args...]). Strips leading +/- if present.
        """
        if not name:
            return '', []
        s = name.strip()
        if s and s[0] in '+-':
            s = s[1:]
        # Try parentheses first, then brackets
        for oc, cc in (("(", ")"), ("[", "]")):
            i = s.find(oc)
            j = s.rfind(cc)
            if i >= 0 and j >= 0 and j > i:
                head = s[:i].strip()
                args_str = s[i+1:j].strip()
                args = [a.strip() for a in args_str.split(',')] if args_str else []
                return head, args
        # No args
        return s, []

    def _node_kind(self, node: object) -> str:
        """
        Try to identify if 'node' is an Operator or an AbstractTask.
        We avoid importing classes directly and use attribute-based detection.
        Returns one of: 'operator', 'task', 'unknown'.
        """
        if node is None:
            return 'unknown'
        # Operators in the provided class have apply(), cost, pos_precons, etc.
        if hasattr(node, 'apply') and hasattr(node, 'cost') and hasattr(node, 'get_precons'):
            return 'operator'
        # AbstractTask has 'decompositions' list and a 'name'
        if hasattr(node, 'decompositions') and hasattr(node, 'name'):
            return 'task'
        return 'unknown'


# Factory method (optional) if the framework expects callable to build the heuristic
def build(barman_gpt5_heuristic_11=True):
    def _ctor(model):
        return BarmanGPT5Heuristic11(model, barman_gpt5_heuristic_11=barman_gpt5_heuristic_11)
    return _ctor
