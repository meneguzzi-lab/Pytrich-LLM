from typing import Dict, Tuple, List, Set, Optional, Union

# Fallback Heuristic base if the framework import path differs
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    class Heuristic:
        def __init__(self, model):
            self.model = model
        def __call__(self, state, task_network=None):
            return 0


class BarmanGPT5Heuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design overview and rationale:
    - This heuristic focuses on the hierarchical structure of the Barman domain, where
      high-level tasks (e.g., AchieveContainsShotCocktail) decompose into sequences of
      sub-tasks that ultimately require a small set of primitive actions: grasp, fill-shot,
      pour-shot-to-...-shaker, shake, pour-shaker-to-shot, empty/clean, etc.
    - We assign a tight, admissible lower bound to each abstract task by leveraging
      domain knowledge about the minimal number of primitive actions required in the
      best case (ignoring negative interactions and resource/contention, which keeps
      the estimate a lower bound under delete-relaxation style reasoning).
    - We also use current state information (bitwise checks on facts) to reduce costs
      further if partial progress has already been made (e.g., shaker already holds an
      ingredient or the target shot is already clean/empty).
    - To avoid double-counting, we detect AchieveHolding tasks in the current task network
      and refrain from charging the grasp/hold cost inside related tasks (e.g., the shaker
      that is going to be held anyway later for pouring).

    Key lower-bound building blocks used (typical minimal counts):
    - One ingredient added to the shaker via a fresh clean shot:
        grasp(shot) + fill-shot + pour-shot-to-(clean/used)-shaker  => 3 actions
    - Shake after both ingredients are in the shaker: 1 action (shake)
      (requires holding shaker and an empty other hand; we only charge holding once as LB)
    - Pour cocktail from shaker to a target shot: 1 action (pour-shaker-to-shot)
    - Grasp/holding cost (when needed): 1 if some hand is free and the container is on table,
      else at least 2 (drop + grasp). If already holding, 0.
    - Cleaning a shot (for AchieveCleanShot): LB is 1 when not already clean (because clean-shot
      is a single action in the best case, even if more steps may actually be needed), else 0.

    Main abstract task LB formulas (context-sensitive, based on state):
    - AchieveContainsShotCocktail(shot, cocktail): if already contains => 0
      Otherwise, pick the best shaker w.r.t. progress for that cocktail and return the minimum of:
        * If shaker already contains the cocktail (and is shaked): clean(shot)_lb + hold_cost(shaker) + 1 (pour)
        * If shaker has both ingredients: clean(shot)_lb + hold_cost(shaker) + 1 (shake) + 1 (pour)
        * If shaker has exactly one ingredient: clean(shot)_lb + 3 (add remaining) + hold_cost + 1 (shake) + 1 (pour)
        * Else (no ingredients): clean(shot)_lb + 6 (add both) + hold_cost + 1 (shake) + 1 (pour)
      To avoid double counting when the TN already includes AchieveHolding for that shaker, hold_cost is set to 0.

    - AchieveContainsShakerCocktail(shaker, cocktail): if already contains => 0
      Else (looking at progress wrt. cocktail's two ingredients):
        * both ingredients in shaker: hold_cost(shaker) + 1 (shake)
        * one ingredient in shaker: 3 (add remaining) + hold_cost + 1 (shake)
        * none: 6 (add both) + hold_cost + 1 (shake)
      As above, hold_cost is waived if AchieveHolding for that very shaker is already in the TN.

    - AchieveContainsShakerIngredient(shaker, ingredient): 0 if already contains, else 3 (best-case cycle)
    - AchieveContainsShotIngredient(shot, ingredient): 0 if already contains, else
        clean(shot)_lb + hold_cost(shot, specific hand not enforced) + 1 (fill-shot)
    - AchieveCleanShot(shot): 0 if clean, else 1
    - AchieveCleanShaker(shaker): 0 if clean, else 1
    - AchieveHolding(hand, container): exact-hand aware LB: 0 if already holding; else 1 if that hand is empty and
      container on table; else 2
    - AchieveHandEmpty(hand): 0 if empty, else 1 (drop)
    - AchieveOnTable(container): 0 if ontable, else 1 (drop)
    - DoPourShakerToShot(shaker, shot, cocktail): 0 if shot already has cocktail, else 1 (best-case pour)

    Primitive operators in the current TN contribute at least their action cost (we use operator.cost if available,
    else 1). This is safe because primitive actions explicitly in the TN must be executed.

    Efficiency:
    - Preprocess once: build a fast index over grounded facts and extract cocktail parts, hands, containers, shakers.
    - Per-evaluation: use bitwise fact checks only (no slow state-to-list conversions).
    - Light parsing for task/operator names in bracket notation.

    Notes:
    - This heuristic is consistent and admissible by construction (delete-relaxation style counting), and it avoids
      double counting common holds when AchieveHolding for a container is already in the task network.
    - It scales well from small to large Barman instances due to constant-time bit checks and simple per-task formulas.
    """

    def __init__(self, model, barman_gpt5_heuristic_10: bool = False):
        super().__init__(model)
        self.model = model
        self.barman_gpt5_heuristic_10 = barman_gpt5_heuristic_10

        # Preprocessed structures
        self.fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self.cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self.shakers: Set[str] = set()
        self.containers: Set[str] = set()
        self.hands: Set[str] = set()

        # Fast-access sets of keys for scanning
        self._holding_keys: List[Tuple[str, str, int]] = []  # (hand, container, fact_id)
        self._ontable_keys: Dict[str, int] = {}  # container -> fact_id
        self._handempty_keys: Dict[str, int] = {}  # hand -> fact_id

        self._preprocess()

    # ---------- Public API ----------
    def evaluate(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        # Support empty TN heuristics (if goals are only tasks, not state)
        if task_network is None:
            task_network = []

        # Build current-hand status caches from the bitwise state
        hands_empty, holding_by_hand, holding_by_container, on_table = self._compute_hand_and_table_status(state_bitwise)

        # Pre-scan TN to detect explicit AchieveHolding targets to avoid double-counting holds
        hold_targets = self._extract_hold_targets_from_tn(task_network)

        h_val = 0
        for item in task_network:
            # Primitive operator: safe to add at least its cost (mandatory execution)
            if self._is_operator(item):
                op_cost = getattr(item, 'cost', 1)
                if not isinstance(op_cost, int) or op_cost <= 0:
                    op_cost = 1
                h_val += op_cost
                continue

            # Abstract task: use domain-dependent LB formulas
            name = getattr(item, 'name', '')
            tpred, targs = self._parse_name_brackets(name)
            if not tpred:
                # Unknown/unsupported task naming -> no guidance (admissible)
                continue

            # Route to specific estimators
            if tpred == 'AchieveContainsShotCocktail' and len(targs) == 2:
                shot, cocktail = targs
                h_val += self._estimate_AchieveContainsShotCocktail(state_bitwise, shot, cocktail, hands_empty,
                                                                    holding_by_container, on_table, hold_targets)
            elif tpred == 'AchieveContainsShakerCocktail' and len(targs) == 2:
                shaker, cocktail = targs
                h_val += self._estimate_AchieveContainsShakerCocktail(state_bitwise, shaker, cocktail, hands_empty,
                                                                       holding_by_container, on_table, hold_targets)
            elif tpred == 'AchieveContainsShakerIngredient' and len(targs) == 2:
                shaker, ingredient = targs
                h_val += self._estimate_AchieveContainsShakerIngredient(state_bitwise, shaker, ingredient)
            elif tpred == 'AchieveContainsShotIngredient' and len(targs) == 2:
                shot, ingredient = targs
                h_val += self._estimate_AchieveContainsShotIngredient(state_bitwise, shot, ingredient, hands_empty,
                                                                      holding_by_container, on_table)
            elif tpred == 'AchieveCleanShot' and len(targs) == 1:
                shot = targs[0]
                h_val += self._clean_shot_lb(state_bitwise, shot)
            elif tpred == 'AchieveCleanShaker' and len(targs) == 1:
                shaker = targs[0]
                if not self._is_true(state_bitwise, 'clean', (shaker,)):
                    h_val += 1
            elif tpred == 'AchieveHolding' and len(targs) == 2:
                hand, container = targs
                h_val += self._hold_cost_exact_hand(state_bitwise, hand, container, hands_empty, holding_by_hand, on_table)
            elif tpred == 'AchieveHandEmpty' and len(targs) == 1:
                hand = targs[0]
                if not self._is_true(state_bitwise, 'handEmpty', (hand,)):
                    h_val += 1
            elif tpred == 'AchieveOnTable' and len(targs) == 1:
                container = targs[0]
                if not self._is_true(state_bitwise, 'ontable', (container,)):
                    h_val += 1
            elif tpred == 'DoPourShakerToShot' and len(targs) == 3:
                shaker, shot, cocktail = targs
                # Minimal: if not already in shot, at least 1 action (pour)
                if not self._is_true(state_bitwise, 'contains', (shot, cocktail)):
                    h_val += 1
            else:
                # Unhandled abstract task -> no contribution (keeps admissible)
                pass

        return int(h_val)

    # Provide common aliases used by different planners
    def h(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        return self.evaluate(state_bitwise, task_network)

    def __call__(self, state_or_node, task_network: Optional[List[Union[object, object]]] = None) -> int:
        # Accept either a node object (with .state and .task_network) or (state, tn)
        if task_network is None and hasattr(state_or_node, 'state') and hasattr(state_or_node, 'task_network'):
            return self.evaluate(state_or_node.state, state_or_node.task_network)
        return self.evaluate(state_or_node, task_network)

    # ---------- Internal helpers ----------
    def _preprocess(self):
        # Build fact index and collect domain objects
        # Map: (predicate, args_tuple) -> fact.global_id
        facts = getattr(self.model, 'facts', [])
        for fact in facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            key = (pred, tuple(args))
            self.fact_by_key[key] = fact.global_id

            # Collect hands, containers, shakers from observed predicates
            if pred == 'handEmpty' and len(args) == 1:
                self.hands.add(args[0])
            if pred in ('ontable', 'clean', 'empty') and len(args) == 1:
                self.containers.add(args[0])
            if pred in ('shakerLevel', 'shakerEmptyLevel') and len(args) == 2:
                self.shakers.add(args[0])

            # Pre-cache some fact ids for faster scans per-evaluation
            if pred == 'holding' and len(args) == 2:
                hand, cont = args
                self._holding_keys.append((hand, cont, fact.global_id))
            if pred == 'ontable' and len(args) == 1:
                self._ontable_keys[args[0]] = fact.global_id
            if pred == 'handEmpty' and len(args) == 1:
                self._handempty_keys[args[0]] = fact.global_id

            # Cocktail parts
            if pred == 'cocktailPart1' and len(args) == 2:
                c, ing1 = args
                c2 = self.cocktail_parts.get(c, (None, None))[1]
                self.cocktail_parts[c] = (ing1, c2)
            if pred == 'cocktailPart2' and len(args) == 2:
                c, ing2 = args
                c1 = self.cocktail_parts.get(c, (None, None))[0]
                self.cocktail_parts[c] = (c1, ing2)

        # Ensure containers set contains all that may appear in holding pairs
        for _, cont, _ in self._holding_keys:
            self.containers.add(cont)

    def _compute_hand_and_table_status(self, state_bitwise: int):
        hands_empty: Set[str] = set()
        holding_by_hand: Dict[str, str] = {}
        holding_by_container: Dict[str, str] = {}
        on_table: Set[str] = set()

        # Hand empties
        for h, fid in self._handempty_keys.items():
            if (state_bitwise >> fid) & 1:
                hands_empty.add(h)

        # Holding pairs
        for hand, cont, fid in self._holding_keys:
            if (state_bitwise >> fid) & 1:
                holding_by_hand[hand] = cont
                holding_by_container[cont] = hand

        # On table containers
        for cont, fid in self._ontable_keys.items():
            if (state_bitwise >> fid) & 1:
                on_table.add(cont)

        return hands_empty, holding_by_hand, holding_by_container, on_table

    def _extract_hold_targets_from_tn(self, task_network) -> Set[str]:
        # Find containers that are going to be held explicitly in the TN (AchieveHolding or primitive grasp)
        targets: Set[str] = set()
        for item in task_network:
            name = getattr(item, 'name', '')
            pred, args = self._parse_name_brackets(name)
            if not pred:
                continue
            if pred == 'AchieveHolding' and len(args) == 2:
                _, cont = args
                targets.add(cont)
            elif pred == 'grasp' and len(args) == 2:
                # Primitive operator 'grasp[hand,container]'
                _, cont = args
                targets.add(cont)
        return targets

    # ---------- Estimators for key abstract tasks ----------
    def _estimate_AchieveContainsShotCocktail(self, state: int, shot: str, cocktail: str,
                                              hands_empty: Set[str],
                                              holding_by_container: Dict[str, str],
                                              on_table: Set[str],
                                              hold_targets: Set[str]) -> int:
        # Already done?
        if self._is_true(state, 'contains', (shot, cocktail)):
            return 0

        clean_shot_lb = self._clean_shot_lb(state, shot)  # AchieveCleanShot LB

        # Evaluate across all shakers and take the minimal LB
        best = None
        for shaker in (self.shakers if self.shakers else self.containers):
            # If no shakers are known, we still provide a conservative LB using containers
            hold_cost = 0 if shaker in hold_targets else self._hold_cost_container(state, shaker, hands_empty, holding_by_container, on_table)

            has_cocktail = self._is_true(state, 'contains', (shaker, cocktail)) and self._is_true(state, 'shaked', (shaker,))
            ing1, ing2 = self.cocktail_parts.get(cocktail, (None, None))
            has_i1 = ing1 is not None and self._is_true(state, 'contains', (shaker, ing1))
            has_i2 = ing2 is not None and self._is_true(state, 'contains', (shaker, ing2))

            if has_cocktail:
                cand = clean_shot_lb + hold_cost + 1  # just pour to shot
            elif has_i1 and has_i2:
                cand = clean_shot_lb + hold_cost + 1 + 1  # shake + pour
            elif (has_i1 and not has_i2) or (has_i2 and not has_i1):
                cand = clean_shot_lb + 3 + hold_cost + 1 + 1  # add 1 ingredient (3) + shake + pour
            else:
                cand = clean_shot_lb + 6 + hold_cost + 1 + 1  # add 2 ingredients (6) + shake + pour

            if best is None or cand < best:
                best = cand

        if best is None:
            # Fallback minimal: clean shot + at least a pour
            best = clean_shot_lb + 1
        return best

    def _estimate_AchieveContainsShakerCocktail(self, state: int, shaker: str, cocktail: str,
                                                hands_empty: Set[str],
                                                holding_by_container: Dict[str, str],
                                                on_table: Set[str],
                                                hold_targets: Set[str]) -> int:
        # Already done?
        if self._is_true(state, 'contains', (shaker, cocktail)):
            return 0

        hold_cost = 0 if shaker in hold_targets else self._hold_cost_container(state, shaker, hands_empty, holding_by_container, on_table)

        ing1, ing2 = self.cocktail_parts.get(cocktail, (None, None))
        has_i1 = ing1 is not None and self._is_true(state, 'contains', (shaker, ing1))
        has_i2 = ing2 is not None and self._is_true(state, 'contains', (shaker, ing2))

        if has_i1 and has_i2:
            return hold_cost + 1  # shake
        elif (has_i1 and not has_i2) or (has_i2 and not has_i1):
            return 3 + hold_cost + 1  # add missing + shake
        else:
            return 6 + hold_cost + 1  # add two + shake

    def _estimate_AchieveContainsShakerIngredient(self, state: int, shaker: str, ingredient: str) -> int:
        # If already in shaker, 0; else best-case 3 actions
        if self._is_true(state, 'contains', (shaker, ingredient)):
            return 0
        return 3

    def _estimate_AchieveContainsShotIngredient(self, state: int, shot: str, ingredient: str,
                                                hands_empty: Set[str],
                                                holding_by_container: Dict[str, str],
                                                on_table: Set[str]) -> int:
        if self._is_true(state, 'contains', (shot, ingredient)):
            return 0
        clean_lb = self._clean_shot_lb(state, shot)
        hold_cost = self._hold_cost_container(state, shot, hands_empty, holding_by_container, on_table)
        return clean_lb + hold_cost + 1  # fill-shot

    # ---------- Cost atoms ----------
    def _clean_shot_lb(self, state: int, shot: str) -> int:
        # If already clean, zero. Otherwise at least one action (clean-shot) will be needed.
        return 0 if self._is_true(state, 'clean', (shot,)) else 1

    def _hold_cost_container(self, state: int, container: str, hands_empty: Set[str],
                              holding_by_container: Dict[str, str], on_table: Set[str]) -> int:
        # 0 if already holding the container with any hand
        if container in holding_by_container:
            return 0
        # 1 if any hand is empty and container is on the table (can grasp directly)
        if hands_empty and (container in on_table or not self._has_fact_key('ontable', (container,))):
            # If ontable fact is missing, be optimistic and allow grasp in 1 step
            return 1
        # Otherwise we need at least a drop (1) and then a grasp (1)
        return 2

    def _hold_cost_exact_hand(self, state: int, hand: str, container: str,
                              hands_empty: Set[str], holding_by_hand: Dict[str, str], on_table: Set[str]) -> int:
        # 0 if that exact hand already holds container
        if holding_by_hand.get(hand) == container:
            return 0
        # 1 if that hand is empty and container on table (or optimistically unknown ontable fact)
        if hand in hands_empty and (container in on_table or not self._has_fact_key('ontable', (container,))):
            return 1
        # If the other hand holds the container, we need at least drop + grasp => 2
        return 2

    # ---------- Bitwise fact access ----------
    def _get_fact_id(self, predicate: str, args: Tuple[str, ...]) -> Optional[int]:
        return self.fact_by_key.get((predicate, tuple(args)))

    def _has_fact_key(self, predicate: str, args: Tuple[str, ...]) -> bool:
        return (predicate, tuple(args)) in self.fact_by_key

    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        fid = self._get_fact_id(predicate, args)
        if fid is None:
            return False
        return ((state_bitwise >> fid) & 1) == 1

    # ---------- Parsing helpers ----------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        # Facts format: +predicate[arg1,arg2]
        if not name:
            return None, []
        n = name.strip()
        if n.startswith('+') or n.startswith('-'):
            n = n[1:]
        b = n.find('[')
        if b > 0 and n.endswith(']'):
            pred = n[:b]
            args_str = n[b+1:-1]
            args = args_str.split(',') if args_str else []
            return pred, args
        return None, []

    def _parse_name_brackets(self, name: str) -> Tuple[Optional[str], List[str]]:
        # Tasks/operators are usually formatted as Predicate[arg1,arg2,...]
        # Be permissive but fast.
        if not name:
            return None, []
        n = name.strip()
        b = n.find('[')
        if b > 0 and n.endswith(']'):
            pred = n[:b]
            args_str = n[b+1:-1]
            args = args_str.split(',') if args_str else []
            return pred, args
        return None, []

    def _is_operator(self, item) -> bool:
        # Heuristically detect Operator by presence of typical fields
        return hasattr(item, 'pos_precons') and hasattr(item, 'add_effects') and hasattr(item, 'apply')


# Alias to match expected external name
__all__ = ['BarmanGPT5Heuristic10']
