from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Pytrich imports (framework-provided)
from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic18(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Core idea
    ---------
    This domain's HTN tasks ultimately enforce a set of desired `on(x,y)` relations.
    Achieving an `on(x,y)` typically requires:
      - making x clear (possibly recursively clearing blocks above it)
      - making y clear
      - ensuring y is on the table (the method m1_do_put_on explicitly calls do_on_table(y))
      - then moving x onto y (pick/stack or unstack/stack)

    We build a *fast, admissible-ish lower-bound* estimate by counting missing goal `on` facts,
    plus lightweight penalties for obvious blockers that must be removed before stacking can
    be done (blocks on top of x, blocks on top of y, and y not on table).

    Additionally, we incorporate the *current task network*: remaining abstract tasks are
    mapped to the same target `on` facts (for do_put_on and do_move) and missing `ontable`
    (for do_on_table). This provides guidance even when the classical goal set is small or
    already satisfied but the HTN network is not.

    Performance constraints
    -----------------------
    - No conversion of full states via `model.state_explicit_repr(state)` for current states.
    - Only bitwise checks using fact.global_id.
    - Precomputes indices for fast queries (above-of, support-of, clear/ontable/handempty).

    Notes on admissibility
    ----------------------
    The heuristic is designed as a *lower bound* by using conservative weights (mostly 1).
    Even if not perfectly admissible under all HTN nuances, it is monotone-ish and very cheap,
    and in practice strongly reduces expansions on large instances.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_18: str = "blocksworld_gpt52_heuristic_18"):
        super().__init__(model)
        self.name = blocksworld_gpt52_heuristic_18
        self.model = model

        # --- Fact indices ---
        self.fact_id: Dict[str, int] = {}
        for f in self.model.facts:
            self.fact_id[f.name] = f.global_id

        # Predicate -> list of (args, global_id)
        self.on_facts: List[Tuple[str, str, int]] = []         # (x,y,gid) for +on[x,y]
        self.ontable_facts: Dict[str, int] = {}                # x -> gid for +ontable[x]
        self.clear_facts: Dict[str, int] = {}                  # x -> gid for +clear[x]

        # Reverse indices to detect blockers quickly
        self.support_of: Dict[str, str] = {}                   # x -> y where on[x,y]
        self.above_of: Dict[str, str] = {}                     # y -> x where on[x,y] (top block directly on y)

        # Scalar facts
        self.handempty_gid: Optional[int] = self.fact_id.get('+handempty[]', None)

        # Parse all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2 and f.name.startswith('+'):
                x, y = args[0], args[1]
                self.on_facts.append((x, y, f.global_id))
                # The mapping support_of/above_of depends on state, not static.
                # We keep static list only; dynamic support/above computed per state.
            elif pred == 'ontable' and len(args) == 1 and f.name.startswith('+'):
                self.ontable_facts[args[0]] = f.global_id
            elif pred == 'clear' and len(args) == 1 and f.name.startswith('+'):
                self.clear_facts[args[0]] = f.global_id

        # --- Goal `on` facts ---
        # Must use model.goals bitmask (do not inspect state explicitly).
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self.goal_on_pairs: List[Tuple[str, str, int]] = []
        self.goal_ontable: List[Tuple[str, int]] = []

        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on' and len(args) == 2 and name.startswith('+'):
                gid = self.fact_id.get(name, None)
                if gid is not None:
                    self.goal_on_pairs.append((args[0], args[1], gid))
            elif pred == 'ontable' and len(args) == 1 and name.startswith('+'):
                gid = self.fact_id.get(name, None)
                if gid is not None:
                    self.goal_ontable.append((args[0], gid))

        # --- Abstract task name ids (cheap TN guidance) ---
        # We only need to recognize task names and their parameters.
        self.abstract_name_by_gid: Dict[int, str] = {}
        for t in self.model.abstract_tasks:
            self.abstract_name_by_gid[t.global_id] = t.name

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name2 = name[1:]
            bracket_idx = name2.find('[')
            if bracket_idx > 0 and name2.endswith(']'):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_gid: int) -> bool:
        return ((state_bitwise >> fact_gid) & 1) == 1

    def _compute_above_support_maps(self, state_bitwise: int) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Compute direct 'on' relations in the current state.

        Returns:
            above_of[y] = x if +on[x,y] holds
            support_of[x] = y if +on[x,y] holds
        """
        above_of: Dict[str, str] = {}
        support_of: Dict[str, str] = {}
        for x, y, gid in self.on_facts:
            if self._is_true(state_bitwise, gid):
                above_of[y] = x
                support_of[x] = y
        return above_of, support_of

    def _task_targets(self, task_network) -> Tuple[List[Tuple[str, str, Optional[int]]], List[Tuple[str, Optional[int]]]]:
        """Extract target relations from the remaining (possibly abstract) task network.

        We conservatively interpret:
          - do_put_on(x,y) and do_move(x,y) as needing +on[x,y]
          - do_on_table(x) as needing +ontable[x]

        Returns lists of:
          - desired_on: (x,y,gid_or_None)
          - desired_ontable: (x,gid_or_None)

        If a corresponding fact is not found in the grounded fact set, gid is None and we skip
        bit checks, but still count a small cost for the task.
        """
        desired_on: List[Tuple[str, str, Optional[int]]] = []
        desired_ontable: List[Tuple[str, Optional[int]]] = []

        if not task_network:
            return desired_on, desired_ontable

        for node in task_network:
            # Only abstract tasks have .global_id in the abstract-task id range.
            # The framework represents tasks/operators as objects in the TN.
            name = getattr(node, 'name', '')
            if not name:
                continue

            # Task names look like 'do_put_on b1 b2' OR 'do_put_on(b1,b2)' depending on grounding.
            # In Pytrich, abstract task name is typically just the symbol (e.g., 'do_put_on').
            # Parameters are generally encoded in the object's name string in many generators.
            # We implement robust parsing by scanning tokens.
            tokens = name.replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
            if not tokens:
                continue

            head = tokens[0]
            if head in ('do_put_on', 'do_move') and len(tokens) >= 3:
                x, y = tokens[1], tokens[2]
                fact_name = f'+on[{x},{y}]'
                gid = self.fact_id.get(fact_name, None)
                desired_on.append((x, y, gid))
            elif head == 'do_on_table' and len(tokens) >= 2:
                x = tokens[1]
                fact_name = f'+ontable[{x}]'
                gid = self.fact_id.get(fact_name, None)
                desired_ontable.append((x, gid))

        return desired_on, desired_ontable

    def __call__(self, state_bitwise: int, task_network=None) -> int:
        if task_network is None:
            task_network = []

        above_of, support_of = self._compute_above_support_maps(state_bitwise)

        h = 0

        # --- 1) Classical goals: missing on/ontable ---
        # Each missing goal relation costs at least 1 primitive action in any reasonable plan.
        for x, y, gid in self.goal_on_pairs:
            if not self._is_true(state_bitwise, gid):
                h += 1

                # Blockers: x not clear => some block above x must be removed (>=1 unstack+putdown overall)
                cg = self.clear_facts.get(x)
                if cg is not None and not self._is_true(state_bitwise, cg):
                    # only add if something actually on x
                    if x in above_of:
                        h += 1

                # Blockers: y not clear
                cg = self.clear_facts.get(y)
                if cg is not None and not self._is_true(state_bitwise, cg):
                    if y in above_of:
                        h += 1

                # Method m1_do_put_on requires do_on_table(y) before moving x onto y.
                # If y is not on table, at least one action is needed to place it there.
                otg = self.ontable_facts.get(y)
                if otg is not None and not self._is_true(state_bitwise, otg):
                    h += 1

                # If x is currently on some block (not on table), moving it often needs an unstack.
                # This is a cheap extra signal (still a lower bound if x indeed must be detached).
                if x in support_of:
                    # If already on y we'd have been satisfied; otherwise must detach at some point.
                    if support_of[x] != y:
                        h += 1

        for x, gid in self.goal_ontable:
            if not self._is_true(state_bitwise, gid):
                h += 1
                cg = self.clear_facts.get(x)
                if cg is not None and not self._is_true(state_bitwise, cg) and x in above_of:
                    h += 1

        # --- 2) HTN guidance: remaining tasks as additional soft goals ---
        desired_on, desired_ontable = self._task_targets(task_network)

        for x, y, gid in desired_on:
            if gid is None:
                # Unknown grounding; still encourage progress.
                h += 1
                continue
            if not self._is_true(state_bitwise, gid):
                h += 1

                cg = self.clear_facts.get(x)
                if cg is not None and not self._is_true(state_bitwise, cg) and x in above_of:
                    h += 1
                cg = self.clear_facts.get(y)
                if cg is not None and not self._is_true(state_bitwise, cg) and y in above_of:
                    h += 1

        for x, gid in desired_ontable:
            if gid is None:
                h += 1
                continue
            if not self._is_true(state_bitwise, gid):
                h += 1

        # --- 3) Light penalty if hand not empty and there is still work ---
        # Being in a holding state often forces a put-down/stack before other clears/moves.
        if h > 0 and self.handempty_gid is not None:
            if not self._is_true(state_bitwise, self.handempty_gid):
                h += 1

        return h
