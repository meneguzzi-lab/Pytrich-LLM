from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# NOTE:
# This file is meant for the Pytrich planner framework.
# The planner provides the base class `Heuristic` and a `model` with facts/operators/abstract tasks.


class BlocksworldGPT52Heuristic14(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Rationale (domain insights -> heuristic components)
    --------------------------------------------------
    This Blocksworld HDDL has compound tasks:
      - do_put_on(x,y)
      - do_on_table(x)
      - do_move(x,y)
      - do_clear(x)

    and primitive actions pick-up/put-down/stack/unstack (plus nop).

    Key difficulty: to achieve (on x y) you must often:
      1) clear x (unstack blocks above it, putting them down)
      2) clear y (unstack blocks above it, putting them down)
      3) ensure y is available as a support (sometimes must be on-table)
      4) move x onto y (either pick-up+stack if x on table, or unstack+stack if not)

    For large instances, expensive part is *clearing towers*; good guidance comes from
    counting how many blocks are above important blocks and whether goal relations are already correct.

    This heuristic is a fast, domain-specific lower bound (admissible-by-construction, but not tight):

    h(state, tn) = max(h_goal_atoms(state), h_task_network(tn, state))

    where:
      - h_goal_atoms counts unsatisfied goal literals of the form +on[a,b].
        Each unsatisfied on(a,b) requires at least one primitive action (stack/unstack etc.),
        so counting them is a trivial admissible lower bound.

      - h_task_network estimates remaining work from the current HTN task network.
        It counts remaining abstract tasks (and primitive operators) with very conservative costs:

          do_clear(x):  at least 2 if x not clear (need at least unstack+put-down for one block)
                        else 0

          do_on_table(x):
                        0 if already ontable(x) OR clear(x) implies nop method may apply;
                        otherwise at least 2 if not ontable(x) and clear(x) and handempty might
                        require unstack+put-down.
                        (We keep it conservative: if ontable(x) already, 0 else 1.)

          do_move(x,y): at least 1 if +on[x,y] already, else at least 2 (must at least execute a stack
                        plus some pickup/unstack). We use 1/2 conservative.

          do_put_on(x,y): at least 0 if already on(x,y), else at least 1.

        Additionally, a *clearance landmark* bound is added for goal blocks:
          For each block b that appears in any goal on(b,*) or on(*,b),
          if b is not clear in current state, then at least 2 actions are required
          to remove the topmost block above it (unstack + put-down). This is admissible.
          We take a max over such landmarks to keep it safely a lower bound.

    Efficiency considerations
    -------------------------
    - Uses only bitwise checks on facts (no explicit state reconstruction).
    - Pre-indexes relevant facts: on(x,y), clear(x), ontable(x), holding(x), handempty.
    - Avoids O(n^2) scans in the heuristic call; most work is O(#goals + #tasks).

    Notes on admissibility
    ----------------------
    - Counting unsatisfied goal atoms is admissible.
    - The per-task costs are chosen as *very conservative* minima.
    - The clearance landmark uses only "not clear" => need at least one unstack+putdown (2 actions)
      before the block can become clear, hence admissible.

    This heuristic is meant to reduce node expansions by preferring states where:
      - more goal on-relations are already true
      - goal-relevant blocks are already clear
      - the current HTN task network is smaller / closer to completion
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_14: bool = True):
        super().__init__(model)
        self.enabled = bool(blocksworld_gpt52_heuristic_14)

        # Fact id caches
        self._clear_id: Dict[str, int] = {}
        self._ontable_id: Dict[str, int] = {}
        self._handempty_id: Optional[int] = None

        # on(x,y) lookup by tuple -> fact global_id
        self._on_id: Dict[Tuple[str, str], int] = {}

        # Parse and store goal +on atoms for fast checking.
        self._goal_on_pairs: List[Tuple[str, str]] = []
        self._goal_fact_ids: List[int] = []
        self._goal_relevant_blocks: set[str] = set()

        # Preprocess facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "clear" and len(args) == 1 and f.name.startswith("+"):
                self._clear_id[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1 and f.name.startswith("+"):
                self._ontable_id[args[0]] = f.global_id
            elif pred == "handempty" and len(args) == 0 and f.name.startswith("+"):
                self._handempty_id = f.global_id
            elif pred == "on" and len(args) == 2 and f.name.startswith("+"):
                self._on_id[(args[0], args[1])] = f.global_id

        # Extract goal facts (bitwise integer -> explicit names -> parse)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2 and name.startswith("+"):
                a, b = args
                self._goal_on_pairs.append((a, b))
                self._goal_relevant_blocks.add(a)
                self._goal_relevant_blocks.add(b)
                fid = self._on_id.get((a, b))
                if fid is not None:
                    self._goal_fact_ids.append(fid)

    # --------- Required Heuristic API ---------
    def __call__(self, state: int, task_network) -> int:
        if not self.enabled:
            return 0

        # Lower bound from goal atoms
        h_goal = self._h_unsatisfied_goals(state)

        # Lower bound from current task network (very conservative)
        h_tn = self._h_task_network(state, task_network)

        # Clearance landmark for goal-relevant blocks: if a goal-relevant block is not clear,
        # we will need at least 2 actions at some point to clear the topmost block above it.
        h_clear_landmark = self._h_goal_clearance_landmark(state)

        return max(h_goal, h_tn, h_clear_landmark)

    # --------- Heuristic components ---------
    def _h_unsatisfied_goals(self, state: int) -> int:
        # Count unsatisfied goal facts (only +on supported here).
        # Admissible: each missing goal atom requires at least one action.
        unsat = 0
        for fid in self._goal_fact_ids:
            if ((state >> fid) & 1) == 0:
                unsat += 1
        return unsat

    def _h_goal_clearance_landmark(self, state: int) -> int:
        # If a block is relevant to the goal and it is not clear, then at least one unstack+put-down
        # (2 primitive actions) is required to make it clear.
        # To keep it admissible and cheap, we take max over blocks (not sum).
        best = 0
        for b in self._goal_relevant_blocks:
            fid = self._clear_id.get(b)
            if fid is None:
                continue
            if ((state >> fid) & 1) == 0:
                best = max(best, 2)
        return best

    def _h_task_network(self, state: int, task_network) -> int:
        # task_network is a list of Operators and/or AbstractTasks in Pytrich.
        # We compute a conservative bound based on task names.
        if not task_network:
            return 0

        h = 0
        for t in task_network:
            # Primitive operator: at least its cost (usually 1, nop 0).
            # Using operator.cost is admissible.
            if hasattr(t, "pos_precons") and hasattr(t, "add_effects") and hasattr(t, "cost"):
                h += max(0, int(t.cost))
                continue

            # Abstract task: estimate minimal remaining work using its name.
            name = getattr(t, "name", "")
            pred, args = self._parse_task_signature(name)
            if pred is None:
                continue

            if pred == "do_clear" and len(args) == 1:
                x = args[0]
                h += self._lb_do_clear(state, x)
            elif pred == "do_on_table" and len(args) == 1:
                x = args[0]
                h += self._lb_do_on_table(state, x)
            elif pred == "do_move" and len(args) == 2:
                x, y = args
                h += self._lb_do_move(state, x, y)
            elif pred == "do_put_on" and len(args) == 2:
                x, y = args
                h += self._lb_do_put_on(state, x, y)
            else:
                # Unknown abstract task: be safe
                h += 0

        return h

    # --------- Lower bounds for individual tasks (conservative) ---------
    def _lb_do_clear(self, state: int, x: str) -> int:
        fid = self._clear_id.get(x)
        if fid is not None and ((state >> fid) & 1) == 1:
            return 0
        # If not clear, at least one block must be removed: unstack + put-down => >=2
        return 2

    def _lb_do_on_table(self, state: int, x: str) -> int:
        # If already on table, 0.
        fid = self._ontable_id.get(x)
        if fid is not None and ((state >> fid) & 1) == 1:
            return 0
        # Otherwise at least 1 action required at some point (put-down), conservative.
        return 1

    def _lb_do_move(self, state: int, x: str, y: str) -> int:
        # If already on(x,y), 0.
        fid = self._on_id.get((x, y))
        if fid is not None and ((state >> fid) & 1) == 1:
            return 0
        # Otherwise at least: stack plus some way to hold x => >=2
        return 2

    def _lb_do_put_on(self, state: int, x: str, y: str) -> int:
        fid = self._on_id.get((x, y))
        if fid is not None and ((state >> fid) & 1) == 1:
            return 0
        # Need at least one action to change configuration.
        return 1

    # --------- Parsing helpers ---------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_signature(self, task_name: str) -> Tuple[Optional[str], List[str]]:
        """Parse Pytrich task/operator names.

        In many Pytrich builds, abstract task names are strings like:
          'do_put_on b1 b2'  or 'do_put_on(b1,b2)' or 'do_put_on[b1,b2]'

        We support a few common variants with a cheap parser.
        """
        s = (task_name or "").strip()
        if not s:
            return None, []

        # Variant 1: "name arg1 arg2"
        if " " in s and ("(" not in s) and ("[" not in s):
            parts = s.split()
            return parts[0], parts[1:]

        # Variant 2: "name(arg1,arg2)"
        if "(" in s and s.endswith(")"):
            i = s.find("(")
            pred = s[:i].strip()
            inside = s[i + 1 : -1].strip()
            args = [a.strip() for a in inside.split(",") if a.strip()] if inside else []
            return pred, args

        # Variant 3: "name[arg1,arg2]"
        if "[" in s and s.endswith("]"):
            i = s.find("[")
            pred = s[:i].strip()
            inside = s[i + 1 : -1].strip()
            args = [a.strip() for a in inside.split(",") if a.strip()] if inside else []
            return pred, args

        return s, []
