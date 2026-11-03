"""
DepotChatGPTHeuristic2 for Pytrich (improved, domain-aware heuristic)

Class name: DepotChatGPTHeuristic2
__init__ parameter: depot_gpt_heuristic_2   <-- required by your instructions

Key ideas:
 - Count crates that must be moved above a crate (stack depth) and above target surface.
 - Add per-goal handling costs (load/unload/lift/drop).
 - Amortize DRIVE cost across unique source->target location transfers.
 - Robust state parsing from several node/model representations.
 - Always returns a numeric float and calls super().update_info(h_value).
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import math

from Pytrich.Heuristics.heuristic import Heuristic


class DepotChatGPTHeuristic2(Heuristic):
    def __init__(self, depot_gpt_heuristic_2: str = "depot_gpt_heuristic_2",
                 lift_cost: float = 1.0,
                 load_cost: float = 0.8,
                 unload_cost: float = 0.8,
                 drop_cost: float = 0.4,
                 drive_cost: float = 2.2,
                 amortize_drive: bool = True,
                 debug: bool = False):
        """
        depot_gpt_heuristic_2 : required parameter name by spec.
        Other params tune relative costs (higher -> heuristic thinks it's 'harder').

        Set debug=True to get occasional prints (only used if you call .__output__ or inspect the object).
        """
        # required parameter naming / super init
        super().__init__(name=depot_gpt_heuristic_2)
        self.depot_gpt_heuristic_2 = depot_gpt_heuristic_2

        # tunable weights (domain-driven)
        self.LIFT = float(lift_cost)
        self.LOAD = float(load_cost)
        self.UNLOAD = float(unload_cost)
        self.DROP = float(drop_cost)
        self.DRIVE = float(drive_cost)

        # whether we amortize drive costs across identical transfers
        self.amortize_drive = bool(amortize_drive)

        # model / static data filled in initialize()
        self.model = None
        self.goals: List[Tuple[str, str]] = []  # list of (crate, target_surface)
        self.debug = bool(debug)

    # -------------------------
    # Initialization / helpers
    # -------------------------
    def initialize(self, model: Any, initial_node: Any):
        """Called by planner once. Extract goals and other static info if possible."""
        self.model = model
        self.goals = self._extract_goals_from_node_or_model(initial_node, model)
        # call parent initialize with an initial heuristic (numeric)
        h0 = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, float(h0))

    def _extract_goals_from_node_or_model(self, node: Any, model: Any) -> List[Tuple[str, str]]:
        """
        Best-effort extraction of ordered-do_put_on goals from the initial node or model.
        We return list of pairs (crate_name, surface_name).
        """
        goals: List[Tuple[str, str]] = []

        # try node.task_network-like objects first
        try:
            tn = getattr(node, "task_network", None) or getattr(node, "tn", None) or getattr(node, "htn", None)
            if tn:
                # tn may be an object with .subtasks, or a list
                subtasks = getattr(tn, "subtasks", None) or tn
                for t in subtasks:
                    name = getattr(t, "name", None)
                    params = getattr(t, "parameters", None) or getattr(t, "parameters", None) or getattr(t, "args", None)
                    # robust checks if params are objects vs strings
                    if name and name.lower().endswith("do_put_on"):
                        if params and len(params) >= 2:
                            crate = params[0]
                            target = params[1]
                            # params might be objects; pick string repr if necessary
                            crate_s = crate if isinstance(crate, str) else str(crate)
                            target_s = target if isinstance(target, str) else str(target)
                            goals.append((crate_s, target_s))
        except Exception:
            pass

        # fallback: try model-level HTN / goal lists
        if not goals and model is not None:
            # some models store an 'htn' or 'task_network' or 'tasks' attr
            for attr in ("htn", "task_network", "tasks", "subtasks"):
                tn = getattr(model, attr, None)
                if tn:
                    try:
                        for t in tn:
                            name = getattr(t, "name", None) or str(getattr(t, "task", t))
                            params = getattr(t, "parameters", None) or getattr(t, "args", None) or getattr(t, "parameters", None)
                            if name and "do_put_on" in name.lower():
                                if params and len(params) >= 2:
                                    crate_s = params[0] if isinstance(params[0], str) else str(params[0])
                                    target_s = params[1] if isinstance(params[1], str) else str(params[1])
                                    goals.append((crate_s, target_s))
                        if goals:
                            break
                    except Exception:
                        continue

        # final fallback: try model.goals if it contains pairs in some textual form
        if not goals and model is not None:
            try:
                mg = getattr(model, "goals", None)
                if isinstance(mg, (list, set)):
                    for g in mg:
                        # accept facts like ('on','crate0','pallet1') or "(on crate0 pallet1)"
                        if isinstance(g, tuple) and len(g) >= 3 and g[0] == "on":
                            goals.append((str(g[1]), str(g[2])))
                        elif isinstance(g, str) and g.strip().startswith("(on "):
                            toks = g.strip().lstrip("(").rstrip(")").split()
                            if len(toks) >= 3:
                                goals.append((toks[1], toks[2]))
            except Exception:
                pass

        return goals

    # -------------------------
    # State parsing utilities
    # -------------------------
    def _state_to_facts(self, node: Any) -> Set[Tuple[str, ...]]:
        """
        Convert node/state to a set of predicate tuples like ("on","crate0","pallet1").
        Handles:
         - node.state as a set/frozenset of tuples
         - node.state as int bitmask: uses model.state_explicit_repr(state) if available
         - node has 'state_set' or 'facts' attributes
         - model exposes a helper to render state
        """
        facts: Set[Tuple[str, ...]] = set()

        # try common attributes
        state_candidate = None
        for attr in ("state", "state_set", "s", "facts", "state_facts", "state_bitwise"):
            if hasattr(node, attr):
                state_candidate = getattr(node, attr)
                if state_candidate is not None:
                    break

        # If state is already a set of tuples / strings
        if isinstance(state_candidate, (set, frozenset)):
            for fact in state_candidate:
                if isinstance(fact, tuple):
                    facts.add(tuple(map(str, fact)))
                elif isinstance(fact, str):
                    parsed = self._parse_fact_string(fact)
                    if parsed:
                        facts.add(parsed)
            if facts:
                return facts

        # If state is an integer bitmask and model has conversion
        if isinstance(state_candidate, int) and self.model is not None:
            try:
                if hasattr(self.model, "state_explicit_repr"):
                    raw_list = self.model.state_explicit_repr(state_candidate)
                elif hasattr(self.model, "render_state"):
                    raw_list = self.model.render_state(state_candidate)
                else:
                    raw_list = None
                if raw_list:
                    for fr in raw_list:
                        if isinstance(fr, tuple):
                            facts.add(tuple(map(str, fr)))
                        elif isinstance(fr, str):
                            parsed = self._parse_fact_string(fr)
                            if parsed:
                                facts.add(parsed)
                    if facts:
                        return facts
            except Exception:
                pass

        # If heuristics can't find a good state representation, try node.to_string-like fallback
        # (sometimes nodes implement __repr__ containing facts)
        if isinstance(state_candidate, str):
            parsed = self._parse_fact_string(state_candidate)
            if parsed:
                facts.add(parsed)
            return facts

        # last resort: try node.__dict__ for 'facts' or 'state'
        try:
            nd = getattr(node, "__dict__", {})
            for v in nd.values():
                if isinstance(v, (set, frozenset)):
                    for fact in v:
                        if isinstance(fact, tuple):
                            facts.add(tuple(map(str, fact)))
                        elif isinstance(fact, str):
                            parsed = self._parse_fact_string(fact)
                            if parsed:
                                facts.add(parsed)
                elif isinstance(v, str):
                    parsed = self._parse_fact_string(v)
                    if parsed:
                        facts.add(parsed)
            return facts
        except Exception:
            return set()

    def _parse_fact_string(self, s: str) -> Optional[Tuple[str, ...]]:
        """
        Parse a fact string like "(on crate0 pallet1)" -> ("on","crate0","pallet1").
        Returns None if unable.
        """
        try:
            s2 = s.strip()
            if s2.startswith("(") and s2.endswith(")"):
                inner = s2[1:-1].strip()
                toks = inner.split()
                if len(toks) >= 1:
                    return tuple(map(str, toks))
            else:
                # also accept 'on crate0 pallet1' form
                toks = s2.split()
                if len(toks) >= 1:
                    return tuple(map(str, toks))
        except Exception:
            return None
        return None

    # -------------------------
    # Stack helpers
    # -------------------------
    def _build_maps_from_facts(self, facts: Iterable[Tuple[str, ...]]):
        """
        Build:
         - children_map: object -> [objects directly on it]
         - on_map: object -> object it's on (surface or crate)
         - at_map: object -> place
         - in_map: crate -> truck
         - hoist_available: hoist -> bool
        """
        children_map: Dict[str, List[str]] = {}
        on_map: Dict[str, str] = {}
        at_map: Dict[str, str] = {}
        in_map: Dict[str, str] = {}
        hoist_available: Dict[str, bool] = {}

        for f in facts:
            if not f:
                continue
            pred = f[0]
            args = f[1:]
            if pred == "on" and len(args) >= 2:
                child = args[0]
                parent = args[1]
                on_map[child] = parent
                children_map.setdefault(parent, []).append(child)
            elif pred == "at" and len(args) >= 2:
                obj = args[0]
                place = args[1]
                at_map[obj] = place
            elif pred == "in" and len(args) >= 2:
                crate = args[0]
                truck = args[1]
                in_map[crate] = truck
            elif pred == "available" and len(args) >= 1:
                ho = args[0]
                hoist_available[ho] = True
            elif pred == "lifting" and len(args) >= 2:
                # not used directly here, but presence indicates hoist busy
                ho = args[0]
                hoist_available[ho] = False

        return children_map, on_map, at_map, in_map, hoist_available

    def _count_above_recursive(self, obj: str, children_map: Dict[str, List[str]], visited: Optional[Set[str]] = None) -> int:
        """
        Number of crates that sit (directly or indirectly) on top of obj.
        """
        if visited is None:
            visited = set()
        if obj in visited:
            return 0
        visited.add(obj)
        total = 0
        for ch in children_map.get(obj, []):
            total += 1
            total += self._count_above_recursive(ch, children_map, visited)
        return total

    # -------------------------
    # Main heuristic calculation
    # -------------------------
    def _calculate_node_heuristic(self, node: Any) -> float:
        """
        Compute the heuristic estimate for `node`. Returns a float.
        Heuristic = sum(per-goal handling costs + amortized drive cost).
        """
        # parse facts
        facts = self._state_to_facts(node)
        if not facts:
            # fallback: if cannot parse state, return a conservative numeric based on task count
            try:
                tn = getattr(node, "task_network", None) or getattr(node, "tn", None)
                if tn:
                    length = len(getattr(tn, "subtasks", tn))
                    # fallback estimate: 0.8 * #tasks
                    h_fallback = float(length) * 0.8
                    return float(h_fallback)
            except Exception:
                return 0.0

        children_map, on_map, at_map, in_map, hoist_available = self._build_maps_from_facts(facts)

        total_cost = 0.0
        transfers: Set[Tuple[Optional[str], Optional[str]]] = set()

        # use the cached goals discovered in initialize() if present; else try to extract on-node
        goals = self.goals or self._extract_goals_from_node_or_model(node, self.model)
        for (crate, target) in goals:
            crate_s = str(crate)
            target_s = str(target)

            # already satisfied?
            if ("on", crate_s, target_s) in facts:
                continue

            # compute "above" counts
            crates_above = self._count_above_recursive(crate_s, children_map)
            target_above = self._count_above_recursive(target_s, children_map)

            # base cost to clear crate and clear target
            cost = crates_above * self.LIFT + target_above * self.LIFT

            # determine source location (crate or truck containing crate)
            src_loc = None
            if crate_s in in_map:
                truck = in_map[crate_s]
                src_loc = at_map.get(truck)
                # crate is already in truck -> unloading + drop required at target
                cost += self.UNLOAD + self.DROP
            else:
                # not in truck => will need load/unload/drop sequence
                cost += self.LOAD + self.UNLOAD + self.DROP
                src_loc = at_map.get(crate_s)

            # determine target location
            tgt_loc = at_map.get(target_s)

            # if locations are known and different, mark a transfer
            if src_loc is None or tgt_loc is None:
                # unknown location: assume a transfer needed (conservative)
                transfers.add((src_loc, tgt_loc))
            else:
                if src_loc != tgt_loc:
                    transfers.add((src_loc, tgt_loc))

            # If target surface itself is a crate that is also a goal to be put somewhere else in
            # the same HTN (goal-on-goal), some stacking/opportunity exists. Here we only conservatively
            # leave the per-goal cost as-is (we don't double count further).
            total_cost += cost

        # amortize drive costs across unique transfers (helps reduce over-counting)
        if self.amortize_drive and transfers:
            # remove transfers where src==tgt or one is None (count them but as 1)
            unique = set()
            for (s, t) in transfers:
                # treat None as distinct; we'll count it as one drive
                unique.add((s, t))
            drive_cost_total = len(unique) * self.DRIVE
            total_cost += drive_cost_total
        else:
            # if not amortizing, conservative per-goal drive add: each goal that had src!=tgt pays DRIVE once
            # (but we already added transfers above; this branch is rarely used)
            for (s, t) in transfers:
                if s != t:
                    total_cost += self.DRIVE

        # clamp and ensure numeric
        try:
            h_val = float(total_cost)
        except Exception:
            h_val = 0.0

        return max(0.0, h_val)

    # -------------------------
    # Heuristic callable
    # -------------------------
    def __call__(self, parent_node: Any, node: Any):
        """
        Planner calls this for node expansion. Must return numeric float.
        """
        h_value = float(self._calculate_node_heuristic(node))
        # update base class bookkeeping with a numeric value
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotChatGPTHeuristic2(name={self.depot_gpt_heuristic_2})"

    def __output__(self):
        return (
            f"DepotChatGPTHeuristic2(name={self.depot_gpt_heuristic_2}, "
            f"LIFT={self.LIFT}, LOAD={self.LOAD}, UNLOAD={self.UNLOAD}, DROP={self.DROP}, DRIVE={self.DRIVE}, amortize_drive={self.amortize_drive})"
        )
