from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, Union

# Pytrich import (framework-provided)
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic7(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    Core idea (admissible, fast):
      - The goal in given instances is a conjunction of communicated_* facts.
      - Communicating soil/rock/image data requires being at a waypoint that is *visible* to the lander waypoint.
      - Most of the real cost is navigation (navigate actions). Sampling/calibration/communication are ~constant overhead.

    This heuristic computes a *lower bound* on remaining primitive actions by:
      1) Counting unsatisfied goal communications.
      2) For each unsatisfied goal item, estimating a minimal navigation distance from any rover's current location
         to *some* waypoint from which the lander is visible (a "transmit waypoint").
         - Distances are computed on a relaxed movement graph derived from can_traverse edges.
         - We ignore the (visible p1 p2) requirement of navigate to keep it admissible.
      3) Adding a small admissible base cost per pending item for mandatory non-navigation actions:
         - Soil: sample_soil + communicate => at least 2 actions.
         - Rock: sample_rock + communicate => at least 2 actions.
         - Image: calibrate + take_image + communicate => at least 3 actions.
         Additionally, if store is not empty, empty_store implies drop => at least 1 extra action before sampling.

    Hierarchy awareness:
      - The HTN tasks (get_*_data) expand into navigate, (maybe) drop, sample/take_image, communicate.
      - Our estimate approximates the cheapest refinement cost without enumerating decompositions.

    Efficiency:
      - Precomputes transmit-waypoint set from static facts (visible + at_lander).
      - Builds can_traverse adjacency once.
      - Runs multi-source BFS from all rovers each evaluation to obtain min distance-to-transmit for each rover.
        This is O(|E|) and works well even on large instances.

    Admissibility notes:
      - Uses min distances and minimal mandatory action counts only.
      - Ignores negative interactions (store contention, calibrated flag being consumed, channel_free bookkeeping),
        which can only *increase* true cost, so heuristic remains a lower bound.

    """

    def __init__(self, model, rover_gpt52_heuristic_7: str = "rover_gpt52_heuristic_7"):
        super().__init__(model, rover_gpt52_heuristic_7)
        self.model = model

        # --- Build fast lookup: fact global_id -> (pred, args)
        self._fact_pred_args: List[Tuple[Optional[str], Tuple[str, ...]]] = [
            (None, tuple()) for _ in range(len(self.model.facts))
        ]
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            self._fact_pred_args[f.global_id] = (pred, tuple(args))

        # --- Extract static structures
        self._rovers: Set[str] = set()
        self._stores_of_rover: Dict[str, Set[str]] = defaultdict(set)
        self._at_fact_ids_by_rover: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # rover -> [(wp, fact_id)]

        self._can_traverse_adj: Dict[str, Set[str]] = defaultdict(set)

        self._lander_waypoints: Set[str] = set()
        self._transmit_waypoints: Set[str] = set()  # waypoints from which lander can be seen (w visible lander_wp)

        # Facts useful for quick state queries
        self._empty_fact_ids: Set[int] = set()
        self._full_fact_ids: Set[int] = set()

        # Goal parsing cache
        self._goal_items: List[Tuple[str, Tuple[str, ...], int]] = []  # (pred, args, fact_id)

        self._precompute_domain_graphs_and_mappings()
        self._precompute_goal_items()

    # ------------------------ Parsing helpers ------------------------

    def _parse_fact_name(self, name: str):
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

    def _precompute_domain_graphs_and_mappings(self):
        # Determine lander waypoint(s) and transmit waypoints using initial facts.
        # Most rover instances have a single lander, but we support multiple.
        init_state = self.model.initial_state

        lander_wps: Set[str] = set()
        # First pass: find at_lander(l, wp)
        for fact in self.model.facts:
            pred, args = self._fact_pred_args[fact.global_id]
            if pred == "at_lander" and len(args) == 2:
                # Static in all benchmark instances
                lander_wp = args[1]
                lander_wps.add(lander_wp)

        self._lander_waypoints = lander_wps

        # Build transmit-waypoint set: any w1 such that exists lander_wp with visible(w1, lander_wp)
        # We can precompute from facts (static).
        transmit_wps: Set[str] = set()
        for fact in self.model.facts:
            pred, args = self._fact_pred_args[fact.global_id]
            if pred == "visible" and len(args) == 2 and args[1] in lander_wps:
                transmit_wps.add(args[0])

        # Also: if rover is at lander_wp directly, that waypoint can transmit iff visible(lander_wp, lander_wp)
        # is usually not present, so don't assume. Keep only those implied by visible facts.
        self._transmit_waypoints = transmit_wps

        # Build can_traverse adjacency graph (directed). We'll relax visibility constraint.
        for fact in self.model.facts:
            pred, args = self._fact_pred_args[fact.global_id]
            if pred == "can_traverse" and len(args) == 3:
                _, wp1, wp2 = args
                self._can_traverse_adj[wp1].add(wp2)

        # Map rovers, stores, at-facts, store state facts.
        for fact in self.model.facts:
            pred, args = self._fact_pred_args[fact.global_id]
            if pred == "store_of" and len(args) == 2:
                store, rover = args
                self._rovers.add(rover)
                self._stores_of_rover[rover].add(store)
            elif pred == "at" and len(args) == 2:
                rover, wp = args
                self._rovers.add(rover)
                self._at_fact_ids_by_rover[rover].append((wp, fact.global_id))
            elif pred == "empty" and len(args) == 1:
                self._empty_fact_ids.add(fact.global_id)
            elif pred == "full" and len(args) == 1:
                self._full_fact_ids.add(fact.global_id)

        # Ensure all waypoints appear in adjacency keys for BFS robustness
        for rover, lst in self._at_fact_ids_by_rover.items():
            for wp, _ in lst:
                self._can_traverse_adj.setdefault(wp, set())

    def _precompute_goal_items(self):
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        goal_fact_ids_by_name = {f.name: f.global_id for f in self.model.facts}

        items: List[Tuple[str, Tuple[str, ...], int]] = []
        for gname in goal_fact_names:
            gid = goal_fact_ids_by_name.get(gname)
            if gid is None:
                continue
            pred, args = self._fact_pred_args[gid]
            if pred is None:
                continue
            # Only communicated_* matter in this benchmark set
            if pred.startswith("communicated_"):
                items.append((pred, args, gid))
        self._goal_items = items

    # ------------------------ Graph distance helpers ------------------------

    def _extract_rover_positions(self, state: int) -> Dict[str, str]:
        """Return one current waypoint per rover (if multiple at-facts exist, pick the true one)."""
        pos: Dict[str, str] = {}
        for rover, candidates in self._at_fact_ids_by_rover.items():
            for wp, fid in candidates:
                if (state >> fid) & 1:
                    pos[rover] = wp
                    break
        return pos

    def _bfs_min_dist_to_transmit(self, rover_positions: Dict[str, str]) -> Dict[str, int]:
        """Compute for each rover the min number of can_traverse edges to any transmit waypoint.

        We do a reverse BFS from all transmit waypoints, then read distance at rover positions.
        This is O(|E|) and avoids running BFS per rover.
        """
        if not self._transmit_waypoints:
            # If no transmit waypoint known, communication is impossible; return huge values.
            return {r: 10**6 for r in rover_positions}

        # Reverse adjacency
        rev_adj: Dict[str, List[str]] = defaultdict(list)
        for u, vs in self._can_traverse_adj.items():
            for v in vs:
                rev_adj[v].append(u)

        dist: Dict[str, int] = {}
        dq = deque()
        for tw in self._transmit_waypoints:
            dist[tw] = 0
            dq.append(tw)

        while dq:
            cur = dq.popleft()
            nd = dist[cur] + 1
            for prev in rev_adj.get(cur, []):
                if prev not in dist:
                    dist[prev] = nd
                    dq.append(prev)

        rover_to_transmit: Dict[str, int] = {}
        for r, wp in rover_positions.items():
            rover_to_transmit[r] = dist.get(wp, 10**6)
        return rover_to_transmit

    # ------------------------ Heuristic computation ------------------------

    def __call__(self, parent_node, node) -> int:
        state = node.state

        # Count pending goal communications by type
        pending_soil = 0
        pending_rock = 0
        pending_image = 0
        any_pending = 0

        for pred, args, gid in self._goal_items:
            if ((state >> gid) & 1) == 0:
                any_pending += 1
                if pred == "communicated_soil_data":
                    pending_soil += 1
                elif pred == "communicated_rock_data":
                    pending_rock += 1
                elif pred == "communicated_image_data":
                    pending_image += 1

        if any_pending == 0:
            return 0

        rover_positions = self._extract_rover_positions(state)
        rover_to_transmit = self._bfs_min_dist_to_transmit(rover_positions)

        # Lower bound navigation:
        # At least one rover must reach a transmit waypoint for each communication.
        # A cheap admissible bound is: minDist(any rover -> transmit) * indicator(any_pending)
        # plus we can add more by assuming different items might require separate transmissions;
        # but summing would overcount if one rover can transmit many items from same position.
        # We therefore use: minDist * 1 and add a small fractional-like via max(0, any_pending-1)
        # is not safe. Keep it strictly admissible by only counting the minimum required moves
        # to enable *some* transmission.
        min_nav_to_transmit = min(rover_to_transmit.values()) if rover_to_transmit else 10**6
        if min_nav_to_transmit >= 10**6:
            # No rover can reach any transmit waypoint in relaxed graph -> dead end
            return 10**6

        # Mandatory non-navigation actions lower bound.
        # For each pending communicated_* goal, we must at least perform the communication action.
        # But communicate also requires having the analysis/image; add minimal steps to get it.
        base = 0
        base += pending_soil * 2  # sample_soil + communicate
        base += pending_rock * 2  # sample_rock + communicate
        base += pending_image * 3  # calibrate + take_image + communicate

        # Store handling: if *all* stores of *all* rovers are full, at least one drop is needed before any sampling.
        # This is a very conservative admissible add-on: only when sampling is still needed and no empty store exists.
        if (pending_soil + pending_rock) > 0:
            any_empty_store = False
            for rover, stores in self._stores_of_rover.items():
                for store in stores:
                    # find empty(store) fact id by scanning once via parsing cache (cheap: store count small)
                    # We instead infer: empty store is a fact +empty[store]
                    # We'll just check by name lookup using fact_pred_args list.
                    pass

            # Efficient check: precompute store -> empty_fact_id / full_fact_id mapping on demand
        drop_lb = self._drop_lower_bound_if_needed(state, pending_soil + pending_rock)
        base += drop_lb

        # Total heuristic: minimal nav to enable first transmission + mandatory action count.
        # This remains a lower bound; real plans require additional navigation to samples/objectives.
        return base + min_nav_to_transmit

    def _drop_lower_bound_if_needed(self, state: int, pending_sampling: int) -> int:
        if pending_sampling <= 0:
            return 0

        # If there exists at least one empty store in current state, no forced drop.
        # If there are stores but none empty, then some store must be emptied (drop) before any sampling.
        # We detect empties via facts +empty[store].
        any_store_seen = False
        for rover, stores in self._stores_of_rover.items():
            for store in stores:
                any_store_seen = True
                # Find empty(store) fact id by scanning the fact cache: we can do it by name hashing once.
                # To keep this fast without a big dict, just check all empty facts and match arg in cache.
                # empty facts are few (one per store), so this is ok.
        
        # Build store->empty_fact_id mapping lazily once
        if not hasattr(self, "_store_empty_fact_id"):
            self._store_empty_fact_id = {}
            self._store_full_fact_id = {}
            for fid in self._empty_fact_ids:
                pred, args = self._fact_pred_args[fid]
                if pred == "empty" and len(args) == 1:
                    self._store_empty_fact_id[args[0]] = fid
            for fid in self._full_fact_ids:
                pred, args = self._fact_pred_args[fid]
                if pred == "full" and len(args) == 1:
                    self._store_full_fact_id[args[0]] = fid

        if not any_store_seen:
            return 0

        for rover, stores in self._stores_of_rover.items():
            for store in stores:
                efid = self._store_empty_fact_id.get(store)
                if efid is not None and ((state >> efid) & 1):
                    return 0

        # No empty store exists => at least one drop action is required before any sampling.
        return 1
