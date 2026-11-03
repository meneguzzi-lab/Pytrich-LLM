from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic18(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    Core domain insight
    -------------------
    The goal facts are exclusively of the form:
      - communicated_soil_data(?waypoint)
      - communicated_rock_data(?waypoint)
      - communicated_image_data(?objective, ?mode)

    In this domain, achieving any of those communications *necessarily* requires:
      1) having the corresponding data onboard:
         - have_soil_analysis(rover, waypoint) / have_rock_analysis(rover, waypoint)
         - have_image(rover, objective, mode)
      2) being at some waypoint p1 that is visible to the lander waypoint p2
         (visible(p1,p2) and at_lander(lander,p2)), and being available, channel_free.

    Also, navigation is the dominant cost driver. The training instances show many
    waypoints and dense visibility edges, while traversal is rover-specific.

    Heuristic design
    ----------------
    We compute a fast *lower bound* by counting unsatisfied goal communications,
    plus a relaxed movement estimate that uses precomputed shortest-path distances
    on the rover's can_traverse graph.

    For each unsatisfied goal item we add:
      - base cost 1  (at least one communicate_* action is required)
      - plus an additional cost to ensure the rover can reach a "transmit waypoint"
        (a waypoint that has visibility to the lander waypoint). This is lower bounded
        by min distance from any rover's current position to any transmit waypoint.

    This is admissible because:
      - each missing communication requires at least one communication operator
      - at least one rover must get to some valid transmit waypoint to communicate
      - shortest-path distances on can_traverse (edge cost 1 per navigate) lower-bound
        the number of navigate actions needed (methods may add visit/unvisit, but those
        only increase cost; ignoring them is a relaxation).

    Efficiency
    ----------
    - Uses bitwise checks for fact truth.
    - Precomputes:
        * lander waypoint(s)
        * transmit waypoints set (waypoints visible to lander waypoint)
        * all-pairs shortest distances for each rover over its can_traverse edges
          using BFS from each waypoint (sufficiently fast for ~130 waypoints).
    - For images/soil/rock we do NOT attempt expensive relaxed planning; we only
      count missing goal facts and add a coarse movement lower bound once.

    The heuristic is consistent enough in practice for A*/GBFS guidance; it is
    deliberately simple to avoid timeouts on large instances.
    """

    def __init__(self, model, rover_gpt52_heuristic_18: str = "rover_gpt52_heuristic_18"):
        super().__init__(model, rover_gpt52_heuristic_18)
        self.model = model

        # ---- Fact indexing ----
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # ---- Parse useful static relations from facts (once) ----
        self._at_fact_ids_by_rover: Dict[str, List[int]] = defaultdict(list)  # rover -> list of fact ids for at(rover, wp)
        self._can_traverse_edges_by_rover: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._visible_edges: Dict[str, Set[str]] = defaultdict(set)  # waypoint -> set(waypoint)
        self._at_lander_waypoints: Set[str] = set()

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == "at" and len(args) == 2:
                rover, wp = args
                self._at_fact_ids_by_rover[rover].append(fact.global_id)
            elif pred == "can_traverse" and len(args) == 3:
                rover, wp1, wp2 = args
                self._can_traverse_edges_by_rover[rover][wp1].add(wp2)
            elif pred == "visible" and len(args) == 2:
                w1, w2 = args
                self._visible_edges[w1].add(w2)
            elif pred == "at_lander" and len(args) == 2:
                _lander, wp = args
                self._at_lander_waypoints.add(wp)

        # In provided instances there is typically a single lander waypoint.
        # If multiple, any of them can serve as receiver; we allow all.
        self._transmit_waypoints: Set[str] = set()
        for lw in self._at_lander_waypoints:
            # Waypoints from which lander is visible
            for w1, neigh in self._visible_edges.items():
                if lw in neigh:
                    self._transmit_waypoints.add(w1)

        # ---- Precompute shortest path distances per rover between all waypoints ----
        # Distances count number of navigate edges (unit cost)
        self._dist: Dict[str, Dict[str, Dict[str, int]]] = {}
        for rover, adj in self._can_traverse_edges_by_rover.items():
            self._dist[rover] = self._all_pairs_bfs(adj)

        # ---- Cache goal facts and which are communications ----
        self._goal_fact_ids: List[int] = []
        self._goal_comm_fact_ids: List[int] = []
        for gname in self.model.state_explicit_repr(self.model.goals):
            fact = self._fact_by_name.get(gname)
            if fact is None:
                continue
            self._goal_fact_ids.append(fact.global_id)
            pred, _args = self._parse_fact_name(gname)
            if pred in ("communicated_soil_data", "communicated_rock_data", "communicated_image_data"):
                self._goal_comm_fact_ids.append(fact.global_id)

        # If there are no transmit waypoints (pathological), movement estimate is 0
        self._has_transmit = len(self._transmit_waypoints) > 0

    # -------- Heuristic API --------
    def __call__(self, node) -> int:
        state = node.state

        # Count unsatisfied communication goals
        missing_comm = 0
        for fid in self._goal_comm_fact_ids:
            if ((state >> fid) & 1) == 0:
                missing_comm += 1

        if missing_comm == 0:
            return 0

        # Base lower bound: each missing communication requires at least one communicate_* op.
        h = missing_comm

        # Add a movement lower bound: at least one rover must reach a transmit waypoint.
        # We add it once (not per goal) to keep it admissible and cheap.
        move_lb = self._min_rover_to_transmit_distance(state)
        if move_lb is not None:
            h += move_lb

        return h

    # -------- Helpers --------
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

    def _all_pairs_bfs(self, adj: Dict[str, Set[str]]) -> Dict[str, Dict[str, int]]:
        """Compute all-pairs shortest path distances (unit edges) via BFS from each node."""
        nodes = set(adj.keys())
        for u, vs in adj.items():
            nodes.update(vs)

        dist_all: Dict[str, Dict[str, int]] = {}
        for s in nodes:
            dist_all[s] = self._bfs_from(s, adj)
        return dist_all

    def _bfs_from(self, start: str, adj: Dict[str, Set[str]]) -> Dict[str, int]:
        from collections import deque

        q = deque([start])
        dist = {start: 0}
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, ()):  # directed edges
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _rover_positions(self, state: int) -> Dict[str, str]:
        """Return one current waypoint per rover (from at(rover,wp) facts true in state)."""
        pos: Dict[str, str] = {}
        for rover, fact_ids in self._at_fact_ids_by_rover.items():
            for fid in fact_ids:
                if (state >> fid) & 1:
                    # parse once from fact name (cheap: only for true at-facts)
                    pred, args = self._parse_fact_name(self.model.facts[fid].name)
                    if pred == "at" and len(args) == 2:
                        pos[rover] = args[1]
                        break
        return pos

    def _min_rover_to_transmit_distance(self, state: int) -> Optional[int]:
        if not self._has_transmit:
            return 0

        rover_pos = self._rover_positions(state)
        best: Optional[int] = None

        for rover, wp in rover_pos.items():
            dmap = self._dist.get(rover)
            if not dmap:
                continue
            from_wp = dmap.get(wp)
            if not from_wp:
                continue

            # If already at a transmit waypoint, distance 0.
            if wp in self._transmit_waypoints:
                return 0

            # Otherwise take min distance to any transmit waypoint reachable.
            local_best = None
            for tw in self._transmit_waypoints:
                d = from_wp.get(tw)
                if d is None:
                    continue
                if local_best is None or d < local_best:
                    local_best = d
                    if local_best == 0:
                        break
            if local_best is None:
                continue

            if best is None or local_best < best:
                best = local_best
                if best == 0:
                    return 0

        # If no rover can reach a transmit waypoint per can_traverse graph, we cannot infer a lower bound.
        # Return 0 (admissible).
        return best if best is not None else 0
