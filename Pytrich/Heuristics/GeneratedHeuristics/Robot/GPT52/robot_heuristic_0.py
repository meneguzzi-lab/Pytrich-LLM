from collections import deque
from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic0(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    Domain summary (after grounding):
      - State predicates: rloc(room), armempty, holding(pkg), in(pkg,room), closed(door), door(room,room,door)
      - Goals: (in obj goal_room) for many objects.
      - Primitive actions (unit costs typically):
          pickup(obj,loc): requires armempty & rloc(loc) & in(obj,loc)
          putdown(obj,loc): requires rloc(loc) & holding(obj) & goal_in(obj,loc)
          move(loc1,loc2,d): requires rloc(loc1) & door(loc1,loc2,d) & not closed(d)
          open(loc1,loc2,d): requires rloc(loc1) & door(loc1,loc2,d) & closed(d)

    Key challenge/structure:
      - Each misplaced package requires at least one pickup and one putdown.
      - Robot can carry at most one package (armempty/holding).
      - Navigation occurs through doors; some are closed and must be opened before moving.

    Heuristic idea (admissible lower bound):
      h(state, tn) = max( 2 * (#misplaced packages),  minimal_nav_cost_to_handle_at_least_one_remaining_package )

    Where:
      - 2*misplaced is a hard lower bound because each misplaced package must be picked up and later put down
        (goals are only 'in', and pickup removes 'in', putdown adds it at goal).
      - minimal_nav_cost_to_handle_one computes a *single-package* optimistic cost:
          if holding a misplaced package: need to reach its goal room and putdown
          else (armempty): need to reach some misplaced package's current room, pick it up,
               reach its goal room, and put it down
        Navigation cost uses shortest-path distances in the room graph with edge cost 1 for move,
        plus +1 if the traversed door is currently closed (optimistically: open then move).

    Combining with max keeps admissibility: both components are lower bounds on the true remaining cost.

    Efficiency notes:
      - Parses and indexes facts once in __init__.
      - Uses bitwise checks only for truth values.
      - Precomputes room-to-room distances from current rloc using BFS/Dijkstra on the fly (small graph: <= 50 rooms).

    This heuristic is domain-dependent (uses door graph, closed doors, carrying constraint, goal_in mapping)
    and is designed to scale to large instances with many packages.
    """

    def __init__(self, model, robot_gpt52_heuristic_0: bool = True):
        super().__init__(model)
        self.robot_gpt52_heuristic_0 = robot_gpt52_heuristic_0

        # --- Fact parsing/indexing ---
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # Maps
        self._goal_room_of: Dict[str, str] = {}          # pkg -> goal room
        self._in_fact_gid: Dict[Tuple[str, str], int] = {}  # (pkg,room) -> fact global_id
        self._holding_fact_gid: Dict[str, int] = {}      # pkg -> fact global_id
        self._rloc_fact_gid: Dict[str, int] = {}         # room -> fact global_id
        self._closed_fact_gid: Dict[str, int] = {}       # door -> fact global_id
        self._armempty_gid: Optional[int] = None

        # Door graph: room -> List[(neighbor_room, door_id)]
        self._adj: Dict[str, List[Tuple[str, str]]] = {}

        # Extract goal_in mapping from model.goals (bitset)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == "goal_in" and len(args) == 2:
                pkg, room = args
                self._goal_room_of[pkg] = room

        # Index all facts for quick bit tests
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "armempty" and len(args) == 0:
                self._armempty_gid = fact.global_id
            elif pred == "rloc" and len(args) == 1:
                self._rloc_fact_gid[args[0]] = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self._holding_fact_gid[args[0]] = fact.global_id
            elif pred == "in" and len(args) == 2:
                self._in_fact_gid[(args[0], args[1])] = fact.global_id
            elif pred == "closed" and len(args) == 1:
                self._closed_fact_gid[args[0]] = fact.global_id

        # Build static door adjacency (door predicate is static in given benchmarks)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "door" and len(args) == 3:
                r1, r2, d = args
                self._adj.setdefault(r1, []).append((r2, d))

        # Precompute list of packages we care about (those with a goal)
        self._packages: List[str] = list(self._goal_room_of.keys())

    def __call__(self, state: int, task_network) -> int:
        # If no remaining tasks, heuristic should be 0 (the planner will check goal reached elsewhere)
        if task_network is not None and len(task_network) == 0:
            # Still could have state goals unmet, but in this domain HTN drives to achieve goals.
            # Return an admissible estimate for state goals anyway.
            pass

        # Find current robot room
        cur_room = self._current_room(state)
        if cur_room is None:
            return 0

        # Determine which packages are misplaced w.r.t. goal rooms.
        misplaced: List[str] = []
        pkg_loc: Dict[str, Optional[str]] = {}
        holding_pkg: Optional[str] = None

        for p in self._packages:
            loc = self._current_pkg_room(state, p)
            pkg_loc[p] = loc

            # check holding
            hgid = self._holding_fact_gid.get(p)
            if hgid is not None and ((state >> hgid) & 1):
                holding_pkg = p
                # If holding, it's certainly not 'in goal room' yet.
                misplaced.append(p)
                continue

            goal_r = self._goal_room_of.get(p)
            if goal_r is None:
                continue

            # If already in goal room, ok.
            if loc == goal_r:
                continue

            # Otherwise misplaced.
            misplaced.append(p)

        if not misplaced:
            return 0

        # Lower bound 1: at least pickup+putdown for each misplaced package.
        # (Even if already holding, pickup for that package is already done in the past;
        #  counting 2 per misplaced might overestimate if holding. Keep it admissible by
        #  discounting holding case: holding needs at least 1 putdown; not holding needs at least 2.)
        lb_actions = 0
        for p in misplaced:
            if holding_pkg == p:
                lb_actions += 1  # only putdown remains for this pkg
            else:
                lb_actions += 2

        # Lower bound 2: optimistic cost to finish at least one remaining package, including navigation.
        nav_lb = self._min_cost_to_finish_one(state, cur_room, misplaced, holding_pkg, pkg_loc)

        # Combine with max to remain admissible.
        h = max(lb_actions, nav_lb)
        return int(h)

    # -------------------- Core computations --------------------

    def _min_cost_to_finish_one(
        self,
        state: int,
        cur_room: str,
        misplaced: List[str],
        holding_pkg: Optional[str],
        pkg_loc: Dict[str, Optional[str]],
    ) -> int:
        # Precompute shortest paths from current room to all rooms with door-open costs.
        dist_from_cur = self._dijkstra_room_costs(state, cur_room)

        best = None

        if holding_pkg is not None:
            # Need to go to its goal room and putdown.
            goal_r = self._goal_room_of.get(holding_pkg)
            if goal_r is None:
                return 0
            d = dist_from_cur.get(goal_r, None)
            if d is None:
                # unreachable in current static graph; still return pure putdown lower bound
                return 1
            best = d + 1  # putdown
        else:
            # Armempty or unknown: pick any misplaced package, go to its room, pickup, then to goal, putdown.
            for p in misplaced:
                p_room = pkg_loc.get(p)
                goal_r = self._goal_room_of.get(p)
                if p_room is None or goal_r is None:
                    continue

                d_to_pkg = dist_from_cur.get(p_room)
                if d_to_pkg is None:
                    continue

                # From package room to goal room: compute distances rooted at p_room.
                dist_from_pkg = self._dijkstra_room_costs(state, p_room)
                d_pkg_to_goal = dist_from_pkg.get(goal_r)
                if d_pkg_to_goal is None:
                    continue

                cost = d_to_pkg + 1 + d_pkg_to_goal + 1  # pickup + putdown
                if best is None or cost < best:
                    best = cost

            if best is None:
                # No candidate reachable: at least one pickup+putdown is still needed
                best = 2

        return int(best)

    def _dijkstra_room_costs(self, state: int, start_room: str) -> Dict[str, int]:
        """Compute optimistic navigation costs between rooms.

        Edge traversal (r -> r') via door d has cost:
          1 for move + (1 if closed(d) in current state else 0)
        This is admissible because opening a closed door is required at most once before moving through it.
        """
        # Since costs are 1 or 2, we can do a 0-1-2 BFS with deque by pushing 1-cost edges left-ish.
        # For simplicity and small graphs, use standard Dijkstra with deque buckets.

        # Buckets for small integer costs: use deque-based Dial's algorithm with max edge cost 2.
        dist: Dict[str, int] = {start_room: 0}
        buckets: List[deque] = [deque() for _ in range(3)]
        buckets[0].append(start_room)
        current_distance = 0
        active = 1

        def edge_cost(door_id: str) -> int:
            gid = self._closed_fact_gid.get(door_id)
            if gid is None:
                return 1
            return 2 if ((state >> gid) & 1) else 1

        while active:
            # Find next non-empty bucket
            bidx = current_distance % 3
            while not buckets[bidx]:
                current_distance += 1
                bidx = current_distance % 3
                # If all buckets empty, stop
                if current_distance > max(dist.values()) + 2:
                    # quick check, but keep safe: scan
                    if all(len(b) == 0 for b in buckets):
                        return dist

            room = buckets[bidx].popleft()
            # Might be an outdated entry
            if dist.get(room, None) != current_distance:
                continue

            for nbr, d in self._adj.get(room, []):
                c = edge_cost(d)
                nd = current_distance + c
                old = dist.get(nbr)
                if old is None or nd < old:
                    dist[nbr] = nd
                    buckets[nd % 3].append(nbr)

            # update active
            active = 0
            for b in buckets:
                if b:
                    active = 1
                    break

        return dist

    # -------------------- Bitwise state queries --------------------

    def _current_room(self, state: int) -> Optional[str]:
        # rloc is exclusive; scan rooms we know
        for room, gid in self._rloc_fact_gid.items():
            if (state >> gid) & 1:
                return room
        return None

    def _current_pkg_room(self, state: int, pkg: str) -> Optional[str]:
        # Find any in[pkg,room] true; scan only rooms appearing in in-facts for this pkg.
        # (We did not store per-pkg lists, but scanning all in-facts for a pkg is still fine:
        #  total rooms <= 50 and packages <= 100 in benchmarks; facts count moderate.)
        for (p, r), gid in self._in_fact_gid.items():
            if p != pkg:
                continue
            if (state >> gid) & 1:
                return r
        return None

    # -------------------- Fact parsing --------------------

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
