from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional, Set

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic7(Heuristic):
    """Domain-dependent heuristic for the Robot HDDL domain.

    Domain recap (grounded):
      - Robot is in exactly one room: +rloc[room]
      - Packages are in rooms: +in[obj,room] or robot holds one: +holding[obj]
      - One gripper: +armempty means holding none.
      - Doors connect rooms: +door[loc1,loc2,door]
      - Doors can be closed: +closed[door], and must be opened before moving through.
      - Goal facts are +in[obj,goal_room] for many objects.

    HTN structure:
      - Root task achieve-goals loops:
          * if some object is in current room and not already at its goal -> pickup(obj) then release
          * otherwise it can (move_abstract) or (open_abstract) and recurse
      - release loops until it can putdown an already-held object when at its goal room.

    Key planning challenges:
      1) Navigation with closed doors: reaching rooms may require opening doors en route.
      2) Single-object carrying: each misplaced package costs at least one pickup + one putdown.
      3) The HTN can waste time moving/opening without progressing package placement.

    Heuristic idea (admissible lower bound):
      h(state, tn) = max( h_tasks(tn), h_packages(state) )

      - h_tasks(tn): a cheap HTN-aware lower bound: count of *unavoidable* primitive actions
        already implied by the current task network, but only when their preconditions are
        already satisfied (so they truly can be executed without extra cost).
        This gives guidance when the search has committed to a refinement.

      - h_packages(state): a state-based lower bound on remaining work to satisfy goal facts:
          For each goal +in[o,g]:
            * if already true -> 0
            * else at least 2 actions (pickup+putdown) are needed for that object.
          Additionally, we add a navigation lower bound to reach *some* relevant room:
            * If holding an object not yet at its goal, must at least reach its goal room.
            * Else must at least reach a room containing a misplaced object.
          Navigation lower bound is computed as shortest path in the room graph with edge cost:
            move costs 1, and if the door on that edge is currently closed, at least 1 open is
            required before traversing it. So each traversal costs (1 + closed?).

    Efficiency:
      - Precompute room graph from static +door facts.
      - Precompute goal mapping per object from goal facts.
      - Each heuristic call runs a 0-1-2 BFS (Dijkstra with small integer weights) from current
        room over <= 50 rooms, which is cheap.

    Admissibility:
      - Package cost: using 2 per misplaced object is a strict lower bound.
      - Navigation: shortest path with (move+maybe-open) is a lower bound because any real plan
        must move and open at least those doors to traverse them.
      - Taking max with HTN-implied-available actions remains a lower bound.

    Note: This heuristic intentionally ignores beneficial synergies (carrying object while moving
    towards another) to preserve admissibility.
    """

    def __init__(self, model, robot_gpt52_heuristic_7: str = "robot_gpt52_heuristic_7"):
        super().__init__(model, robot_gpt52_heuristic_7)
        self.model = model

        # --- Index facts we care about ---
        self._fact_by_name: Dict[str, int] = {}
        for f in self.model.facts:
            self._fact_by_name[f.name] = f.global_id

        # Parse goals: +in[obj,room]
        self._goals_in: List[Tuple[str, str, int]] = []  # (obj, room, fact_id)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "in" and len(args) == 2 and n.startswith("+"):
                fid = self._fact_by_name.get(n)
                if fid is not None:
                    self._goals_in.append((args[0], args[1], fid))

        # Map each object -> goal room (unique in this benchmark)
        self._goal_room_of: Dict[str, str] = {}
        for o, r, _ in self._goals_in:
            self._goal_room_of[o] = r

        # Precompute door edges from static facts: +door[loc1,loc2,d]
        # We'll store adjacency: room -> list of (neighbor_room, door_name)
        self._adj: Dict[str, List[Tuple[str, str]]] = {}
        for f in self.model.facts:
            if not f.name.startswith("+"):
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred == "door" and len(args) == 3:
                loc1, loc2, d = args
                self._adj.setdefault(loc1, []).append((loc2, d))

        # Cache fact-ids for faster state checks
        self._rloc_facts: List[Tuple[str, int]] = []
        self._armempty_id: Optional[int] = self._fact_by_name.get("+armempty[]")
        # armempty in this domain has no args, but grounding format could be +armempty[] or +armempty
        if self._armempty_id is None:
            self._armempty_id = self._fact_by_name.get("+armempty")

        for f in self.model.facts:
            if not f.name.startswith("+"):
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred == "rloc" and len(args) == 1:
                self._rloc_facts.append((args[0], f.global_id))

        self._closed_door_fact_id: Dict[str, int] = {}  # door_name -> fact_id of +closed[door]
        for f in self.model.facts:
            if not f.name.startswith("+"):
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred == "closed" and len(args) == 1:
                self._closed_door_fact_id[args[0]] = f.global_id

        # Facts for holding and in: build indices to avoid scanning all facts each call
        self._holding_fact_id: Dict[str, int] = {}  # obj -> fact_id of +holding[obj]
        self._in_fact_id: Dict[Tuple[str, str], int] = {}  # (obj, room) -> fact_id of +in[obj,room]
        for f in self.model.facts:
            if not f.name.startswith("+"):
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred == "holding" and len(args) == 1:
                self._holding_fact_id[args[0]] = f.global_id
            elif pred == "in" and len(args) == 2:
                self._in_fact_id[(args[0], args[1])] = f.global_id

    # ------------------------ core API ------------------------
    def __call__(self, node) -> int:
        state = node.state
        tn = getattr(node, "task_network", [])

        # If already at goal+empty tn, heuristic 0
        if self.model.goal_reached(state, tn):
            return 0

        h_pkg = self._h_packages_and_navigation(state)
        h_tn = self._h_task_network_lower_bound(state, tn)
        return max(h_pkg, h_tn)

    # ------------------------ helpers ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
            # Handle 0-arity facts sometimes printed without brackets
            return name2, []
        return None, []

    def _is_true(self, state: int, fact_id: int) -> bool:
        return ((state >> fact_id) & 1) == 1

    def _current_room(self, state: int) -> Optional[str]:
        for room, fid in self._rloc_facts:
            if self._is_true(state, fid):
                return room
        return None

    def _held_object(self, state: int) -> Optional[str]:
        for obj, fid in self._holding_fact_id.items():
            if self._is_true(state, fid):
                return obj
        return None

    # ---- package + navigation lower bound ----
    def _h_packages_and_navigation(self, state: int) -> int:
        # Count misplaced goal objects
        misplaced: List[Tuple[str, str]] = []  # (obj, goal_room)
        for obj, goal_room, goal_fact_id in self._goals_in:
            if not self._is_true(state, goal_fact_id):
                misplaced.append((obj, goal_room))

        if not misplaced:
            return 0

        # Each misplaced object needs at least pickup+putdown => 2
        base = 2 * len(misplaced)

        # Add minimal navigation needed to make progress on at least one object
        cur = self._current_room(state)
        if cur is None:
            return base

        held = self._held_object(state)
        if held is not None:
            g = self._goal_room_of.get(held)
            if g is None:
                # If holding something not in goals, ignore navigation
                return base
            # If already at goal room, navigation 0; else distance to goal room
            nav = self._min_cost_to_room(state, cur, g)
            return base + nav

        # Not holding: must reach some room that contains a misplaced object to pick it up.
        target_rooms: Set[str] = set()
        for obj, _g in misplaced:
            # Find current location of obj from in-facts among rooms. We avoid scanning all facts
            # by checking candidate rooms via in_fact_id keys for this object.
            # Worst-case still manageable; but we can just scan the in_fact_id dict for matching obj.
            for (o, r), fid in self._in_fact_id.items():
                if o == obj and self._is_true(state, fid):
                    target_rooms.add(r)
                    break

        if not target_rooms:
            return base

        nav = self._min_cost_to_any_room(state, cur, target_rooms)
        return base + nav

    def _edge_cost(self, state: int, door_name: str) -> int:
        # move cost 1; if door closed, need open at least once before traverse => +1
        closed_id = self._closed_door_fact_id.get(door_name)
        if closed_id is not None and self._is_true(state, closed_id):
            return 2
        return 1

    def _min_cost_to_room(self, state: int, start: str, goal: str) -> int:
        if start == goal:
            return 0
        # Dijkstra with small weights (1 or 2)
        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        inq = {start}
        # We use a simple SPFA-like loop since graph is tiny.
        while q:
            u = q.popleft()
            inq.discard(u)
            du = dist[u]
            for v, d in self._adj.get(u, []):
                w = self._edge_cost(state, d)
                nd = du + w
                if nd < dist.get(v, 10**9):
                    dist[v] = nd
                    if v not in inq:
                        q.append(v)
                        inq.add(v)
        return dist.get(goal, 0)

    def _min_cost_to_any_room(self, state: int, start: str, goals: Set[str]) -> int:
        if start in goals:
            return 0
        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        inq = {start}
        best = 10**9
        while q:
            u = q.popleft()
            inq.discard(u)
            du = dist[u]
            if du >= best:
                continue
            if u in goals:
                best = min(best, du)
                continue
            for v, d in self._adj.get(u, []):
                w = self._edge_cost(state, d)
                nd = du + w
                if nd < dist.get(v, 10**9) and nd < best:
                    dist[v] = nd
                    if v not in inq:
                        q.append(v)
                        inq.add(v)
        return 0 if best == 10**9 else best

    # ---- HTN-aware lower bound ----
    def _h_task_network_lower_bound(self, state: int, task_network) -> int:
        """Lower bound derived from current task network.

        We count primitive operators already present in the task network that are
        *currently applicable* in the current state. These are unavoidable next steps
        after the refinement choices already made, and counting only applicable ones
        keeps the estimate admissible (we don't assume extra actions to enable them).

        Additionally, if the first task is an abstract task, we add 0 (admissible).
        """
        if not task_network:
            return 0

        cost = 0
        # Only inspect a prefix for speed; long TNs in this domain are rare, but keep bounded.
        for t in task_network[:25]:
            # Primitive operator
            if hasattr(t, "applicable") and hasattr(t, "cost") and hasattr(t, "pos_precons"):
                try:
                    if t.applicable(state):
                        cost += int(getattr(t, "cost", 1))
                except Exception:
                    pass
        return cost
