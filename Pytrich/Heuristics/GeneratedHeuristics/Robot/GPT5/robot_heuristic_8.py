from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Core idea (admissible lower bound):
    - Each package whose goal (in[obj, goal_room]) is not yet satisfied requires at least:
      - If currently held: 1 action (putdown). Otherwise: 2 actions (pickup + putdown).
    - Additionally, if at least one package cannot be finished without moving from the current
      robot location, then add at least 1 move action to the lower bound.
    - If moving is unavoidable AND all doors from the current room are closed, 
      add at least 1 open action (to be able to leave the current room through some door).

    Admissibility:
    - Pick/put counts per unsatisfied package are unavoidable in this domain (you must hold a
      package to move it and must put it down to achieve in[obj,goal]). If the package is already
      held, you need at least the putdown. Otherwise, you need pickup+putdown.
    - We add at most one “must-move” action as a global lower bound if some remaining package
      cannot be satisfied without leaving the current room. This cannot overestimate since any
      complete plan must include at least one such move in that situation.
    - We add at most one “must-open” action if leaving the current room is necessary and all its
      outgoing doors are closed. This is a safe lower bound: you must open at least one door to
      be able to move anywhere.

    Efficiency:
    - Uses bitwise checks for state membership.
    - Precomputes indices for relevant facts and caches goal/object/door data.

    Parameters:
    - robot_gpt5_heuristic_8: optional flag (kept for framework/config compatibility).
    """

    def __init__(self, model, robot_gpt5_heuristic_8: bool = False):
        super().__init__(model)
        self.model = model
        self.robot_gpt5_heuristic_8 = robot_gpt5_heuristic_8

        # Caches
        self.fact_by_name: Dict[str, object] = {}
        self.rloc_facts: List[object] = []  # Facts for predicate rloc[room]
        self.holding_facts_by_obj: Dict[str, object] = {}  # obj -> Fact for holding[obj]
        self.closed_fact_by_door: Dict[str, object] = {}  # door_id -> Fact for closed[door]
        self.doors_from_room: Dict[str, List[str]] = {}  # room -> list of door IDs leading out

        # Goal info: list of tuples (Fact, obj, goal_room)
        self.goal_facts_info: List[Tuple[object, str, str]] = []
        # Quick lookup: obj -> goal_room (from goals)
        self.goal_by_obj: Dict[str, str] = {}

        self._build_caches()

    ############################################
    # Core API expected by the planner
    ############################################
    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        # 1) Current robot room
        current_room = self._get_current_room(state_bitwise)
        # 2) Currently held object (if any)
        held_obj = self._get_held_object(state_bitwise)

        # 3) Collect unsatisfied goal facts: in[obj, goal_room]
        unsatisfied: List[Tuple[str, str]] = []  # list of (obj, goal_room)
        for fact, obj, goal_room in self.goal_facts_info:
            if not self._is_true(state_bitwise, fact.global_id):
                unsatisfied.append((obj, goal_room))

        if not unsatisfied:
            return 0

        # 4) Base count: per unsatisfied package -> at least 2 actions unless already held (then 1)
        base_count = 0
        for obj, _goal_room in unsatisfied:
            if held_obj == obj:
                base_count += 1  # only putdown remains (moves/opens not counted yet)
            else:
                base_count += 2  # pickup + putdown

        # 5) Movement lower bound: at least one move needed unless the only pending task
        #    is to putdown the currently held object in the current room
        move_lb = 0
        if len(unsatisfied) == 1:
            obj, goal_room = unsatisfied[0]
            if not (held_obj == obj and current_room is not None and goal_room == current_room):
                move_lb = 1
        else:
            # Multiple unsatisfied packages -> inevitably at least one move at some point
            move_lb = 1

        # 6) Open lower bound: if moving is unavoidable and all doors from current_room are closed
        open_lb = 0
        if move_lb == 1 and current_room is not None:
            outgoing_doors = self.doors_from_room.get(current_room, [])
            if outgoing_doors:
                all_closed = True
                for d in outgoing_doors:
                    closed_fact = self.closed_fact_by_door.get(d)
                    if closed_fact is None:
                        # If no explicit closed fact is modeled, treat as open by default (safe LB)
                        all_closed = False
                        break
                    if not self._is_true(state_bitwise, closed_fact.global_id):
                        # This door is not closed => can leave without opening
                        all_closed = False
                        break
                if all_closed:
                    open_lb = 1
            # If there are no outgoing door facts, we cannot infer a necessary open (leave as 0)

        return base_count + move_lb + open_lb

    ############################################
    # Internal helpers and cache builders
    ############################################
    def _build_caches(self):
        # Build name->Fact map
        self.fact_by_name = {f.name: f for f in self.model.facts}

        # Collect rloc, holding, closed, door predicates
        init_state = self.model.initial_state

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'rloc' and len(args) == 1:
                self.rloc_facts.append(f)
            elif pred == 'holding' and len(args) == 1:
                self.holding_facts_by_obj[args[0]] = f
            elif pred == 'closed' and len(args) == 1:
                self.closed_fact_by_door[args[0]] = f
            elif pred == 'door' and len(args) == 3:
                loc1, loc2, d = args
                self.doors_from_room.setdefault(loc1, []).append(d)
                # Note: door facts for both directions exist per problem; we add what we see.

        # Extract final goal facts (model.goals is a bitwise integer)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            gf = self.fact_by_name.get(gname)
            if gf is None:
                # Should not happen, but be robust
                continue
            pred, args = self._parse_fact_name(gf.name)
            if pred == 'in' and len(args) == 2:
                obj, goal_room = args
                self.goal_by_obj[obj] = goal_room
                self.goal_facts_info.append((gf, obj, goal_room))

        # Sanity: if no goals were found via model.goals, try to infer from goal_in in init (fallback)
        if not self.goal_facts_info:
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred == 'goal_in' and len(args) == 2:
                    if self._is_true(init_state, f.global_id):
                        obj, goal_room = args
                        # Construct the corresponding 'in[obj,goal_room]' fact name
                        in_name = f"+in[{obj},{goal_room}]"
                        in_fact = self.fact_by_name.get(in_name)
                        if in_fact is not None:
                            self.goal_by_obj[obj] = goal_room
                            self.goal_facts_info.append((in_fact, obj, goal_room))

    def _get_current_room(self, state_bitwise: int) -> Optional[str]:
        for f in self.rloc_facts:
            if self._is_true(state_bitwise, f.global_id):
                pred, args = self._parse_fact_name(f.name)
                if args:
                    return args[0]
        return None

    def _get_held_object(self, state_bitwise: int) -> Optional[str]:
        for obj, f in self.holding_facts_by_obj.items():
            if self._is_true(state_bitwise, f.global_id):
                return obj
        return None

    @staticmethod
    def _is_true(state_bitwise: int, global_id: int) -> bool:
        return ((state_bitwise >> global_id) & 1) == 1

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []


# Backward compatibility alias if the framework expects a symbol to register
HeuristicClass = RobotGPT5Heuristic8
