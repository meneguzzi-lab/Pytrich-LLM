"""Depots Domain Heuristic - DepotsAnthropicOpus46Heuristic14

Bottleneck: Crates must be moved between places via trucks with hoists as
location-bound resources. Moving a crate requires lift+drop (2 actions min)
if same place, or lift+load+drive+unload+drop (5 actions min) if different
place. Clearing surfaces creates cascading dependencies.

Admissible lower bounds:
1. Goal displacement cost: For each unsatisfied on(c,s) goal, compute minimum
   actions needed (0 if already true, 2 if crate and surface at same place,
   5 if at different places). Take sum over all unsatisfied goals that require
   distinct crate movements (each crate moved at most once in the count).
2. Task network cost: Count remaining primitive actions and estimate minimum
   decomposition depth for abstract tasks based on task type.
3. Clearing cost: For goals requiring a surface to be clear, count stacked
   crates that must be removed first.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic14(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_14"):
        super().__init__(name)
        # Fact ID maps
        self.on_fact_ids = {}      # (crate, surface) -> global_id
        self.at_fact_ids = {}      # (locatable, place) -> global_id
        self.in_fact_ids = {}      # (crate, truck) -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.lifting_fact_ids = {} # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id

        # Goal info
        self.goal_on_pairs = []    # list of (crate, surface, fact_global_id)

        # Object location map: object -> set of places it could be at
        self.object_places = {}    # object -> list of (place, fact_global_id)

        # Crate info
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()

        # For stacking: surface -> list of (crate, on_fact_id) where on(crate, surface)
        self.on_surface_facts = {}  # surface -> [(crate, global_id)]

        # Task type name patterns for cost estimation
        self.task_min_costs = {}

        # Precomputed: which places have hoists
        self.place_has_hoist = set()

    def _parse_fact_name(self, name: str):
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

    def _get_task_type(self, task):
        """Extract the task type from a task name."""
        name = task.name if hasattr(task, 'name') else str(task)
        # Task names look like: do_put_on[crate1,pallet1] or Drive[truck0,depot0,depot1]
        bracket_idx = name.find("[")
        if bracket_idx > 0:
            return name[:bracket_idx]
        return name

    def _get_task_args(self, task):
        """Extract args from task name."""
        name = task.name if hasattr(task, 'name') else str(task)
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            args_str = name[bracket_idx + 1:-1]
            return args_str.split(",") if args_str else []
        return []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model

        # Parse all facts and build lookup maps
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[1])
                # Track what's on each surface
                if args[1] not in self.on_surface_facts:
                    self.on_surface_facts[args[1]] = []
                self.on_surface_facts[args[1]].append((args[0], fact.global_id))
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = fact.global_id
                self.places.add(args[1])
                if args[0] not in self.object_places:
                    self.object_places[args[0]] = []
                self.object_places[args[0]].append((args[1], fact.global_id))
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = fact.global_id
                self.trucks.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = fact.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = fact.global_id
                self.hoists.add(args[0])
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = fact.global_id
                self.hoists.add(args[0])

        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_ids.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))

        # Determine which places have hoists (from initial state)
        initial_state = model.initial_state
        for hoist in self.hoists:
            for place_info in self.object_places.get(hoist, []):
                place, fid = place_info
                if (initial_state >> fid) & 1:
                    self.place_has_hoist.add(place)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _find_object_place(self, obj, state):
        """Find the current place of an object using bitwise state checks."""
        for place, fid in self.object_places.get(obj, []):
            if (state >> fid) & 1:
                return place
        return None

    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in some truck."""
        for (c, t), fid in self.in_fact_ids.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False

    def _count_crates_on_surface(self, surface, state):
        """Count how many crates are currently on this surface (stack height contribution)."""
        count = 0
        for crate, fid in self.on_surface_facts.get(surface, []):
            if (state >> fid) & 1:
                count += 1
        return count

    def _surface_is_clear(self, surface, state):
        """Check if surface is clear."""
        fid = self.clear_fact_ids.get(surface)
        if fid is not None:
            return (state >> fid) & 1
        return True

    def _compute_goal_displacement_cost(self, state):
        """Compute displacement cost for unsatisfied goal on-relations.
        
        For each goal on(c,s):
        - If already satisfied: 0
        - If crate is at same place as surface: 2 (lift + drop)
          But if surface is not clear, need to clear it first: +4 per crate on top
        - If crate is in a truck: 3 (unload + drop + need truck at right place)
        - If crate is at different place: 5 (lift + load + drive + unload + drop)
        
        This is admissible because each crate needs at least these many actions.
        """
        total_cost = 0
        crates_to_move = set()

        for crate, surface, fid in self.goal_on_pairs:
            if (state >> fid) & 1:
                # Goal already satisfied
                continue

            if crate in crates_to_move:
                # Already counted a move for this crate, skip to avoid double counting
                # But we should still count clearing cost for the surface
                pass
            
            crates_to_move.add(crate)

            # Find where crate currently is
            crate_place = self._find_object_place(crate, state)
            crate_in_truck = self._is_crate_in_truck(crate, state)

            # Find where surface currently is
            surface_place = self._find_object_place(surface, state)

            if crate_in_truck:
                # Need: unload + drop = 2, but also need truck at right place
                # Minimum: 2 actions (unload + drop) if truck already at surface place
                # Otherwise: 3 (drive + unload + drop)
                cost = 2
                if surface_place is not None:
                    # Check if any truck with this crate is at surface_place
                    truck_at_dest = False
                    for (c, t), ifid in self.in_fact_ids.items():
                        if c == crate and (state >> ifid) & 1:
                            t_place = self._find_object_place(t, state)
                            if t_place == surface_place:
                                truck_at_dest = True
                                break
                    if not truck_at_dest:
                        cost = 3  # drive + unload + drop
                total_cost += cost
            elif crate_place is not None and surface_place is not None:
                if crate_place == surface_place:
                    # Same place: lift + drop = 2
                    total_cost += 2
                else:
                    # Different place: lift + load + drive + unload + drop = 5
                    total_cost += 5
            elif crate_place is not None:
                # Surface place unknown (maybe surface is a crate not yet placed)
                total_cost += 4  # Conservative: lift + load + drive + unload + drop - 1
            else:
                # Crate being lifted by hoist
                # Need at minimum: drop (1) or load+drive+unload+drop (4)
                total_cost += 1

        return total_cost

    def _compute_clearing_cost(self, state):
        """Estimate cost of clearing surfaces that need to be cleared for goals.
        
        For each goal on(c,s) where s is not clear and on(c,s) is not already true,
        we need to clear s first. Each crate on s costs at least 4 actions to remove
        (lift + load + nop + nop, simplified to: lift + load = 2 minimum per crate).
        
        Admissible because clearing each crate on a surface requires at least 2 actions.
        """
        surfaces_to_clear = set()
        for crate, surface, fid in self.goal_on_pairs:
            if (state >> fid) & 1:
                continue
            # Need to place crate on surface, so surface must be clear
            if not self._surface_is_clear(surface, state):
                surfaces_to_clear.add(surface)

        total_clearing = 0
        for surface in surfaces_to_clear:
            # Count crates on this surface
            n_crates = self._count_crates_on_surface(surface, state)
            # Each blocking crate needs at least 2 actions (lift + load/drop)
            total_clearing += n_crates * 2

        return total_clearing

    def _compute_task_network_cost(self, task_network):
        """Estimate minimum actions from remaining task network.
        
        For each task in the network:
        - Primitive action (operator): costs 1
        - do_put_on: minimum 0 (if already done via m0_do_put_on/nop)
          but typically 2-5 actions
        - do_clear: minimum 0 (nop) or 4 (lift+load+get_truck+nop)
        - do_get_truck: minimum 0 (nop) or 1 (drive)
        - do_lift_crate: minimum 1 (lift or unload)
        - do_load_truck: minimum 2 (lift+load, ignoring gets)
        - do_unload_truck: minimum 2 (unload+drop)
        
        We use conservative (admissible) minimum decomposition sizes.
        """
        total = 0
        for task in task_network:
            # Check if it's an operator (primitive)
            if hasattr(task, 'add_effects'):
                # It's an operator/primitive action
                # nop costs 0 real work, others cost 1
                task_name = task.name if hasattr(task, 'name') else ''
                if 'nop' in task_name:
                    total += 0
                else:
                    total += 1
            elif hasattr(task, 'decompositions'):
                # It's an abstract task
                task_type = self._get_task_type(task)
                if task_type == '__top':
                    # Top-level task, don't add cost directly
                    total += 0
                elif 'do_put_on' in task_type:
                    # Minimum is 0 (m0 = nop if already on)
                    # But typical minimum for non-trivial is 2
                    total += 0  # Admissible: could be nop
                elif 'do_clear' in task_type:
                    total += 0  # Could be nop
                elif 'do_get_truck' in task_type:
                    total += 0  # Could be nop
                elif 'do_lift_crate' in task_type:
                    total += 1  # Minimum 1 action (Lift or Unload)
                elif 'do_load_truck' in task_type:
                    total += 2  # Minimum Lift + Load
                elif 'do_unload_truck' in task_type:
                    total += 2  # Minimum Unload + Drop
                else:
                    total += 0
        return total

    def _compute_unsatisfied_goals(self, state):
        """Count unsatisfied goal on-relations as a simple lower bound."""
        count = 0
        for crate, surface, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                count += 1
        return count

    def _compute_goal_stability_cost(self, state):
        """Detect goals that are currently true but will need to be undone.
        
        A goal on(c, s) is 'false-trivial' if it's currently true but s is
        needed as a destination for another crate (meaning c will need to be
        moved off s first, then back on). This is expensive: costs at least 4
        extra actions (lift c off, put c somewhere, then later put c back).
        
        Also: if a crate c is currently on surface s (goal true), but some other
        crate c2 is on top of c and needs to be moved, c might need to be
        temporarily displaced.
        """
        # Build set of surfaces that are goal destinations
        goal_surfaces = {}  # surface -> crate that should end up on it
        for crate, surface, fid in self.goal_on_pairs:
            goal_surfaces[surface] = crate

        # Check for crates that are goal-satisfied but have stuff on top
        # that needs to move
        cost = 0
        for crate, surface, fid in self.goal_on_pairs:
            if (state >> fid) & 1:
                # Goal is currently true. Check if crate itself is a surface
                # for another goal, and is NOT clear
                if crate in goal_surfaces:
                    goal_crate_for_c = goal_surfaces[crate]
                    on_fid = self.on_fact_ids.get((goal_crate_for_c, crate))
                    if on_fid is not None and not ((state >> on_fid) & 1):
                        # crate needs goal_crate_for_c on top, but it's not there yet
                        # If crate is not clear, whatever is on top must be moved
                        if not self._surface_is_clear(crate, state):
                            # Something else is on crate that needs to move
                            cost += 2  # At least lift + load/drop to clear
        return cost

    def _compute(self, state, task_network):
        """Main heuristic computation using max of multiple admissible bounds."""
        # Term 1: Goal displacement cost
        displacement = self._compute_goal_displacement_cost(state)

        # Term 2: Task network minimum cost
        tn_cost = self._compute_task_network_cost(task_network)

        # Term 3: Clearing cost for surfaces that need to be cleared
        clearing = self._compute_clearing_cost(state)

        # Term 4: Goal stability cost (false-trivial detection)
        stability = self._compute_goal_stability_cost(state)

        # Term 5: Simple unsatisfied goal count (very admissible baseline)
        unsatisfied = self._compute_unsatisfied_goals(state)

        # Combine: max of admissible terms for strongest bound
        # displacement + clearing is admissible because they count different actions:
        # displacement counts actions to move the target crate,
        # clearing counts actions to remove blocking crates
        # However, there might be overlap, so we use max for safety
        
        h_goal = max(displacement, clearing + unsatisfied, displacement + stability)
        h_tasks = tn_cost
        
        return max(h_goal, h_tasks)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
