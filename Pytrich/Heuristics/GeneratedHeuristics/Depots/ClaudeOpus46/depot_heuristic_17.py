# Depots Domain Heuristic
# Bottleneck: Hoists are serializing resources - one crate at a time per hoist.
# Clearing crate stacks and transporting crates between places dominate plan length.
#
# Admissible lower bounds:
# T1: For each unsatisfied 'on' goal, minimum actions needed considering crate/surface location
#     and whether clearing is required. Lower bound because each goal needs at least these actions.
# T2: Task network estimate - each remaining abstract task has a minimum decomposition cost
#     based on its type and current state. Lower bound because tasks must be decomposed.
# T3: Transport displacement - crates needing to change places require drive sequences.
#     Lower bound because each such crate needs at least load+drive+unload+drop.

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.HDDL.operator import Operator as OperatorClass
from Pytrich.HDDL.abstract_task import AbstractTask as AbstractTaskClass


class DepotsAnthropicOpus46Heuristic17(Heuristic):
    """Domain-dependent heuristic for Depots HTN planning.
    
    Uses multi-component max of:
    - Goal-based displacement cost (how far each crate is from its goal position)
    - Task network weighted estimate (state-aware per-task cost)
    - Clearing cost for blocked surfaces
    """

    def __init__(self, name="depot_anthropicopus46_heuristic_17"):
        super().__init__(name)
        # Fact index maps built during initialize
        self.on_fact_id = {}       # (crate, surface) -> global_id
        self.at_fact_id = {}       # (locatable, place) -> global_id
        self.in_fact_id = {}       # (crate, truck) -> global_id
        self.clear_fact_id = {}    # surface -> global_id
        self.lifting_fact_id = {}  # (hoist, crate) -> global_id
        self.available_fact_id = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on = {}          # crate -> target_surface for 'on' goals
        self.goal_on_fact_ids = {} # crate -> global_id of the goal on fact
        
        # Object classification
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # Location info: which place each surface/crate starts at
        # We'll track static pallet locations
        self.pallet_place = {}     # pallet -> place
        self.hoist_place = {}      # hoist -> place (hoists don't move)
        
        # Task type name prefixes for classification
        self.task_type_cache = {}  # task global_id -> estimated min cost

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name like +on[crate0,pallet1] into (predicate, [args])"""
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

    def _classify_object(self, obj_name: str):
        """Classify object by its name prefix."""
        if obj_name.startswith("crate"):
            return "crate"
        elif obj_name.startswith("pallet"):
            return "pallet"
        elif obj_name.startswith("truck"):
            return "truck"
        elif obj_name.startswith("hoist"):
            return "hoist"
        elif obj_name.startswith("depot") or obj_name.startswith("distributor"):
            return "place"
        return "unknown"

    def _get_task_type(self, task_name: str):
        """Extract task type from grounded task name."""
        # Task names look like: do_put_on[crate1,pallet1] or Drive[truck0,depot0,depot1]
        bracket_idx = task_name.find("[")
        if bracket_idx > 0:
            return task_name[:bracket_idx]
        return task_name

    def _get_task_args(self, task_name: str):
        """Extract arguments from grounded task name."""
        bracket_idx = task_name.find("[")
        if bracket_idx > 0 and task_name.endswith("]"):
            args_str = task_name[bracket_idx + 1:-1]
            return args_str.split(",") if args_str else []
        return []

    def initialize(self, model: Model, initial_node: HTNNode):
        """Build all index maps and goal structures."""
        self.model = model
        
        # Build fact index maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_id[(args[0], args[1])] = f.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[0])  # crates are surfaces too
                self.surfaces.add(args[1])
            elif pred == "at" and len(args) == 2:
                self.at_fact_id[(args[0], args[1])] = f.global_id
                obj_type = self._classify_object(args[0])
                if obj_type == "crate":
                    self.crates.add(args[0])
                elif obj_type == "pallet":
                    self.pallets.add(args[0])
                elif obj_type == "truck":
                    self.trucks.add(args[0])
                elif obj_type == "hoist":
                    self.hoists.add(args[0])
                    self.hoist_place[args[0]] = args[1]
                self.places.add(args[1])
                if obj_type == "pallet":
                    self.pallet_place[args[0]] = args[1]
            elif pred == "in" and len(args) == 2:
                self.in_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_id[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_id[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_id[args[0]] = f.global_id
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                crate = args[0]
                surface = args[1]
                self.goal_on[crate] = surface
                fid = self.on_fact_id.get((crate, surface))
                if fid is not None:
                    self.goal_on_fact_ids[crate] = fid
        
        # Build place sets for hoists (hoists are static)
        self.place_hoists = {}  # place -> list of hoist names
        for h, p in self.hoist_place.items():
            if p not in self.place_hoists:
                self.place_hoists[p] = []
            self.place_hoists[p].append(h)
        
        # Precompute: for each crate, which 'on' facts exist (for stack detection)
        self.crate_on_facts = {}  # crate -> list of (crate, surface, global_id)
        for (c, s), fid in self.on_fact_id.items():
            if c not in self.crate_on_facts:
                self.crate_on_facts[c] = []
            self.crate_on_facts[c].append((c, s, fid))
        
        # Precompute: for each surface, which crates could be on it
        self.surface_on_by = {}  # surface -> list of (crate, global_id)
        for (c, s), fid in self.on_fact_id.items():
            if s not in self.surface_on_by:
                self.surface_on_by[s] = []
            self.surface_on_by[s].append((c, fid))
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Compute heuristic for a search node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        """Main heuristic computation: max of multiple admissible terms."""
        t1 = self._goal_displacement_cost(state)
        t2 = self._task_network_estimate(state, task_network)
        return max(t1, t2)

    def _is_fact_true(self, state, global_id):
        """Check if a fact is true in the state using bitwise ops."""
        return (state >> global_id) & 1

    def _find_crate_location(self, state, crate):
        """Find which place a crate is at, or None if not at any place (in truck or being lifted)."""
        for (obj, place), fid in self.at_fact_id.items():
            if obj == crate and self._is_fact_true(state, fid):
                return place
        return None

    def _find_crate_on(self, state, crate):
        """Find what surface a crate is currently on, or None."""
        if crate in self.crate_on_facts:
            for (c, s, fid) in self.crate_on_facts[crate]:
                if self._is_fact_true(state, fid):
                    return s
        return None

    def _find_what_is_on(self, state, surface):
        """Find what crate is currently on a surface, or None."""
        if surface in self.surface_on_by:
            for (c, fid) in self.surface_on_by[surface]:
                if self._is_fact_true(state, fid):
                    return c
        return None

    def _is_crate_in_truck(self, state, crate):
        """Check if crate is in any truck."""
        for (c, t), fid in self.in_fact_id.items():
            if c == crate and self._is_fact_true(state, fid):
                return True
        return False

    def _is_crate_being_lifted(self, state, crate):
        """Check if crate is being lifted by any hoist."""
        for (h, c), fid in self.lifting_fact_id.items():
            if c == crate and self._is_fact_true(state, fid):
                return True
        return False

    def _surface_place(self, surface):
        """Get the static place of a pallet. Crates move, so return None for them."""
        if surface in self.pallet_place:
            return self.pallet_place[surface]
        return None

    def _goal_displacement_cost(self, state):
        """Compute lower bound based on unsatisfied on-goals.
        
        For each unsatisfied on(crate, surface) goal:
        - If crate is already on surface: 0
        - If crate is being lifted at the right place and surface is clear: 1 (drop)
        - If crate is at the same place as target and on some surface: 2 (lift + drop) minimum
        - If crate is at a different place: 4 (lift + load + drive + unload + drop = 5, but
          some steps might overlap; conservative: 4)
        - If crate is in a truck: 2 (unload + drop) minimum
        
        Additionally, if the target surface is not clear, we need to account for clearing cost.
        Each crate that must be moved off the target surface costs at least 2 actions (lift + load/drop).
        
        This is admissible because these are minimum action counts.
        """
        total_cost = 0
        
        for crate, target_surface in self.goal_on.items():
            # Check if goal already satisfied
            goal_fid = self.goal_on_fact_ids.get(crate)
            if goal_fid is not None and self._is_fact_true(state, goal_fid):
                # Goal on(crate, target_surface) is already true
                # But we need to check if something is on top of the crate that shouldn't be
                # (could be disrupted later). For admissibility, count 0.
                continue
            
            cost = 0
            
            # Determine crate's current status
            crate_being_lifted = self._is_crate_being_lifted(state, crate)
            crate_in_truck = self._is_crate_in_truck(state, crate)
            
            if crate_being_lifted:
                # Minimum: drop (1 action), but might need load+drive+unload+drop if wrong place
                cost = 1
            elif crate_in_truck:
                # Minimum: unload + drop = 2 actions
                cost = 2
            else:
                # Crate is on some surface at some place
                # Need to: clear crate if not clear, lift, possibly transport, drop
                crate_clear_fid = self.clear_fact_id.get(crate)
                crate_is_clear = False
                if crate_clear_fid is not None:
                    crate_is_clear = self._is_fact_true(state, crate_clear_fid)
                
                if not crate_is_clear:
                    # Something is on top of crate, need to clear it first
                    # Clearing one crate on top costs at least 2 (lift + load/put somewhere)
                    cost += 2
                
                # Need lift + drop at minimum (2), or lift + load + drive + unload + drop (5) if different place
                cost += 2  # lift + drop minimum
            
            # Check if target surface needs clearing
            target_clear_fid = self.clear_fact_id.get(target_surface)
            if target_clear_fid is not None:
                target_is_clear = self._is_fact_true(state, target_clear_fid)
                if not target_is_clear:
                    # Need to move whatever is on the target surface
                    # Check if the thing on target is supposed to be there in the goal
                    what_on_target = self._find_what_is_on(state, target_surface)
                    if what_on_target is not None:
                        # Check if what_on_target's goal is to be on target_surface
                        if self.goal_on.get(what_on_target) != target_surface:
                            # This crate needs to be moved: at least 2 actions (lift + load/drop)
                            cost += 2
            
            total_cost += cost
        
        return total_cost

    def _task_network_estimate(self, state, task_network):
        """Estimate minimum cost to decompose remaining task network.
        
        For each task in the network:
        - Primitive operators: 1 (except nop: 0)
        - do_put_on: state-dependent cost based on crate position and goal
        - do_clear: state-dependent cost
        - do_get_truck: 0 or 1 depending on truck location
        - do_lift_crate: 1-2 depending on state
        - do_load_truck: 2-4 depending on state  
        - do_unload_truck: 2-3 depending on state
        
        Admissible because we use minimum decomposition cost for each task type.
        """
        total = 0
        
        for task in task_network:
            if isinstance(task, OperatorClass):
                # Primitive action
                task_type = self._get_task_type(task.name)
                if task_type == "nop":
                    total += 0
                else:
                    total += 1
            elif isinstance(task, AbstractTaskClass):
                total += self._estimate_abstract_task(state, task)
        
        return total

    def _estimate_abstract_task(self, state, task):
        """Estimate minimum cost for an abstract task based on current state."""
        task_type = self._get_task_type(task.name)
        args = self._get_task_args(task.name)
        
        if task_type == "do_put_on":
            return self._estimate_do_put_on(state, args)
        elif task_type == "do_clear":
            return self._estimate_do_clear(state, args)
        elif task_type == "do_get_truck":
            return self._estimate_do_get_truck(state, args)
        elif task_type == "do_lift_crate":
            return self._estimate_do_lift_crate(state, args)
        elif task_type == "do_load_truck":
            return self._estimate_do_load_truck(state, args)
        elif task_type == "do_unload_truck":
            return self._estimate_do_unload_truck(state, args)
        
        # Unknown task type, return 1 as minimum
        return 1

    def _estimate_do_put_on(self, state, args):
        """Estimate do_put_on(crate, surface) cost.
        
        Methods:
        m0: on(c,s2) already true -> nop = 0
        m1: at(c,p) -> clear_c + clear_s2 + lift + drop = varies
        m2: in(c,t) -> get_truck + clear_s2 + unload + drop = varies  
        m3: general -> load_truck + drive + unload_truck = varies
        """
        if len(args) < 2:
            return 1
        
        crate = args[0]
        surface = args[1]
        
        # Check if already on target (m0)
        on_fid = self.on_fact_id.get((crate, surface))
        if on_fid is not None and self._is_fact_true(state, on_fid):
            return 0
        
        # Check if crate is being lifted
        if self._is_crate_being_lifted(state, crate):
            # Just need to drop (maybe after clearing surface)
            cost = 1  # drop
            target_clear_fid = self.clear_fact_id.get(surface)
            if target_clear_fid is not None and not self._is_fact_true(state, target_clear_fid):
                cost += 2  # need to clear surface first
            return cost
        
        # Check if in truck (m2)
        if self._is_crate_in_truck(state, crate):
            # get_truck(0-1) + clear_s2(0+) + unload(1) + drop(1) 
            cost = 2  # unload + drop minimum
            target_clear_fid = self.clear_fact_id.get(surface)
            if target_clear_fid is not None and not self._is_fact_true(state, target_clear_fid):
                cost += 2  # clearing surface
            return cost
        
        # Crate is at some place on some surface
        cost = 0
        
        # Need to clear crate
        crate_clear_fid = self.clear_fact_id.get(crate)
        if crate_clear_fid is not None and not self._is_fact_true(state, crate_clear_fid):
            # Count stack height above crate for better estimate
            stack_cost = self._estimate_clear_stack(state, crate)
            cost += stack_cost
        
        # Need to clear target surface
        target_clear_fid = self.clear_fact_id.get(surface)
        if target_clear_fid is not None and not self._is_fact_true(state, target_clear_fid):
            stack_cost_target = self._estimate_clear_stack(state, surface)
            cost += stack_cost_target
        
        # Lift + drop minimum
        cost += 2
        
        # Check if transport needed (crate at different place than target surface)
        # For pallets we know the place; for crates as surfaces, it's dynamic
        target_place = self._surface_place(surface)
        if target_place is not None:
            # Check if crate is at the target place
            at_fid = self.at_fact_id.get((crate, target_place))
            if at_fid is not None:
                if not self._is_fact_true(state, at_fid):
                    # Crate is at different place, need transport
                    # load + drive + unload + drop instead of just lift + drop
                    cost += 3  # additional: load + drive + unload (drop already counted)
        
        return cost

    def _estimate_clear_stack(self, state, surface, depth=0):
        """Estimate cost to clear a surface by counting what's on top.
        
        Each crate on top needs: lift(1) + load(1) = 2 minimum to move away.
        Recursive for stacked crates.
        """
        if depth > 15:  # Safety limit for deep stacks
            return 2
        
        clear_fid = self.clear_fact_id.get(surface)
        if clear_fid is not None and self._is_fact_true(state, clear_fid):
            return 0
        
        # Find what's on this surface
        if surface in self.surface_on_by:
            for (c, fid) in self.surface_on_by[surface]:
                if self._is_fact_true(state, fid):
                    # Crate c is on surface
                    # Cost to clear c: clear what's on c + lift c + load c
                    sub_clear = self._estimate_clear_stack(state, c, depth + 1)
                    return sub_clear + 2  # lift + load/put somewhere
        
        # Nothing found on it but it's not clear? Shouldn't happen, return small cost
        return 2

    def _estimate_do_clear(self, state, args):
        """Estimate do_clear(surface, place) cost.
        
        m4: clear(s1) and at(s1,p1) -> nop = 0
        m5: not clear, need to lift and load = at least 2 per crate on top
        """
        if len(args) < 2:
            return 0
        
        surface = args[0]
        clear_fid = self.clear_fact_id.get(surface)
        if clear_fid is not None and self._is_fact_true(state, clear_fid):
            return 0
        
        return self._estimate_clear_stack(state, surface)

    def _estimate_do_get_truck(self, state, args):
        """Estimate do_get_truck(truck, place) cost.
        
        m6: at(t,p1) -> nop = 0
        m7: not at -> Drive = 1
        """
        if len(args) < 2:
            return 0
        
        truck = args[0]
        place = args[1]
        at_fid = self.at_fact_id.get((truck, place))
        if at_fid is not None and self._is_fact_true(state, at_fid):
            return 0
        return 1

    def _estimate_do_lift_crate(self, state, args):
        """Estimate do_lift_crate(crate, place, hoist) cost.
        
        m8: in truck -> get_truck + unload = 1-2
        m9: on surface at place -> Lift = 1
        """
        if len(args) < 3:
            return 1
        
        crate = args[0]
        
        if self._is_crate_in_truck(state, crate):
            return 2  # get_truck (0-1) + unload (1)
        
        return 1  # Lift

    def _estimate_do_load_truck(self, state, args):
        """Estimate do_load_truck(crate, surface, place, truck) cost.
        
        m10: get_truck + clear_c + Lift + Load = at least 2 (Lift + Load)
        """
        if len(args) < 4:
            return 2
        
        crate = args[0]
        truck = args[3]
        place = args[2]
        
        cost = 2  # Lift + Load minimum
        
        # Check if truck needs to come
        at_fid = self.at_fact_id.get((truck, place))
        if at_fid is not None and not self._is_fact_true(state, at_fid):
            cost += 1  # Drive
        
        # Check if crate needs clearing
        crate_clear_fid = self.clear_fact_id.get(crate)
        if crate_clear_fid is not None and not self._is_fact_true(state, crate_clear_fid):
            cost += 2  # clearing
        
        return cost

    def _estimate_do_unload_truck(self, state, args):
        """Estimate do_unload_truck(crate, surface, place, truck) cost.
        
        m11: clear_s + Unload + Drop = at least 2 (Unload + Drop)
        """
        if len(args) < 4:
            return 2
        
        surface = args[1]
        cost = 2  # Unload + Drop minimum
        
        # Check if surface needs clearing
        clear_fid = self.clear_fact_id.get(surface)
        if clear_fid is not None and not self._is_fact_true(state, clear_fid):
            cost += self._estimate_clear_stack(state, surface)
        
        return cost
