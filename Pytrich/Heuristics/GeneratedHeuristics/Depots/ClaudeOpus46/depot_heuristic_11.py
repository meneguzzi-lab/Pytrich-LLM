"""Depots Domain Heuristic

Bottleneck: Crates must be moved between places via trucks requiring
lift->load->drive->unload->drop (5 actions min for cross-place moves),
and hoists are place-bound. Clearing stacks above targets adds recursive cost.

Admissible lower bounds:
1. Goal displacement: each unsatisfied on(c,s) goal costs >= 2 (same place) or >= 5 (different place)
2. Task network: remaining tasks each need >= 1 action; do_put_on needs >= 2 if not satisfied
3. Stack clearing: if goal target not clear, items above cost >= 2 each to remove

Return max of these terms.
"""

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotsAnthropicOpus46Heuristic11(Heuristic):
    def __init__(self, name="depot_anthropicopus46_heuristic_11"):
        super().__init__(name)
        self.on_fact_ids = {}       # (crate, surface) -> global_id
        self.at_fact_ids = {}       # (locatable, place) -> global_id
        self.clear_fact_ids = {}    # surface -> global_id
        self.in_fact_ids = {}       # (crate, truck) -> global_id
        self.lifting_fact_ids = {}  # (hoist, crate) -> global_id
        self.available_fact_ids = {} # hoist -> global_id
        
        # Goal structures
        self.goal_on_pairs = []     # list of (crate, surface, fact_global_id)
        
        # Object info
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.pallets = set()
        
        # Location mapping: object -> set of places where it could be (from at facts)
        self.crate_locations = {}   # crate -> list of (place, fact_id)
        self.surface_locations = {} # surface -> list of (place, fact_id)
        self.hoist_locations = {}   # hoist -> list of (place, fact_id)
        self.truck_locations = {}   # truck -> list of (place, fact_id)
        
        # Stack tracking: surface -> list of (crate, on_fact_id) for on(crate, surface)
        self.on_surface_facts = {}  # surface -> [(crate, fact_id)]
        self.on_crate_facts = {}    # crate -> [(what_is_on_it_crate, fact_id)]
        
        # For task network analysis
        self.do_put_on_task_ids = set()  # global_ids of do_put_on abstract tasks
        self.do_clear_task_ids = set()
        self.do_get_truck_task_ids = set()
        self.do_lift_crate_task_ids = set()
        self.do_load_truck_task_ids = set()
        self.do_unload_truck_task_ids = set()
        
        # Precomputed: for each do_put_on task, store (crate, surface)
        self.put_on_task_args = {}  # task_global_id -> (crate, surface)
        
        # Nop operator id
        self.nop_op_ids = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        """Parse task/operator name like 'do_put_on[crate0,pallet1]' or '__top_4_do_put_on[crate0,pallet1]'"""
        # Strip any prefix numbers/underscores from grounding
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            task_part = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            # Extract the core task name (last part after underscores that form the task name)
            return task_part, args
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Build fact index maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "at" and len(args) == 2:
                self.at_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "in" and len(args) == 2:
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "available" and len(args) == 1:
                self.available_fact_ids[args[0]] = f.global_id
        
        # Identify objects from at facts
        for (obj, place), fid in self.at_fact_ids.items():
            self.places.add(place)
            if obj.startswith("crate"):
                self.crates.add(obj)
                if obj not in self.crate_locations:
                    self.crate_locations[obj] = []
                self.crate_locations[obj].append((place, fid))
            elif obj.startswith("pallet"):
                self.pallets.add(obj)
                self.surfaces.add(obj)
                if obj not in self.surface_locations:
                    self.surface_locations[obj] = []
                self.surface_locations[obj].append((place, fid))
            elif obj.startswith("hoist"):
                self.hoists.add(obj)
                if obj not in self.hoist_locations:
                    self.hoist_locations[obj] = []
                self.hoist_locations[obj].append((place, fid))
            elif obj.startswith("truck"):
                self.trucks.add(obj)
                if obj not in self.truck_locations:
                    self.truck_locations[obj] = []
                self.truck_locations[obj].append((place, fid))
        
        # Crates are also surfaces
        for c in self.crates:
            self.surfaces.add(c)
        
        # Build on-surface index
        for (crate, surface), fid in self.on_fact_ids.items():
            if surface not in self.on_surface_facts:
                self.on_surface_facts[surface] = []
            self.on_surface_facts[surface].append((crate, fid))
        
        # Parse goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                fid = self.on_fact_ids.get((args[0], args[1]))
                if fid is not None:
                    self.goal_on_pairs.append((args[0], args[1], fid))
        
        # Index abstract tasks
        for at in model.abstract_tasks:
            tname, targs = self._parse_task_name(at.name)
            if "do_put_on" in tname:
                self.do_put_on_task_ids.add(at.global_id)
                if len(targs) >= 2:
                    self.put_on_task_args[at.global_id] = (targs[0], targs[1])
            elif "do_clear" in tname:
                self.do_clear_task_ids.add(at.global_id)
            elif "do_get_truck" in tname:
                self.do_get_truck_task_ids.add(at.global_id)
            elif "do_lift_crate" in tname:
                self.do_lift_crate_task_ids.add(at.global_id)
            elif "do_load_truck" in tname:
                self.do_load_truck_task_ids.add(at.global_id)
            elif "do_unload_truck" in tname:
                self.do_unload_truck_task_ids.add(at.global_id)
        
        # Index nop operators
        for op in model.operators:
            tname, _ = self._parse_task_name(op.name)
            if "nop" in tname:
                self.nop_op_ids.add(op.global_id)
        
        # Precompute place for each pallet (pallets don't move)
        self.pallet_place = {}  # pallet -> place
        for (obj, place), fid in self.at_fact_ids.items():
            if obj.startswith("pallet"):
                self.pallet_place[obj] = place
        
        # Precompute hoist_at_place: place -> list of hoist global_ids for available checks
        self.hoists_at_place = {}  # place -> [hoist_name]
        for h in self.hoists:
            for (place, fid) in self.hoist_locations.get(h, []):
                if place not in self.hoists_at_place:
                    self.hoists_at_place[place] = []
                self.hoists_at_place[place].append(h)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_crate_place(self, crate, state):
        """Find where a crate currently is (at or in truck). Returns place name or None."""
        for (place, fid) in self.crate_locations.get(crate, []):
            if (state >> fid) & 1:
                return place
        return None

    def _get_surface_place(self, surface, state):
        """Find where a surface is. Pallets are fixed; crates move."""
        if surface in self.pallet_place:
            return self.pallet_place[surface]
        # It's a crate acting as surface
        return self._get_crate_place(surface, state)

    def _is_crate_in_truck(self, crate, state):
        """Check if crate is in any truck."""
        for (c, t), fid in self.in_fact_ids.items():
            if c == crate and (state >> fid) & 1:
                return True
        return False

    def _is_clear(self, surface, state):
        """Check if surface is clear."""
        fid = self.clear_fact_ids.get(surface)
        if fid is None:
            return True
        return (state >> fid) & 1

    def _is_on(self, crate, surface, state):
        """Check if on(crate, surface) holds."""
        fid = self.on_fact_ids.get((crate, surface))
        if fid is None:
            return False
        return (state >> fid) & 1

    def _estimate_put_on_cost(self, crate, target_surface, state):
        """Estimate minimum cost to achieve on(crate, target_surface).
        
        Lower bound reasoning:
        - If already on target: 0
        - If crate at same place as target: need lift + drop = 2 minimum
          (plus clearing costs)
        - If crate at different place: need lift + load + drive + unload + drop = 5 minimum
          (plus clearing costs)
        - If crate in truck: need drive(maybe) + unload + drop = 2-3 minimum
        """
        # Already satisfied?
        fid = self.on_fact_ids.get((crate, target_surface))
        if fid is not None and (state >> fid) & 1:
            return 0
        
        cost = 0
        
        # Cost to clear the crate (if it has something on it)
        if not self._is_clear(crate, state):
            # Need to clear crate: at minimum lift + load = 2 per crate on top
            # We estimate 1 level = 2 actions (could be more)
            cost += 2
        
        # Cost to clear target surface
        if not self._is_clear(target_surface, state):
            cost += 2
        
        # Check if crate is in a truck
        in_truck = self._is_crate_in_truck(crate, state)
        
        if in_truck:
            # Need: (maybe drive) + unload + drop = 2-3
            target_place = self._get_surface_place(target_surface, state)
            cost += 3  # conservative: drive + unload + drop
        else:
            crate_place = self._get_crate_place(crate, state)
            target_place = self._get_surface_place(target_surface, state)
            
            if crate_place is not None and target_place is not None:
                if crate_place == target_place:
                    # Same place: lift + drop = 2
                    cost += 2
                else:
                    # Different places: lift + load + drive + unload + drop = 5
                    cost += 5
            elif crate_place is None and not in_truck:
                # Crate is being held (lifting state) - just need drop = 1
                cost += 1
            else:
                # Unknown location, assume some cost
                cost += 2
        
        return cost

    def _goal_displacement_heuristic(self, state):
        """Sum of minimum costs to achieve all unsatisfied goal on-predicates.
        This is admissible because each goal requires independent actions."""
        total = 0
        for crate, surface, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                total += self._estimate_put_on_cost(crate, surface, state)
        return total

    def _task_network_heuristic(self, state, task_network):
        """Estimate cost from remaining task network.
        Each task contributes a minimum action count."""
        total = 0
        for task in task_network:
            gid = task.global_id
            
            # Check if it's a nop
            if gid in self.nop_op_ids:
                continue
            
            # Check if it's an operator (primitive action)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                total += 1
                continue
            
            # Abstract tasks
            if gid in self.do_put_on_task_ids:
                args = self.put_on_task_args.get(gid)
                if args:
                    crate, surface = args
                    fid = self.on_fact_ids.get((crate, surface))
                    if fid is not None and (state >> fid) & 1:
                        # Already satisfied - m0 applies (just nop)
                        continue
                    else:
                        # Need actual work
                        cost = self._estimate_put_on_cost(crate, surface, state)
                        total += max(cost, 2)  # At minimum lift + drop
                else:
                    total += 2
            elif gid in self.do_clear_task_ids:
                # do_clear: 0 if already clear, else at least 2 (lift + load)
                total += 1  # conservative lower bound
            elif gid in self.do_get_truck_task_ids:
                # do_get_truck: 0 if truck already there, else 1 (drive)
                total += 0  # Could be 0 (nop) - conservative
            elif gid in self.do_lift_crate_task_ids:
                total += 1  # At least one lift or unload
            elif gid in self.do_load_truck_task_ids:
                total += 2  # lift + load minimum
            elif gid in self.do_unload_truck_task_ids:
                total += 2  # unload + drop minimum
            else:
                # Unknown abstract task
                total += 1
        
        return total

    def _unsatisfied_goals_heuristic(self, state):
        """Count unsatisfied goals weighted by displacement.
        Simple but effective lower bound."""
        count = 0
        for crate, surface, fid in self.goal_on_pairs:
            if not ((state >> fid) & 1):
                count += 1
        # Each unsatisfied goal needs at least 2 actions (lift + drop)
        return count * 2

    def _compute(self, state, task_network):
        """Compute heuristic as max of multiple admissible lower bounds."""
        if not task_network:
            # Check if goals are met
            if (state & self.model.goals) == self.model.goals:
                return 0
            return 0
        
        # Term 1: Goal displacement - sum of per-goal minimum costs
        h_goal = self._goal_displacement_heuristic(state)
        
        # Term 2: Task network estimation
        h_task = self._task_network_heuristic(state, task_network)
        
        # Term 3: Simple unsatisfied goal count * 2
        h_unsat = self._unsatisfied_goals_heuristic(state)
        
        return max(h_goal, h_task, h_unsat)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
