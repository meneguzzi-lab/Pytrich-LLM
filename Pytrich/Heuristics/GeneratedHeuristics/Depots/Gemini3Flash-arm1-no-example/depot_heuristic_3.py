from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic3(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots domain.

    Bottleneck Analysis:
    The primary bottleneck in Depots is the serial nature of moving crates between locations using 
    a limited number of hoists and trucks, compounded by the stack structure (on/clear predicates). 
    Moving a crate between locations requires a minimum sequence of 5 actions (Lift, Load, Drive, Unload, Drop), 
    while moving a crate within a location requires 2 (Lift, Drop). Additionally, any crates 
    obstructing a goal crate or target surface must be cleared first.

    Admissible Lower Bounds:
    1. Goal-based Movement Cost (sum): For each goal on(C, S) not satisfied, we count the minimum 
       actions needed to move C to S. This is additive because each crate C has a unique goal.
    2. Task Network Cost (sum): Each pending task in the network has a minimum cost to be resolved.
    3. Clearing Cost (additive): If a goal crate C or surface S is not 'clear', at least two additional 
       actions (Lift/Drop or Lift/Load) are required for each obstructing stack. Using a simple +2 
       for 'not clear' is a fast, admissible lower bound.
    """

    def initialize(self, model: Model, initial_node: HTNNode):
        # Fact mappings
        self.at_ids = {}      # (locatable, place) -> global_id
        self.on_ids = {}      # (crate, surface) -> global_id
        self.in_ids = {}      # (crate, truck) -> global_id
        self.lifting_ids = {} # (hoist, crate) -> global_id
        self.clear_ids = {}   # surface -> global_id

        self.places = set()
        self.trucks = set()
        self.hoists = set()
        self.surfaces = set()
        self.crates = set()

        # Pre-parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == "at":
                self.at_ids[(args[0], args[1])] = f.global_id
                self.places.add(args[1])
            elif pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[0])
                self.surfaces.add(args[1])
            elif pred == "in":
                self.in_ids[(args[0], args[1])] = f.global_id
                self.trucks.add(args[1])
            elif pred == "lifting":
                self.lifting_ids[(args[0], args[1])] = f.global_id
                self.hoists.add(args[0])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id

        # Precompute static places for pallets (pallets never move)
        self.pallet_to_place = {}
        initial_state = initial_node.state
        for (loc, p), fid in self.at_ids.items():
            if "pallet" in loc and (initial_state >> fid) & 1:
                self.pallet_to_place[loc] = p

        # Precompute goal information
        self.goal_on_map = {}
        self.goal_bits = []
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "on":
                # Find bit ID for this goal name
                for f in model.facts:
                    if f.name == gn:
                        self.goal_bits.append(f.global_id)
                        self.goal_on_map[f.global_id] = (args[0], args[1])
                        break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # 1. Goal Sum (Admissible because crates are moved independently)
        goal_cost = 0
        for g_bit in self.goal_bits:
            if not ((state >> g_bit) & 1):
                C, S = self.goal_on_map[g_bit]
                goal_cost += self._estimate_move_cost(C, S, state)

        # 2. Task Network Sum (Pending tasks must be executed)
        task_cost = 0
        for task in task_network:
            task_cost += self._estimate_task_cost(task, state)

        # Return max to remain admissible while capturing both sources of work
        return max(goal_cost, task_cost)

    def _estimate_move_cost(self, crate, surface, state):
        cost = 0
        
        # Locating current place of crate and target surface
        c_loc = self._get_loc_location(crate, state)
        s_loc = self._get_loc_location(surface, state)

        if c_loc is not None and s_loc is not None:
            if c_loc == s_loc:
                cost += 2 # Lift, Drop
            else:
                cost += 5 # Lift, Load, Drive, Unload, Drop
        elif c_loc is None:
            # Crate is in a truck or being lifted
            is_lifting = False
            for h in self.hoists:
                fid = self.lifting_ids.get((h, crate))
                if fid is not None and (state >> fid) & 1:
                    cost += 1 # Just Drop
                    is_lifting = True
                    break
            if not is_lifting:
                # Must be in a truck
                cost += 2 # Unload, Drop
        
        # Clearing costs: If crate or surface is not clear, add 2 actions (Lift/Drop of obstructer)
        c_clear_id = self.clear_ids.get(crate)
        if c_clear_id is not None and not ((state >> c_clear_id) & 1):
            cost += 2
        
        s_clear_id = self.clear_ids.get(surface)
        if s_clear_id is not None and not ((state >> s_clear_id) & 1):
            cost += 2

        return cost

    def _estimate_task_cost(self, task, state):
        name, args = self._parse_any_name(task.name)
        if name == "do_put_on":
            return self._estimate_move_cost(args[0], args[1], state)
        if name == "do_clear":
            fid = self.clear_ids.get(args[0])
            return 0 if (fid is not None and (state >> fid) & 1) else 2
        if name == "do_get_truck":
            fid = self.at_ids.get((args[0], args[1]))
            return 0 if (fid is not None and (state >> fid) & 1) else 1
        if name == "do_lift_crate":
            return 1
        if name in ["do_load_truck", "do_unload_truck"]:
            return 2
        return 0

    def _get_loc_location(self, item, state):
        if item in self.pallet_to_place:
            return self.pallet_to_place[item]
        for p in self.places:
            fid = self.at_ids.get((item, p))
            if fid is not None and (state >> fid) & 1:
                return p
        return None

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

    def _parse_any_name(self, name: str):
        # Normalize delimiters from various task formats
        clean = name.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace(',', ' ')
        parts = clean.split()
        if not parts: return "", []
        return parts[0], parts[1:]
