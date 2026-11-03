from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic10(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots Domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Depots domain is the sequential displacement of crates.
    A crate requires a specific sequence of operations (clearing, lifting, loading, 
    driving, unloading, and dropping) to move from an initial stack to a goal stack. 
    Only one hoist can be used per location at a time, and trucks have finite movement,
    serializing the movement of crates across locations.

    Admissibility Sketches:
    1. Goal-based displacement: Every crate not on its goal surface must be moved, 
       requiring at least a Lift and a Drop (2 actions). If the goal surface is in 
       a different location, a Load and Unload are also required (4 actions total).
    2. Clearance cost: For every crate or surface involved in a goal, any crate 
       stacked on top of it must be moved elsewhere (at least 2 actions each: Lift/Drop).
    3. Task Network: Every non-satisfied task in the task network represents unavoidable 
       future actions. For example, a do_put_on task for a crate not currently in 
       place must eventually result in at least 2 or 4 actions.
    """

    def __init__(self, depot_gemini3flash_heuristic_10="depot_gemini3flash_heuristic_10"):
        super().__init__(name=depot_gemini3flash_heuristic_10)
        self.goals_on = {} # crate -> target surface
        self._parsed_facts = []
        self.pallets = set()
        self.hoists = set()
        self.trucks = set()
        self.places = set()
        self.static_locs = {} # pallet/hoist -> place

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
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # Pre-parse all facts for efficiency
        self._parsed_facts = [self._parse_fact_name(f.name) for f in model.facts]
        
        # Identify static locations and types from grounded facts/operators
        for i, (pred, args) in enumerate(self._parsed_facts):
            if pred == "at":
                self.places.add(args[1])
            elif pred == "available":
                self.hoists.add(args[0])
            elif pred == "in":
                self.trucks.add(args[1])

        # Infer pallet status from on facts involving crates and non-hoist surfaces
        # We can also scrape static facts from operator names as per hint
        for op in model.operators:
            # Drop[hoist, crate, surface, place]
            if "Drop" in op.name:
                p, args = self._parse_fact_name(op.name)
                if len(args) == 4:
                    self.static_locs[args[0]] = args[3] # Hoist
                    self.static_locs[args[2]] = args[3] # Surface (Pallet)
                    self.pallets.add(args[2])

        # Parse goals
        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                self.goals_on[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # 1. Scrape Current State efficiently
        on_rel = {}     # s -> c (c is on s)
        on_rel_inv = {} # c -> s
        at_loc = {}     # obj -> p
        in_truck = {}   # c -> t
        lifting_h = {}  # h -> c
        clear_objs = set()

        temp_state = state
        idx = 0
        while temp_state:
            if temp_state & 1:
                p, args = self._parsed_facts[idx]
                if p == "on":
                    on_rel[args[1]] = args[0]
                    on_rel_inv[args[0]] = args[1]
                elif p == "at":
                    at_loc[args[0]] = args[1]
                elif p == "in":
                    in_truck[args[0]] = args[1]
                elif p == "lifting":
                    lifting_h[args[0]] = args[1]
                elif p == "clear":
                    clear_objs.add(args[0])
            temp_state >>= 1
            idx += 1

        # Add static locations
        for obj, p in self.static_locs.items():
            if obj not in at_loc: at_loc[obj] = p

        # Helper to find current location of a crate
        def get_loc(obj):
            if obj in at_loc: return at_loc[obj]
            if obj in in_truck: return at_loc.get(in_truck[obj])
            for h, c in lifting_h.items():
                if c == obj: return at_loc.get(h)
            # Follow stack down
            curr = obj
            visited = {curr}
            while curr in on_rel_inv:
                s = on_rel_inv[curr]
                if s in at_loc: return at_loc[s]
                if s in visited: break
                curr = s
                visited.add(curr)
            return None

        # Helper to count crates above an object
        def count_above(obj):
            cnt = 0
            curr = obj
            while curr in on_rel:
                cnt += 1
                curr = on_rel[curr]
            return cnt

        # Calculate Stability (Blocksworld-style)
        stable_crates = set()
        for p in self.pallets:
            curr = p
            while curr in on_rel:
                top = on_rel[curr]
                if self.goals_on.get(top) == curr:
                    stable_crates.add(top)
                    curr = top
                else: break

        # Term 1: Goal-based Displacement and Clearance
        h_goal = 0
        for c, target_s in self.goals_on.items():
            if c not in stable_crates:
                # Base move cost
                loc_c = get_loc(c)
                loc_s = get_loc(target_s)
                if loc_c == loc_s:
                    h_goal += 2 # Lift + Drop
                else:
                    h_goal += 4 # Lift + Load + Unload + Drop
                
                # Clearance requirement for both c and target_s
                h_goal += count_above(c) * 2
                h_goal += count_above(target_s) * 2

        # Term 2: Task Network Progress
        h_tn = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                h_tn += task.cost
            else:
                # Abstract Task estimation
                tname, targs = self._parse_fact_name(task.name)
                if tname == "do_put_on":
                    c, s = targs[0], targs[1]
                    # Nop bonus logic: if not satisfied, add cost
                    if not (c in on_rel_inv and on_rel_inv[c] == s):
                        h_tn += 2
                        loc_c, loc_s = get_loc(c), get_loc(s)
                        if loc_c != loc_s: h_tn += 2
                        h_tn += (count_above(c) + count_above(s)) * 2
                elif tname == "do_clear":
                    h_tn += count_above(targs[0]) * 2
                elif tname == "do_lift_crate":
                    h_tn += 1
                elif tname == "do_load_truck":
                    h_tn += 2
                elif tname == "do_unload_truck":
                    h_tn += 2

        return max(h_goal, h_tn)
