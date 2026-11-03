from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic3(Heuristic):
    """
    Bottleneck Analysis:
    The main bottleneck in the Depots domain is the interaction between stack clearance and transportation. 
    A crate cannot be moved until all crates above it are cleared (Lift/Drop bottleneck), and 
    cross-location movement requires synchronized truck and hoist availability (Truck/Hoist bottleneck).

    Admissibility Sketches:
    - Goal Distance Term: Each crate not in its goal position requires a sequence of actions. 
      If at the correct location but wrong surface: Lift + Drop = 2 actions. 
      If at a different location: Lift + Load + Drive + Unload + Drop = 5 actions. 
      These are absolute minimums for any valid plan, thus admissible.
    - Task Network Term: Each HTN task represents a commitment to achieve a state. 
      A 'do_put_on(c, s)' not yet achieved must cost at least 2 or 5 depending on location.
      A 'do_clear(s, p)' must cost at least 2 for every crate currently on 's'.
    """

    def __init__(self, depot_gemini3flash_heuristic_3="depot_gemini3flash_heuristic_3"):
        super().__init__(depot_gemini3flash_heuristic_3)
        self.at_fact_ids = {}      # (obj, place) -> global_id
        self.on_fact_ids = {}      # (obj, surface) -> global_id
        self.clear_fact_ids = {}   # surface -> global_id
        self.in_fact_ids = {}      # (crate, truck) -> global_id
        self.lifting_facts = {}    # crate -> list of (hoist, global_id)
        self.crate_at_options = {} # crate -> list of (place, global_id)
        self.crate_on_options = {} # crate -> list of (surface, global_id)
        
        self.static_loc = {}       # pallet/hoist -> place
        self.goals_to_check = []   # list of (crate, surface, crate_name, surface_name, goal_fact_id)
        self.places = set()
        self.trucks = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Static Locations and Types from Operator names (to bypass grounder stripping)
        for op in model.operators:
            # Lift[hoist, crate, surface, place] or Drop[hoist, crate, surface, place]
            if "Lift" in op.name or "Drop" in op.name:
                parts = self._parse_bracket_args(op.name)
                if len(parts) == 4:
                    h, c, s, p = parts
                    self.static_loc[h] = p
                    self.static_loc[s] = p
                    self.places.add(p)
            # Drive[truck, p1, p2]
            elif "Drive" in op.name:
                parts = self._parse_bracket_args(op.name)
                if len(parts) == 3:
                    self.trucks.add(parts[0])
                    self.places.add(parts[1])
                    self.places.add(parts[2])

        # 2. Index Facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "at":
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.crate_at_options: self.crate_at_options[args[0]] = []
                self.crate_at_options[args[0]].append((args[1], f.global_id))
            elif pred == "on":
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.crate_on_options: self.crate_on_options[args[0]] = []
                self.crate_on_options[args[0]].append((args[1], f.global_id))
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "in":
                self.in_fact_ids[(args[0], args[1])] = f.global_id
            elif pred == "lifting":
                if args[1] not in self.lifting_facts: self.lifting_facts[args[1]] = []
                self.lifting_facts[args[1]].append((args[0], f.global_id))

        # 3. Process Goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "on":
                crate, surface = args[0], args[1]
                fid = self.on_fact_ids.get((crate, surface), -1)
                self.goals_to_check.append((crate, surface, fid))

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        h_goals = 0
        # Term 1: Goal displacement
        for crate, target_surface, goal_fid in self.goals_to_check:
            if goal_fid != -1 and (state >> goal_fid) & 1:
                continue
            
            # If not at goal, calculate min steps
            c_loc = self._get_loc(state, crate)
            s_loc = self.static_loc.get(target_surface) or self._get_loc(state, target_surface)
            
            if c_loc and s_loc:
                if c_loc == s_loc:
                    h_goals += 2 # Lift + Drop
                else:
                    h_goals += 5 # Lift, Load, Drive, Unload, Drop
            else:
                h_goals += 2 # Fallback

        # Term 2: Task Network Progress
        h_tn = 0
        for task in task_network:
            t_name, t_args = self._parse_compound_name(task.name)
            if t_name == "do_put_on":
                c, s = t_args[0], t_args[1]
                fid = self.on_fact_ids.get((c, s), -1)
                if fid == -1 or not ((state >> fid) & 1):
                    c_loc = self._get_loc(state, c)
                    s_loc = self.static_loc.get(s) or self._get_loc(state, s)
                    h_tn += 5 if c_loc != s_loc else 2
            elif t_name == "do_clear":
                s = t_args[0]
                cfid = self.clear_fact_ids.get(s, -1)
                if cfid != -1 and not ((state >> cfid) & 1):
                    h_tn += 2 # At least one Lift + Load/Drop needed
            elif t_name == "do_get_truck":
                t, p = t_args[0], t_args[1]
                afid = self.at_fact_ids.get((t, p), -1)
                if afid != -1 and not ((state >> afid) & 1):
                    h_tn += 1 # Drive

        return max(h_goals, h_tn)

    def _get_loc(self, state, obj):
        # Fastest possible location lookup via bitwise
        # Check direct 'at' facts
        if obj in self.crate_at_options:
            for place, fid in self.crate_at_options[obj]:
                if (state >> fid) & 1: return place
        # Check 'on' facts (if on a pallet or crate with known location)
        if obj in self.crate_on_options:
            for surf, fid in self.crate_on_options[obj]:
                if (state >> fid) & 1:
                    if surf in self.static_loc: return self.static_loc[surf]
        # Check if in a truck
        for t in self.trucks:
            in_fid = self.in_fact_ids.get((obj, t), -1)
            if in_fid != -1 and (state >> in_fid) & 1:
                return self._get_loc(state, t)
        # Check if being lifted
        if obj in self.lifting_facts:
            for hoist, fid in self.lifting_facts[obj]:
                if (state >> fid) & 1: return self.static_loc.get(hoist)
        # Static locs
        if obj in self.static_loc: return self.static_loc[obj]
        return None

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return None, []

    def _parse_bracket_args(self, name: str):
        start = name.find("[")
        if start == -1: return []
        return name[start+1:-1].split(",")

    def _parse_compound_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []
