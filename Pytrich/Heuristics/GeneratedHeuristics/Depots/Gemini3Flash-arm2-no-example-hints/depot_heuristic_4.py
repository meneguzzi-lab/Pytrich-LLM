from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic4(Heuristic):
    """
    Bottleneck Analysis:
    The stacking and transportation constraints represent the primary bottleneck. A crate must be cleared (similar to Blocksworld)
    before it can be lifted, and then it must be moved (possibly between locations via trucks) to its goal surface.

    Admissibility Sketches:
    1. h_goal: For each crate not 'correctly placed' (on its goal surface, and that surface is either a pallet or a correctly
       placed crate), we count a minimum of 2 actions (Lift, Drop). If the crate is at a different location than its goal,
       the cost is at least 5 (Lift, Load, Drive, Unload, Drop).
    2. h_clear: For every crate that needs to move (misplaced), every crate currently above it in a stack must also be moved.
       Each such 'clearing' movement costs at least 2 actions.
    3. h_tn: The number of non-operator tasks in the HTN network provides a lower bound on the remaining high-level work.
    """

    def __init__(self, depot_gemini3flash_heuristic_4="depot_gemini3flash_heuristic_4"):
        super().__init__(depot_gemini3flash_heuristic_4)
        self.on_goals = {} # crate_id -> surface_id
        self.pallet_locs = {} # pallet_id -> place_id
        self.hoist_locs = {} # hoist_id -> place_id
        self.is_pallet = set()
        self.is_crate = set()
        self.is_place = set()
        self.fact_id_to_pred = {}

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map all facts
        self.on_fact_ids = {} # (crate, surface) -> global_id
        self.at_fact_ids = {} # (locatable, place) -> global_id
        self.clear_fact_ids = {} # (surface) -> global_id
        
        # Reverse maps for state analysis
        self.crate_on_surface_facts = {} # crate_id -> list of (surface_id, global_id)

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            arg_ids = [model.desc.get_id(a) for a in args]
            
            if pred == "on":
                self.on_fact_ids[(arg_ids[0], arg_ids[1])] = f.global_id
                if arg_ids[0] not in self.crate_on_surface_facts:
                    self.crate_on_surface_facts[arg_ids[0]] = []
                self.crate_on_surface_facts[arg_ids[0]].append((arg_ids[1], f.global_id))
                self.is_crate.add(arg_ids[0])
            elif pred == "at":
                self.at_fact_ids[(arg_ids[0], arg_ids[1])] = f.global_id
                # Pallets and places are static-ish. Hoists too.
                # Determine types from initial state
                is_true = (initial_node.state >> f.global_id) & 1
                if is_true:
                    # Heuristic guess for static types based on initial state
                    # PANDA grounder strips typing, but 'at' for pallets/places is usually persistent
                    pass
            elif pred == "clear":
                self.clear_fact_ids[arg_ids[0]] = f.global_id

        # 2. Extract Goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                c_id = model.desc.get_id(args[0])
                s_id = model.desc.get_id(args[1])
                self.on_goals[c_id] = s_id

        # 3. Recover location info from operators (PANDA hack)
        for op in model.operators:
            if "Drop" in op.name or "Lift" in op.name:
                # Format: Drop[hoist,crate,surface,place]
                parts = re.split(r'[\[,\],]+', op.name)
                if len(parts) >= 5:
                    s_id = model.desc.get_id(parts[3])
                    p_id = model.desc.get_id(parts[4])
                    self.pallet_locs[s_id] = p_id
                    self.is_place.add(p_id)

        return super().initialize(model, self._compute(initial_node.state, initial_node.task_network))

    def _get_current_on(self, state_bitwise, crate_id):
        if crate_id not in self.crate_on_surface_facts:
            return None
        for s_id, f_id in self.crate_on_surface_facts[crate_id]:
            if (state_bitwise >> f_id) & 1:
                return s_id
        return None

    def _get_crate_above(self, state_bitwise, surface_id):
        # Which crate is 'on' this surface?
        for c_id, s_ids in self.on_fact_ids.items():
            if c_id[1] == surface_id:
                if (state_bitwise >> s_ids) & 1:
                    return c_id[0]
        return None

    def _get_location(self, state_bitwise, item_id):
        # item_id could be a crate, truck, or pallet
        if item_id in self.pallet_locs:
            return self.pallet_locs[item_id]
        # If crate, find what it is on
        curr = item_id
        visited = {curr}
        while True:
            on_s = self._get_current_on(state_bitwise, curr)
            if on_s is None or on_s in visited:
                break
            if on_s in self.pallet_locs:
                return self.pallet_locs[on_s]
            curr = on_s
            visited.add(curr)
        return None

    def _compute(self, state, task_network):
        # h_goal calculation
        correctly_placed = set()
        
        # Pallets are always correctly placed bases
        for p_id in self.pallet_locs:
            correctly_placed.add(p_id)

        # Fixed-point iteration for correctly placed crates
        changed = True
        while changed:
            changed = False
            for crate_id, goal_surface_id in self.on_goals.items():
                if crate_id in correctly_placed: continue
                
                # Is it on its goal surface?
                current_on = self._get_current_on(state, crate_id)
                if current_on == goal_surface_id:
                    # Is its goal surface correctly placed?
                    if goal_surface_id in correctly_placed:
                        correctly_placed.add(crate_id)
                        changed = True

        h_dist = 0
        misplaced_crates = []
        for crate_id in self.on_goals:
            if crate_id not in correctly_placed:
                misplaced_crates.append(crate_id)
                
                # Base cost for moving the crate
                goal_surface = self.on_goals[crate_id]
                curr_loc = self._get_location(state, crate_id)
                goal_loc = self._get_location(state, goal_surface)
                
                if curr_loc is not None and goal_loc is not None and curr_loc != goal_loc:
                    h_dist += 5 # Lift, Load, Drive, Unload, Drop
                else:
                    h_dist += 2 # Lift, Drop

        # h_clear calculation: for every misplaced crate, count crates above it
        for crate_id in misplaced_crates:
            # If there's a crate on top, it must be cleared
            above = self._get_crate_above(state, crate_id)
            if above is not None:
                # This crate 'above' must be moved to clear 'crate_id'
                # We assign +2 for the clearing movement
                h_dist += 2

        # h_tn: Number of non-primitive tasks
        tn_h = 0
        for task in task_network:
            if hasattr(task, 'decompositions'): # Abstract task
                tn_h += 2 # Rough estimate
            else:
                # If it's a nop, it's free. Operators cost 1.
                if "nop" not in task.name.lower():
                    tn_h += 1

        return max(h_dist, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
