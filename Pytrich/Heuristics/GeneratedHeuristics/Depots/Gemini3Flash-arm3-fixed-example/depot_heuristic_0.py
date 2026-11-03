import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    Bottleneck Analysis:
    The primary bottleneck in Depots is the serial nature of stack manipulation at specific places.
    Crates must be cleared before moving, and target surfaces must be cleared before dropping.
    Transport requires a truck to be present, which is a secondary resource constraint.

    Admissible Lower Bounds:
    1. Transport/Placement Cost: Every crate not on its goal surface requires at least 2 actions (Lift/Drop)
       if at the goal location, or 5 actions (Lift/Load/Drive/Unload/Drop) if at a different location.
    2. Clearance Cost: For every crate between a target crate and the top of its stack, at least 2 actions 
       are required to move it out of the way. Similarly for the target surface stack.
    3. Task Network Guidance: The number of non-nop tasks in the network represents a minimum count of work remaining.
    """

    def __init__(self, depot_gemini3flash_heuristic_0=None, name="DepotsGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.goal_on = {} # crate -> surface
        self.goal_place = {} # crate -> place
        self.pallet_to_place = {} # pallet -> place
        
        self.on_fact_info = {} # f_id -> (crate, surface)
        self.at_fact_info = {} # f_id -> (locatable, place)
        self.in_fact_info = {} # f_id -> (crate, truck)
        self.lifting_fact_info = {} # f_id -> (hoist, crate)
        self.clear_fact_ids = {} # surface -> f_id
        
        self.on_mask = 0
        self.at_mask = 0
        self.in_mask = 0
        self.lifting_mask = 0
        
        self.crate_ids = set()
        self.surface_ids = set()

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
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map pallets to places (static)
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at" and len(args) == 2:
                obj, place = args[0], args[1]
                # Check if obj is a pallet (only way to know is name or types, but pallets are surfaces)
                if "pallet" in obj.lower():
                    self.pallet_to_place[obj] = place

        # 2. Extract Goals
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                self.goal_on[args[0]] = args[1]

        # 3. Fact Indexing and Masks
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_fact_info[f.global_id] = (args[0], args[1])
                self.on_mask |= (1 << f.global_id)
                self.crate_ids.add(args[0])
                self.surface_ids.add(args[1])
            elif pred == "at":
                self.at_fact_info[f.global_id] = (args[0], args[1])
                self.at_mask |= (1 << f.global_id)
            elif pred == "in":
                self.in_fact_info[f.global_id] = (args[0], args[1])
                self.in_mask |= (1 << f.global_id)
            elif pred == "lifting":
                self.lifting_fact_info[f.global_id] = (args[0], args[1])
                self.lifting_mask |= (1 << f.global_id)
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id

        # 4. Precompute Goal Places for Crates
        for crate in self.crate_ids:
            curr = crate
            visited = {curr}
            while curr in self.goal_on:
                target = self.goal_on[curr]
                if target in self.pallet_to_place:
                    self.goal_place[crate] = self.pallet_to_place[target]
                    break
                if target in visited: break
                visited.add(target)
                curr = target

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _get_num_above(self, surface, curr_on):
        count = 0
        curr = surface
        while curr in curr_on:
            count += 1
            curr = curr_on[curr]
        return count

    def _compute(self, state, task_network):
        curr_on = {}
        temp = state & self.on_mask
        while temp:
            lsb = temp & -temp
            fid = lsb.bit_length() - 1
            c, s = self.on_fact_info[fid]
            curr_on[s] = c
            temp ^= lsb
            
        curr_at = {}
        temp = state & self.at_mask
        while temp:
            lsb = temp & -temp
            fid = lsb.bit_length() - 1
            obj, p = self.at_fact_info[fid]
            curr_at[obj] = p
            temp ^= lsb

        curr_in = {}
        temp = state & self.in_mask
        while temp:
            lsb = temp & -temp
            fid = lsb.bit_length() - 1
            c, t = self.in_fact_info[fid]
            curr_in[c] = t
            temp ^= lsb
            
        curr_lifting = {}
        temp = state & self.lifting_mask
        while temp:
            lsb = temp & -temp
            fid = lsb.bit_length() - 1
            h, c = self.lifting_fact_info[fid]
            curr_lifting[c] = h
            temp ^= lsb

        h_goals = 0
        for crate, g_surface in self.goal_on.items():
            is_on_goal = curr_on.get(g_surface) == crate
            g_place = self.goal_place.get(crate)
            
            # Find current place
            c_place = curr_at.get(crate)
            if c_place is None:
                if crate in curr_in:
                    c_place = curr_at.get(curr_in[crate])
                elif crate in curr_lifting:
                    c_place = curr_at.get(curr_lifting[crate])
            
            if is_on_goal:
                if c_place != g_place:
                    h_goals += 5 # Must eventually move
            else:
                # Placement cost
                if c_place == g_place:
                    h_goals += 2
                else:
                    h_goals += 5
                # Clearance cost
                h_goals += self._get_num_above(crate, curr_on) * 2
                h_goals += self._get_num_above(g_surface, curr_on) * 2

        h_tasks = sum(1 for t in task_network if t.name != "nop")
        return h_goals + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
