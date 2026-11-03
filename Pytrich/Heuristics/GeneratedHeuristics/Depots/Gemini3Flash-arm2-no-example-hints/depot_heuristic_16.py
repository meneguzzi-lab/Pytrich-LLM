from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# BOTTLENECK ANALYSIS: 
# The Depots domain is a hybrid of Blocksworld and Logistics. The primary bottleneck is the serial constraints 
# on crate transport: each hoist can only handle one crate at a time (available/lifting), and crates must 
# be transported by trucks between locations, while also being managed in stacks (clear/on/at). 
# Efficiency depends on minimizing hoist actions (Lift/Drop/Load/Unload) and shared truck movements (Drive).

# ADMISSIBILITY SKETCHES:
# Term 1: h_goal. For every crate goal (on C S) not satisfied, at least 2 hoist actions (Lift/Drop) are needed.
# If C and S are at different locations, at least 2 additional hoist actions (Load/Unload) are needed, 
# plus 1 shared Drive action per place-to-place movement. This is a conservative lower bound.
# Additionally, for every crate above C or S in the stack, at least 2 actions are needed to clear it.
# Term 2: h_tn. Every non-nop task in the network (do_put_on, do_clear, etc.) must eventually transform 
# into at least its minimal number of grounded actions to be satisfied from the current state.
# Final h = max(h_goal, h_tn) is admissible because both components are independent lower bounds.

class DepotsGemini3FlashHeuristic16(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_16="depot_gemini3flash_heuristic_16"):
        super().__init__(depot_gemini3flash_heuristic_16)
        self.at_map = {} # obj -> place -> fact_id
        self.on_map = {} # crate -> surface -> fact_id
        self.on_facts_by_surface = {} # surface -> list of (f_id, crate_name)
        self.clear_map = {} # surface -> fact_id
        self.in_map = {} # crate -> truck -> fact_id
        self.lifting_map = {} # crate -> hoist -> fact_id
        self.hoist_at = {} # hoist_name -> place_name
        self.pallet_to_place = {} # pallet_name -> place_name
        self.goal_on = [] # list of (crate_name, surface_name)

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            pred, args = self._parse_fact_name(gn)
            if pred == "on":
                self.goal_on.append((args[0], args[1]))

        # 2. Parse facts and build maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "at":
                l, p = args[0], args[1]
                if l not in self.at_map: self.at_map[l] = {}
                self.at_map[l][p] = f.global_id
            elif pred == "on":
                c, s = args[0], args[1]
                if c not in self.on_map: self.on_map[c] = {}
                self.on_map[c][s] = f.global_id
                if s not in self.on_facts_by_surface: self.on_facts_by_surface[s] = []
                self.on_facts_by_surface[s].append((f.global_id, c))
            elif pred == "clear":
                self.clear_map[args[0]] = f.global_id
            elif pred == "in":
                c, t = args[0], args[1]
                if c not in self.in_map: self.in_map[c] = {}
                self.in_map[c][t] = f.global_id
            elif pred == "lifting":
                h, c = args[0], args[1]
                if c not in self.lifting_map: self.lifting_map[c] = {}
                self.lifting_map[c][h] = f.global_id

        # 3. Recover static locations from operator grounded names (PANDA Grounding Warning fix)
        for op in model.operators:
            # Pattern: Drop[hoist, crate, surface, place] or Drop hoist crate surface place
            name = op.name.replace("[", " ").replace("]", " ").replace(",", " ").replace("(", " ").replace(")", " ")
            parts = name.split()
            if len(parts) >= 5 and parts[0] in ["Drop", "Lift", "Load", "Unload"]:
                hoist, surface, place = parts[1], parts[3], parts[4]
                if "hoist" in hoist: self.hoist_at[hoist] = place
                if "pallet" in surface: self.pallet_to_place[surface] = place

        depot_gemini3flash_heuristic_16 = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, depot_gemini3flash_heuristic_16)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        depot_gemini3flash_heuristic_16 = self._compute(node.state, node.task_network)
        super().update_info(depot_gemini3flash_heuristic_16)
        return depot_gemini3flash_heuristic_16

    def _compute(self, state: int, task_network) -> int:
        h_goal = self._estimate_goal_cost(state)
        h_tn = self._estimate_tn_cost(state, task_network)
        # max is admissible for independent lower bounds
        return max(h_goal, h_tn)

    def _estimate_goal_cost(self, state):
        h = 0
        any_drive = False
        for c, s in self.goal_on:
            if not self._is_on(c, s, state):
                # Base hoist actions: Lift and Drop
                h += 2
                c_loc = self._get_loc(c, state)
                s_loc = self._get_loc(s, state)
                # Logistics overhead
                if c_loc and s_loc and c_loc != s_loc:
                    h += 2 # Load and Unload
                    any_drive = True
                # Clearance costs (2 per obstacle)
                h += 2 * self._count_above(c, state)
                h += 2 * self._count_above(s, state)
        if any_drive: h += 1 # At least one Drive
        return h

    def _estimate_tn_cost(self, state, task_network):
        h = 0
        for task in task_network:
            name, args = self._parse_name(task.name)
            if not name or not args: continue
            if name == "do_put_on":
                c, s = args[0], args[1]
                if not self._is_on(c, s, state):
                    h += 2
                    c_loc, s_loc = self._get_loc(c, state), self._get_loc(s, state)
                    if c_loc and s_loc and c_loc != s_loc: h += 2
                    h += 2 * (self._count_above(c, state) + self._count_above(s, state))
            elif name == "do_clear":
                h += 2 * self._count_above(args[0], state)
            elif name == "do_get_truck":
                if self._get_loc(args[0], state) != args[1]: h += 1
        return h

    def _count_above(self, surface, state):
        count = 0
        curr = surface
        while True:
            found = False
            for f_id, crate_name in self.on_facts_by_surface.get(curr, []):
                if (state >> f_id) & 1:
                    count += 1
                    curr = crate_name
                    found = True
                    break
            if not found: break
        return count

    def _get_loc(self, obj, state):
        if obj in self.at_map:
            for place, f_id in self.at_map[obj].items():
                if (state >> f_id) & 1: return place
        if obj in self.pallet_to_place: return self.pallet_to_place[obj]
        if obj in self.hoist_at: return self.hoist_at[obj]
        if obj in self.in_map:
            for truck, f_id in self.in_map[obj].items():
                if (state >> f_id) & 1: return self._get_loc(truck, state)
        if obj in self.lifting_map:
            for hoist, f_id in self.lifting_map[obj].items():
                if (state >> f_id) & 1: return self.hoist_at.get(hoist)
        return None

    def _is_on(self, crate, surface, state):
        f_id = self.on_map.get(crate, {}).get(surface)
        return (state >> f_id) & 1 if f_id is not None else False

    def _is_clear(self, surface, state):
        f_id = self.clear_map.get(surface)
        return (state >> f_id) & 1 if f_id is not None else False

    def _parse_name(self, name: str):
        clean = name.replace("[", " ").replace("]", " ").replace("(", " ").replace(")", " ").replace(",", " ")
        parts = clean.split()
        return (parts[0], parts[1:]) if parts else (None, [])

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []
