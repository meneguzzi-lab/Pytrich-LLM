from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

class DepotsGemini3FlashHeuristic11(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots domain.
    Bottleneck Analysis: Hoist availability and crate stack manipulation (clearing/unstacking) 
    are the primary serialization constraints. Trucks facilitate movement between locations,
    but the local hoist operations drive the plan complexity.
    
    Terms for Multi-component Max Heuristic:
    1. h_goal: Sum of estimated costs for crates to reach stable goal positions.
    2. h_tn: Sum of costs for remaining tasks in the task network.
    Admissibility Sketch: Costs are based on the minimum required actions (Lift, Drop, Load, Unload, Drive).
    Tie-breaking: A small penalty for the total length of the task network to guide search towards completion.
    """
    def __init__(self, depot_gemini3flash_heuristic_11="depot_gemini3flash_heuristic_11"):
        super().__init__(depot_gemini3flash_heuristic_11)
        self.at_ids = {}
        self.on_ids = {}
        self.in_truck_ids = {}
        self.clear_ids = {}
        self.lifting_ids = {}
        self.on_ids_by_surface = {}
        self.goal_map = {}
        self.pallets = set()
        self.hoist_to_place = {}
        self.pallet_to_place = {}
        self.all_relevant_crates = set()

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
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                self.at_ids.setdefault(args[0], []).append((args[1], f.global_id))
            elif pred == "on":
                self.on_ids.setdefault(args[0], []).append((args[1], f.global_id))
                self.on_ids_by_surface.setdefault(args[1], []).append((args[0], f.global_id))
            elif pred == "in":
                self.in_truck_ids.setdefault(args[0], []).append((args[1], f.global_id))
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "lifting":
                self.lifting_ids.setdefault(args[1], []).append((args[0], f.global_id))

        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred in ["Lift", "Drop", "Load", "Unload"]:
                self.hoist_to_place[args[0]] = args[-1]
                if pred in ["Lift", "Drop"]:
                    self.pallet_to_place[args[2]] = args[-1]
        
        for fid in model.initial_state:
            f = model.facts[fid]
            p, args = self._parse_fact_name(f.name)
            if p == "at":
                if args[0].startswith("pallet"):
                    self.pallets.add(args[0])
                    self.pallet_to_place[args[0]] = args[1]

        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            p, args = self._parse_fact_name(name)
            if p == "on":
                self.goal_map[args[0]] = args[1]
                self.all_relevant_crates.add(args[0])

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        loc_cache = {}
        stable_cache = {}
        
        h_goal = 0
        for crate, target_s in self.goal_map.items():
            h_goal += self._estimate_move_cost(crate, target_s, state, loc_cache, stable_cache)
            
        h_tn = 0
        for task in task_network:
            h_tn += self._get_task_cost(task, state, loc_cache, stable_cache)
            
        h_total = max(h_goal, h_tn) + 0.001 * len(task_network)
        return int(h_total)

    def _get_location(self, obj, state, cache):
        if obj in cache: return cache[obj]
        if obj in self.pallet_to_place: return self.pallet_to_place[obj]
        if obj in self.hoist_to_place: return self.hoist_to_place[obj]
        for p, fid in self.at_ids.get(obj, []):
            if (state >> fid) & 1:
                cache[obj] = p
                return p
        for t, fid in self.in_truck_ids.get(obj, []):
            if (state >> fid) & 1:
                loc = self._get_location(t, state, cache)
                cache[obj] = loc
                return loc
        for h, fid in self.lifting_ids.get(obj, []):
            if (state >> fid) & 1:
                loc = self._get_location(h, state, cache)
                cache[obj] = loc
                return loc
        return None

    def _is_stable(self, c, state, cache):
        if c in cache: return cache[c]
        target_s = self.goal_map.get(c)
        if not target_s: 
            cache[c] = True
            return True
        current_on = None
        for s, fid in self.on_ids.get(c, []):
            if (state >> fid) & 1:
                current_on = s
                break
        if current_on != target_s:
            cache[c] = False
            return False
        if target_s in self.pallets:
            cache[c] = True
            return True
        res = self._is_stable(target_s, state, cache)
        cache[c] = res
        return res

    def _get_stack_height(self, surface, state):
        count = 0
        curr = surface
        while True:
            found = False
            for crate, fid in self.on_ids_by_surface.get(curr, []):
                if (state >> fid) & 1:
                    count += 1
                    curr = crate
                    found = True
                    break
            if not found: break
        return count

    def _estimate_move_cost(self, c, target_s, state, loc_cache, stable_cache):
        if self._is_stable(c, state, stable_cache): return 0
        cost = 2
        loc_c = self._get_location(c, state, loc_cache)
        loc_s = self._get_location(target_s, state, loc_cache)
        if loc_c and loc_s and loc_c != loc_s:
            cost += 3
        cost += 2 * self._get_stack_height(c, state)
        cost += 2 * self._get_stack_height(target_s, state)
        return cost

    def _get_task_cost(self, task, state, loc_cache, stable_cache):
        name = task.name
        if "nop" in name or name.startswith("__method_precondition"): return 0
        pred, args = self._parse_fact_name("+" + name)
        if pred == "do_put_on":
            return self._estimate_move_cost(args[0], args[1], state, loc_cache, stable_cache)
        if pred == "do_clear":
            return 2 * self._get_stack_height(args[0], state)
        if pred == "do_get_truck":
            loc_t = self._get_location(args[0], state, loc_cache)
            return 1 if loc_t != args[1] else 0
        return 1
