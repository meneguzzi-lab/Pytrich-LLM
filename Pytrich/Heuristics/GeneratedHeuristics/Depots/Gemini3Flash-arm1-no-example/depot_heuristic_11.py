from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from typing import List, Union

# Bottleneck Analysis:
# The Depots domain combines logistics and stacking logic. The primary bottleneck is the 
# serializing nature of stacking/unstacking crates at locations. To move or use a crate, 
# every crate above it must be cleared. To place a crate on a target, the target must also be clear.
# This leads to a cost that grows linearly with the height of the stacks involved.

# Admissibility Sketches:
# 1. Network Cost (State-Aware): For each task in the network, we estimate the minimum number of actions 
#    needed to fulfill it based on the current state (e.g., location of crates, height of stacks). 
#    Since each required action (Lift, Drop, Drive, Load, Unload) contributes 1 to the plan length,
#    summing these minimum required counts (while accounting for stacking clearances) provides a 
#    realistic lower bound for satisfying the remaining task network.
# 2. Distance/Move Cost: Moving a crate between different locations requires a sequence of exactly 
#    5 actions (Lift, Load, Drive, Unload, Drop). This is an absolute lower bound for any cross-location move.

class DepotsGemini3FlashHeuristic11(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_11="depot_gemini3flash_heuristic_11"):
        super().__init__(depot_gemini3flash_heuristic_11)
        self.at_ids_by_obj = {}
        self.on_facts = []
        self.in_ids_by_obj = {}
        self.lifting_ids_by_obj = {}
        self.on_ids = {}
        self.hoist_at = {}
        self.static_pallet_at = {}
        self.all_places = []
        self.all_trucks = []
        self.all_hoists = []
        self.all_crates = []

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

    def _parse_task(self, name: str):
        parts = name.split()
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        places = set()
        trucks = set()
        hoists = set()
        crates = set()
        pallets = set()

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue

            if pred == 'at':
                obj, loc = args[0], args[1]
                if obj not in self.at_ids_by_obj: self.at_ids_by_obj[obj] = []
                self.at_ids_by_obj[obj].append((loc, f.global_id))
                places.add(loc)
            elif pred == 'on':
                self.on_facts.append((args[0], args[1], f.global_id))
                self.on_ids[(args[0], args[1])] = f.global_id
                crates.add(args[0])
            elif pred == 'in':
                if args[0] not in self.in_ids_by_obj: self.in_ids_by_obj[args[0]] = []
                self.in_ids_by_obj[args[0]].append((args[1], f.global_id))
                trucks.add(args[1])
            elif pred == 'lifting':
                if args[1] not in self.lifting_ids_by_obj: self.lifting_ids_by_obj[args[1]] = []
                self.lifting_ids_by_obj[args[1]].append((args[0], f.global_id))
                hoists.add(args[0])

        self.all_places = list(places)
        self.all_trucks = list(trucks)
        self.all_hoists = list(hoists)
        self.all_crates = list(crates)

        # Find static locations for hoists and pallets
        for h in self.all_hoists:
            for p in self.all_places:
                g_id = next((gid for loc, gid in self.at_ids_by_obj.get(h, []) if loc == p), None)
                if g_id is not None and (initial_node.state >> g_id) & 1:
                    self.hoist_at[h] = p
                    break

        # Identify pallets (surfaces that have 'at' facts in init and are not crates)
        for obj, facts in self.at_ids_by_obj.items():
            if obj not in self.all_crates and obj not in self.all_trucks and obj not in self.all_hoists and obj not in self.all_places:
                for p, g_id in facts:
                    if (initial_node.state >> g_id) & 1:
                        self.static_pallet_at[obj] = p
                        break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_location(self, obj_name, state):
        if obj_name in self.static_pallet_at: return self.static_pallet_at[obj_name]
        # 1. Check at(obj, p)
        for p, g_id in self.at_ids_by_obj.get(obj_name, []):
            if (state >> g_id) & 1: return p
        # 2. Check in(obj, t)
        for t, g_id in self.in_ids_by_obj.get(obj_name, []):
            if (state >> g_id) & 1:
                for p_t, at_g_id in self.at_ids_by_obj.get(t, []):
                    if (state >> at_g_id) & 1: return p_t
        # 3. Check lifting(h, obj)
        for h, g_id in self.lifting_ids_by_obj.get(obj_name, []):
            if (state >> g_id) & 1: return self.hoist_at.get(h)
        return None

    def _get_stack_height(self, surface_name, on_map):
        count = 0
        curr = surface_name
        visited = set()
        while curr in on_map and curr not in visited:
            visited.add(curr)
            count += 1
            curr = on_map[curr]
        return count

    def _compute(self, state, task_network):
        on_map = {}
        for c_name, s_name, g_id in self.on_facts:
            if (state >> g_id) & 1:
                on_map[s_name] = c_name

        total_h = 0
        for task in task_network:
            name, args = self._parse_task(task.name)
            if name == 'nop': continue
            if name in ['Drive', 'Lift', 'Drop', 'Load', 'Unload']: 
                total_h += 1
                continue

            if name == 'do_put_on':
                c, s = args[0], args[1]
                if (state >> self.on_ids.get((c, s), -1)) & 1: continue
                
                p_c = self._get_location(c, state)
                p_s = self._get_location(s, state)
                
                cost = 2 # Base Lift + Drop
                if p_c and p_s and p_c != p_s: cost += 3 # Load + Drive + Unload
                cost += 2 * self._get_stack_height(c, on_map)
                cost += 2 * self._get_stack_height(s, on_map)
                total_h += cost

            elif name == 'do_clear':
                total_h += 2 * self._get_stack_height(args[0], on_map)

            elif name == 'do_get_truck':
                t, p = args[0], args[1]
                truck_at = next((loc for loc, gid in self.at_ids_by_obj.get(t, []) if (state >> gid) & 1), None)
                if truck_at != p: total_h += 1

            elif name == 'do_lift_crate':
                total_h += 1

            elif name == 'do_load_truck':
                c = args[0]
                total_h += 2 + 2 * self._get_stack_height(c, on_map)

            elif name == 'do_unload_truck':
                s = args[1]
                total_h += 2 + 2 * self._get_stack_height(s, on_map)
            
            else:
                total_h += 1

        # Sub-unit tie-breaker to prefer smaller task networks
        total_h += 0.001 * len(task_network)
        return int(total_h)
