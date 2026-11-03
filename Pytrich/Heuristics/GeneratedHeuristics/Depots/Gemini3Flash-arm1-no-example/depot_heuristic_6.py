from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The single constraint that most serializes work is the requirement to use a single hoist 
# at a location for stacking/unstacking and the limited capacity of trucks for transporting 
# crates between locations, combined with surface clearance requirements.
#
# Admissibility Sketches:
# 1. Movement Cost: A crate at the wrong location requires at least Lift + Load + Drive + Unload + Drop (5 actions).
# 2. Stacking Cost: A crate at the correct location but on the wrong surface requires at least Lift + Drop (2 actions).
# 3. Clearance Cost: Every crate blocking a required crate or surface must be moved, costing at least 2 actions (Lift + Load/Drop).

class DepotsGemini3FlashHeuristic6(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_6="DepotsGemini3FlashHeuristic6"):
        super().__init__(depot_gemini3flash_heuristic_6)
        self.at_fact_list_by_obj = {}
        self.in_fact_list_by_crate = {}
        self.on_ids_by_crate_surface = {}
        self.clear_ids_by_surface = {}
        self.lifting_fact_list_by_crate = {}
        self.goal_on_ids = {}
        self.pallet_locs = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Fact parsing
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == 'at':
                obj, loc = args[0], args[1]
                self.at_fact_list_by_obj.setdefault(obj, []).append((loc, fact.global_id))
            elif pred == 'in':
                crate, truck = args[0], args[1]
                self.in_fact_list_by_crate.setdefault(crate, []).append((truck, fact.global_id))
            elif pred == 'on':
                crate, surface = args[0], args[1]
                self.on_ids_by_crate_surface.setdefault(crate, {})[surface] = fact.global_id
            elif pred == 'clear':
                surface = args[0]
                self.clear_ids_by_surface[surface] = fact.global_id
            elif pred == 'lifting':
                hoist, crate = args[0], args[1]
                self.lifting_fact_list_by_crate.setdefault(crate, []).append((hoist, fact.global_id))

        # Goal parsing
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == 'on':
                self.goal_on_ids[(args[0], args[1])] = self.on_ids_by_crate_surface.get(args[0], {}).get(args[1])

        # Pre-identify pallet locations (pallets are stationary)
        init_facts = model.state_explicit_repr(initial_node.state)
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'at':
                obj, loc = args[0], args[1]
                # Heuristic: simple naming convention usually identifies pallets/depots
                if "pallet" in obj:
                    self.pallet_locs[obj] = loc

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network) -> int:
        h = 0
        pursued_goals = set()

        for task in task_network:
            name, args = self._parse_task_name(task.name)
            if name == 'do_put_on':
                c, s = args[0], args[1]
                pursued_goals.add((c, s))
                # Check if satisfied
                fid = self.on_ids_by_crate_surface.get(c, {}).get(s)
                if fid is not None and (state >> fid) & 1: continue
                
                # Distance cost
                loc_c = self._find_loc(c, state)
                loc_s = self._find_loc(s, state)
                if loc_c and loc_s:
                    h += 2 if loc_c == loc_s else 5
                else:
                    h += 2 # Fallback
            elif name == 'do_clear':
                s = args[0]
                fid = self.clear_ids_by_surface.get(s)
                if fid is not None and not ((state >> fid) & 1): h += 2
            elif name == 'do_load_truck': h += 3
            elif name == 'do_unload_truck': h += 2
            elif name == 'do_lift_crate': h += 1
            elif name == 'do_get_truck':
                t, p = args[0], args[1]
                at_list = self.at_fact_list_by_obj.get(t, [])
                is_at = False
                for loc, fid in at_list: 
                    if loc == p and (state >> fid) & 1: 
                        is_at = True
                        break
                if not is_at: h += 1
            elif hasattr(task, 'cost'):
                h += task.cost
            elif name != 'nop':
                h += 1

        # Account for goals not currently in the task network
        for (c, s), fid in self.goal_on_ids.items():
            if (c, s) not in pursued_goals:
                if fid is not None and not ((state >> fid) & 1):
                    loc_c = self._find_loc(c, state)
                    loc_s = self._find_loc(s, state)
                    if loc_c and loc_s:
                        h += 2 if loc_c == loc_s else 5
                    else:
                        h += 2
        return h

    def _find_loc(self, obj, state):
        if obj in self.pallet_locs: return self.pallet_locs[obj]
        # Check direct at
        for loc, fid in self.at_fact_list_by_obj.get(obj, []):
            if (state >> fid) & 1: return loc
        # Check in truck
        for truck, fid in self.in_fact_list_by_crate.get(obj, []):
            if (state >> fid) & 1:
                for loc, tfid in self.at_fact_list_by_obj.get(truck, []):
                    if (state >> tfid) & 1: return loc
        # Check lifting
        for hoist, fid in self.lifting_fact_list_by_crate.get(obj, []):
            if (state >> fid) & 1:
                for loc, hfid in self.at_fact_list_by_obj.get(hoist, []):
                    if (state >> hfid) & 1: return loc
        return None

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []
