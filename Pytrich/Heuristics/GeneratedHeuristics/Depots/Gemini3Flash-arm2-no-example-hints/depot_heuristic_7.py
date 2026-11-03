from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class DepotsGemini3FlashHeuristic7(Heuristic):
    """
    Heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The bottleneck in the Depots domain is the serialization of clearing crates and vehicle movement. 
    A crate cannot be moved until it is clear, and the target surface cannot receive a crate until it is also clear.
    Additionally, crates must be moved between locations using trucks, which requires coordination of hoists and trucks.

    Admissible Lower Bounds:
    1. Goal-based stacking cost: Every goal 'on(C, S)' that is not satisfied requires at least a Lift and a Drop (2 actions).
       If C and S are at different places, it requires Load, Drive, Unload, and Drop (at least 4 actions).
    2. Task Network cost: Every task in the network represents pending work. 
       'do_put_on' costs at least 2-4 actions, 'do_clear' costs 2 actions per obstructing crate.
    3. Clear-obstruction cost: Each crate obstructing a goal crate or a goal surface must be moved,
       costing at least a Lift and a Load/Drop (2 actions per obstruction).
    """

    def __init__(self, depot_gemini3flash_heuristic_7="depots_heuristic"):
        super().__init__(name=depot_gemini3flash_heuristic_7)
        self.on_goals = []
        self.on_ids = collections.defaultdict(dict) # [crate][surface] -> fact_id
        self.at_ids = collections.defaultdict(dict) # [locatable][place] -> fact_id
        self.in_ids = collections.defaultdict(dict) # [crate][truck] -> fact_id
        self.clear_ids = {} # [surface] -> fact_id
        self.lifting_ids = collections.defaultdict(dict) # [hoist][crate] -> fact_id
        self.surface_at_static = {} # [surface] -> place
        self.crates = set()
        self.surfaces = set()
        self.places = set()
        self.trucks = set()

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == "on":
                self.on_ids[args[0]][args[1]] = f.global_id
                self.crates.add(args[0])
                self.surfaces.add(args[1])
            elif pred == "at":
                self.at_ids[args[0]][args[1]] = f.global_id
                self.places.add(args[1])
            elif pred == "in":
                self.in_ids[args[0]][args[1]] = f.global_id
                self.trucks.add(args[1])
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
                self.surfaces.add(args[0])
            elif pred == "lifting":
                self.lifting_ids[args[0]][args[1]] = f.global_id

        for op in model.operators:
            if "Drop" in op.name or "Lift" in op.name or "Unload" in op.name:
                parts = op.name.replace("[", ",").replace("]", "").split(",")
                if len(parts) >= 5:
                    surf, place = parts[3], parts[4]
                    self.surface_at_static[surf] = place

        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.on_goals.append((args[0], args[1]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, tn) -> int:
        crate_on = {}
        crate_at = {}
        is_clear = {}
        
        for c in self.crates:
            for s, fid in self.on_ids[c].items():
                if (state >> fid) & 1: 
                    crate_on[c] = s
                    break
            for p, fid in self.at_ids[c].items():
                if (state >> fid) & 1:
                    crate_at[c] = p
                    break

        for s in self.surfaces:
            fid = self.clear_ids.get(s)
            is_clear[s] = (state >> fid) & 1 if fid is not None else False

        def get_loc(item):
            if item in crate_at: return crate_at[item]
            return self.surface_at_static.get(item)

        def get_height(s):
            h = 0
            curr = s
            while True:
                top_crate = None
                for c in self.crates:
                    fid = self.on_ids[c].get(curr)
                    if fid is not None and (state >> fid) & 1:
                        top_crate = c
                        break
                if top_crate:
                    h += 1
                    curr = top_crate
                else:
                    break
            return h

        goal_h = 0
        for c, target_s in self.on_goals:
            fid = self.on_ids[c].get(target_s)
            if fid is not None and (state >> fid) & 1: continue
            
            loc_c = get_loc(c)
            loc_s = get_loc(target_s)
            
            if loc_c == loc_s:
                goal_h += 2
            else:
                goal_h += 4
            
            if not is_clear.get(c): goal_h += 2
            if not is_clear.get(target_s): goal_h += 2

        tn_h = 0
        for task in tn:
            if hasattr(task, 'name'):
                name = task.name
                if "do_put_on" in name:
                    tn_h += 2
                elif "do_clear" in name:
                    tn_h += 1
                elif "do_lift_crate" in name:
                    tn_h += 1
                elif "do_load_truck" in name:
                    tn_h += 2
                elif "do_unload_truck" in name:
                    tn_h += 2
                elif "Drive" in name:
                    tn_h += 1
                elif "Lift" in name or "Drop" in name or "Load" in name or "Unload" in name:
                    tn_h += 1

        return max(goal_h, tn_h)
