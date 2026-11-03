from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic13(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots Domain.

    Bottleneck Analysis:
    The bottleneck in the Depots domain is the sequence of actions required to move crates across locations 
    and stack them correctly. Moving a crate requires lifting (or unloading), potentially driving (if cross-location), 
    and dropping. Stacking is constrained by 'clear' surfaces, meaning crates blocking a goal surface or a goal 
    crate must be moved first.

    Admissible Lower Bounds:
    1. Goal-Based Cost: Each misplaced crate requires at least a Lift/Unload and a Drop (cost 2). 
       If the current location differs from the goal location, it additionally requires a Load and a Drive (cost 2). 
       Clearing a blocked crate or surface adds at least 1 action (Lift).
    2. Task-Network Cost: Every remaining task in the network represents a commitment to future actions. 
       Decomposing 'do_put_on' will always result in at least 2-4 actions. 
       This is a lower bound on the remaining work to reach a terminal state.
    """

    def __init__(self, depot_gemini3flash_heuristic_13="depot_gemini3flash_heuristic_13"):
        super().__init__(name=depot_gemini3flash_heuristic_13)
        self.goals_map = {} # crate -> goal_surface
        self.crates = []
        self.trucks = []
        self.hoists = []
        self.places = []
        self.surfaces = []
        self.pallets = []

        self.at_ids = {} # locatable -> place -> global_id
        self.on_ids = {} # crate -> surface -> global_id
        self.in_ids = {} # crate -> truck -> global_id
        self.clear_ids = {} # surface -> global_id
        self.lifting_ids = {} # crate -> global_id
        self.hoist_at_ids = {} # hoist -> place -> global_id
        
        self.at_fact_list = []
        self.at_fact_map = {} # id -> (loc, place)
        self.pallet_places = {}

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
        goal_fact_names = model.state_explicit_repr(model.goals)
        for f_name in goal_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == "on":
                self.goals_map[args[0]] = args[1]

        crates_set = set()
        trucks_set = set()
        hoists_set = set()
        places_set = set()
        surfaces_set = set()

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                places_set.add(args[1])
                self.at_fact_list.append(f.global_id)
                self.at_fact_map[f.global_id] = (args[0], args[1])
                if args[0] not in self.at_ids: self.at_ids[args[0]] = {}
                self.at_ids[args[0]][args[1]] = f.global_id
            elif pred == "on":
                crates_set.add(args[0])
                surfaces_set.add(args[0])
                surfaces_set.add(args[1])
                if args[0] not in self.on_ids: self.on_ids[args[0]] = {}
                self.on_ids[args[0]][args[1]] = f.global_id
            elif pred == "in":
                trucks_set.add(args[1])
                if args[0] not in self.in_ids: self.in_ids[args[0]] = {}
                self.in_ids[args[0]][args[1]] = f.global_id
            elif pred == "clear":
                self.clear_ids[args[0]] = f.global_id
            elif pred == "lifting":
                self.lifting_ids[args[1]] = f.global_id
                if args[0] not in self.hoist_at_ids: self.hoist_at_ids[args[0]] = {}

        self.crates = list(crates_set)
        self.trucks = list(trucks_set)
        self.hoists = list(hoists_set)
        self.places = list(places_set)
        self.surfaces = list(surfaces_set)
        self.pallets = [s for s in self.surfaces if s not in crates_set]

        for fid in self.at_fact_list:
            if (model.initial_state >> fid) & 1:
                loc, p = self.at_fact_map[fid]
                if loc in self.pallets:
                    self.pallet_places[loc] = p

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> int:
        goal_h = 0
        
        current_at = {}
        for fid in self.at_fact_list:
            if (state >> fid) & 1:
                loc, p = self.at_fact_map[fid]
                current_at[loc] = p

        misplaced = set()
        for c in self.goals_map:
            s_g = self.goals_map[c]
            goal_fid = self.on_ids.get(c, {}).get(s_g)
            if goal_fid is None or not ((state >> goal_fid) & 1):
                misplaced.add(c)

        changed = True
        while changed:
            changed = False
            for c in self.goals_map:
                if c in misplaced: continue
                s_g = self.goals_map[c]
                if s_g in misplaced:
                    misplaced.add(c)
                    changed = True

        for c, s_g in self.goals_map.items():
            c_place = current_at.get(c)
            g_place = current_at.get(s_g) or self.pallet_places.get(s_g)
            
            if c not in misplaced:
                pass
            else:
                # Misplaced crate logic
                c_lifting = False
                for fid in self.lifting_ids.get(c, {}).values() if isinstance(self.lifting_ids.get(c), dict) else [self.lifting_ids.get(c)]:
                    if fid is not None and (state >> fid) & 1: 
                        c_lifting = True
                        break
                
                if c_lifting:
                    goal_h += 1
                else:
                    c_in_truck = None
                    for t in self.trucks:
                        if (state >> self.in_ids.get(c, {}).get(t, 0)) & 1:
                            c_in_truck = t
                            break
                    
                    if c_in_truck:
                        goal_h += 2
                        if current_at.get(c_in_truck) != g_place:
                            goal_h += 1
                    else:
                        goal_h += 2
                        if c_place != g_place:
                            goal_h += 2
                
                if not (state >> self.clear_ids.get(c, 0)) & 1: goal_h += 1
                if not (state >> self.clear_ids.get(s_g, 0)) & 1: goal_h += 1

        task_h = 0
        for task in task_network:
            if hasattr(task, 'decompositions'): # AbstractTask
                if "do_put_on" in task.name: task_h += 2
                else: task_h += 1
            else: # Operator
                task_h += 1

        return max(goal_h, task_h)
