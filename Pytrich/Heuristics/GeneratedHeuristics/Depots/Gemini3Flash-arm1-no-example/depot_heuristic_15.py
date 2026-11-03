from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic15(Heuristic):
    """
    Domain: Depots
    Bottleneck Analysis:
    The primary bottleneck in the Depots domain is the serialization of crate movements 
    at specific locations. Because a hoist can only handle one crate at a time and 
    crates must be clear to be moved, the stacking/unstacking constraints (Blocksworld-like) 
    combine with the logistics constraints (location-to-location travel) to form a high-cost 
    path for each crate goal.

    Admissibility Sketches:
    1. Goal Displacement Cost: Every unsatisfied on[crate, surface] goal requires a sequence 
       of physical actions. If the crate is at a different location than the surface, it 
       unavoidably requires at least {Lift, Load, Unload, Drop} (4 actions). If it is at 
       the same location, it requires at least {Lift, Drop} (2 actions).
    2. Clear/Available Constraints: A crate cannot be lifted unless it is clear, and a surface 
       cannot receive a crate unless it is clear. Each obstruction represents at least one 
       additional required action (though we keep this minimal to maintain admissibility).
    """
    def __init__(self, depot_gemini3flash_heuristic_15="DepotsHeuristic"):
        super().__init__(name=depot_gemini3flash_heuristic_15)
        self.on_goals = []  # List of (crate, surface, goal_fact_id)
        self.crate_at_map = {}  # crate -> {place -> fact_id}
        self.surface_at_map = {}  # surface -> {place -> fact_id}
        self.crate_in_map = {}  # crate -> {truck -> fact_id}
        self.hoist_lifting_map = {}  # hoist -> {crate -> fact_id}
        self.on_map = {}  # crate -> {surface -> fact_id}

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        # Pre-parse facts and build index maps
        for fact in model.facts:
            pred, args, prefix = self._parse_fact_name(fact.name)
            if not pred or prefix == "-":
                continue
            
            if pred == "at":
                locatable, place = args[0], args[1]
                if "crate" in locatable:
                    if locatable not in self.crate_at_map: self.crate_at_map[locatable] = {}
                    self.crate_at_map[locatable][place] = fact.global_id
                if "pallet" in locatable or "crate" in locatable:
                    if locatable not in self.surface_at_map: self.surface_at_map[locatable] = {}
                    self.surface_at_map[locatable][place] = fact.global_id
            elif pred == "on":
                crate, surface = args[0], args[1]
                if crate not in self.on_map: self.on_map[crate] = {}
                self.on_map[crate][surface] = fact.global_id
            elif pred == "in":
                crate, truck = args[0], args[1]
                if crate not in self.crate_in_map: self.crate_in_map[crate] = {}
                self.crate_in_map[crate][truck] = fact.global_id
            elif pred == "lifting":
                hoist, crate = args[0], args[1]
                if hoist not in self.hoist_lifting_map: self.hoist_lifting_map[hoist] = {}
                self.hoist_lifting_map[hoist][crate] = fact.global_id

        # Pre-identify goals from bitwise integer
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args, _ = self._parse_fact_name(g_name)
            if pred == "on":
                crate, surface = args[0], args[1]
                fact_id = self.on_map.get(crate, {}).get(surface)
                if fact_id is not None:
                    self.on_goals.append((crate, surface, fact_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network) -> int:
        h_goals = 0
        
        # 1. Goal-based displacement cost (Admissible)
        for crate, surface, goal_fid in self.on_goals:
            # If goal is already satisfied, cost is 0
            if (state >> goal_fid) & 1:
                continue
            
            # Find current location of surface
            surf_loc = None
            for loc, fid in self.surface_at_map.get(surface, {}).items():
                if (state >> fid) & 1:
                    surf_loc = loc
                    break
            
            # Find current location/status of crate
            crate_loc = None
            in_truck = False
            is_lifted = False
            
            # Check physical location
            for loc, fid in self.crate_at_map.get(crate, {}).items():
                if (state >> fid) & 1:
                    crate_loc = loc
                    break
            
            # If not at a location, it might be in a truck or lifted
            if crate_loc is None:
                for truck, fid in self.crate_in_map.get(crate, {}).items():
                    if (state >> fid) & 1:
                        in_truck = True
                        break
                if not in_truck:
                    for hoist_dict in self.hoist_lifting_map.values():
                        fid = hoist_dict.get(crate)
                        if fid and (state >> fid) & 1:
                            is_lifted = True
                            break
            
            # Accumulate minimum actions to achieve on[crate, surface]
            if is_lifted:
                h_goals += 1  # Action: Drop
            elif in_truck:
                h_goals += 2  # Actions: Unload, Drop
            elif crate_loc == surf_loc and crate_loc is not None:
                h_goals += 2  # Actions: Lift, Drop
            elif crate_loc != surf_loc:
                h_goals += 4  # Actions: Lift, Load, Unload, Drop
            else:
                h_goals += 2  # Default conservative lower bound

        # 2. Task Network Commitment
        # Every non-trivial task in the network corresponds to at least 1 action.
        h_tasks = 0
        for task in task_network:
            if "nop" not in task.name:
                h_tasks += 1

        # Return max to maintain admissibility while capturing both state and task info
        return max(h_goals, h_tasks)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
