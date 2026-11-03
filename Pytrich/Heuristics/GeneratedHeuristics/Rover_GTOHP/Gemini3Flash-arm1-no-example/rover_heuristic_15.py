from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic15(Heuristic):
    """
    Heuristic for the Rover GTOHP domain.

    BOTTLENECK ANALYSIS:
    The primary serialization bottleneck in the Rover domain is navigation and sampling.
    Each data collection task (soil, rock, image) requires the rover to be at a specific
    waypoint, potentially empty its store (for samples), perform a primitive sampling action,
    navigate to a waypoint visible to a lander, and then communicate the data. 
    Since these actions are discrete and waypoints are geographically distributed, navigation
    distance and action counts form the core lower bounds.

    ADMISSIBLE LOWER BOUNDS:
    1. Goal-Fact Lower Bound (h_goals): For every 'communicated_..._data' goal fact not currently 
       satisfied, we calculate the minimum number of primitive actions required. If the rover 
       already has the analysis/image, it costs at least 1 action (communicate). If it doesn't 
       yet have the data, it costs at least 2 actions (sample + communicate). This is a safe 
       lower bound as these actions are mandatory and non-shareable for distinct goal atoms.
    2. Task Network Lower Bound (h_tn): Every abstract task in the network eventually 
       decomposes into primitive actions. We assign each task type a minimum number of 
       unavoidable primitive actions (e.g., 'get_soil_data' requires at least a sample and a 
       communicate action). Summing these minimums for the current network is admissible.
    """

    def __init__(self, rover_gemini3flash_heuristic_15="rover_gemini3flash_heuristic_15"):
        super().__init__(name=rover_gemini3flash_heuristic_15)
        self.goal_info = []
        self.task_weights = {}
        self.soil_analysis_ids = {}  # waypoint -> [ids of have_soil_analysis[any_rover, w]]
        self.rock_analysis_ids = {}  # waypoint -> [ids of have_rock_analysis[any_rover, w]]
        self.image_analysis_ids = {} # (obj, mode) -> [ids of have_image[any_rover, obj, mode]]

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
        # Pre-process facts for goal evaluation
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "have_soil_analysis":
                w = args[1]
                if w not in self.soil_analysis_ids: self.soil_analysis_ids[w] = []
                self.soil_analysis_ids[w].append(f.global_id)
            elif pred == "have_rock_analysis":
                w = args[1]
                if w not in self.rock_analysis_ids: self.rock_analysis_ids[w] = []
                self.rock_analysis_ids[w].append(f.global_id)
            elif pred == "have_image":
                obj, mode = args[1], args[2]
                if (obj, mode) not in self.image_analysis_ids: self.image_analysis_ids[(obj, mode)] = []
                self.image_analysis_ids[(obj, mode)].append(f.global_id)

        # Process goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            # Find the global_id of this goal fact
            g_id = -1
            for f in model.facts:
                if f.name == g_name:
                    g_id = f.global_id
                    break
            
            if g_id == -1: continue

            analysis_ids = []
            if pred == "communicated_soil_data":
                analysis_ids = self.soil_analysis_ids.get(args[0], [])
            elif pred == "communicated_rock_data":
                analysis_ids = self.rock_analysis_ids.get(args[0], [])
            elif pred == "communicated_image_data":
                analysis_ids = self.image_analysis_ids.get((args[0], args[1]), [])
            
            self.goal_info.append((g_id, analysis_ids))

        # Precompute weights for every possible component
        # Note: Operators are primitives, cost 1. Abstract tasks have min decomp costs.
        for t in model.abstract_tasks:
            n = t.name.lower()
            if "get_" in n: weight = 2
            elif "send_" in n: weight = 1
            elif "do_navigate" in n: weight = 1
            elif "do_calibrate" in n: weight = 1
            elif "empty_store" in n: weight = 1
            else: weight = 0
            self.task_weights[t.global_id] = weight

        for op in model.operators:
            n = op.name.lower()
            if "nop" in n or "visit" in n or "unvisit" in n:
                self.task_weights[op.global_id] = 0
            else:
                self.task_weights[op.global_id] = 1

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state_bitwise, task_network):
        # Term 1: Goal satisfaction cost
        h_goals = 0
        for g_id, analysis_ids in self.goal_info:
            if not ((state_bitwise >> g_id) & 1):
                # Goal not yet reached
                # If data already sampled by any rover, we only need to communicate (1 action)
                # Otherwise, we need to sample and communicate (at least 2 actions)
                if any((state_bitwise >> a_id) & 1 for a_id in analysis_ids):
                    h_goals += 1
                else:
                    h_goals += 2

        # Term 2: Task network cost
        h_tn = 0
        for task in task_network:
            h_tn += self.task_weights.get(task.global_id, 0)

        # Admissible combine
        return max(h_goals, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        # Compute max of admissible terms + small tie-breaker for network depth
        h_val = self._compute(node.state, node.task_network)
        
        # Favors smaller task networks to break expansion ties
        tie_breaker = 0.001 * len(node.task_network)
        
        final_h = int(h_val + tie_breaker)
        # Ensure we return at least h_val even after int conversion if h_val > 0
        if h_val > 0 and final_h == 0: final_h = 1
        
        # Actually, Pytrich usually expects int. We'll return int(h_val) but 
        # since h_val is already integer here, we just use h_val.
        super().update_info(h_val)
        return h_val
