from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.
    
    Bottleneck Analysis:
    The main bottleneck in the Rover domain is the sequential nature of data lifecycle: 
    A rover must first be at a specific waypoint, ensure its store is empty, sample soil/rock 
    or take an image (after calibration), and then travel to a waypoint visible to a lander 
    to communicate that data. The primary cost is navigation and the actions required for 
    each data acquisition and transmission.

    Admissibility Sketches:
    - h_data (Goal Term): For every goal fact 'communicated_data' missing from the state, at 
      least 1 action (communication) is required. If no rover currently 'has_data' for that 
      target, at least one additional action (sample/take_image) is unavoidable. This is a 
      lower bound on the actions needed to fulfill pending goals.
    - h_tn (Task Term): Each task in the hierarchical network must eventually decompose into 
      at least its minimum primitive action count (e.g., get_soil_data needs at least 
      sample + communicate). This is a lower bound based on the remaining work in the HTN.
    """

    def __init__(self, rover_gemini3flash_heuristic_9="rover_gemini_heuristic"):
        super().__init__(name=rover_gemini3flash_heuristic_9)
        self.goal_info = [] # List of (type, args, goal_bit_id)
        self.have_analysis_ids = {} # (type, target_args) -> list of rover bit_ids
        self.task_min_costs = {
            "get_soil_data": 2,      # sample + communicate
            "get_rock_data": 2,      # sample + communicate
            "get_image_data": 3,     # calibrate + take_image + communicate
            "send_soil_data": 1,     # communicate
            "send_rock_data": 1,     # communicate
            "send_image_data": 1,    # communicate
            "do_calibrate": 1,       # calibrate
            "sample_soil": 1,
            "sample_rock": 1,
            "take_image": 1,
            "calibrate": 1,
            "communicate_soil_data1": 1,
            "communicate_soil_data2": 1,
            "communicate_rock_data1": 1,
            "communicate_rock_data2": 1,
            "communicate_image_data": 1,
            "navigate": 1,
            "drop": 1,
            "visit": 1,
            "unvisit": 1
        }

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Precompute goal information
        goal_names = self.model.state_explicit_repr(self.model.goals)
        
        # Map communicated goals to their corresponding 'have' predicates
        # communicated_soil_data[waypoint]
        # communicated_rock_data[waypoint]
        # communicated_image_data[objective, mode]
        
        self.goal_list = []
        fact_map = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            fact_map[f.name] = f.global_id
            
            if pred == "have_soil_analysis":
                key = ("soil", args[1]) # waypoint
                self.have_analysis_ids.setdefault(key, []).append(f.global_id)
            elif pred == "have_rock_analysis":
                key = ("rock", args[1]) # waypoint
                self.have_analysis_ids.setdefault(key, []).append(f.global_id)
            elif pred == "have_image":
                key = ("image", args[1], args[2]) # objective, mode
                self.have_analysis_ids.setdefault(key, []).append(f.global_id)

        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "communicated_soil_data":
                self.goal_list.append(("soil", args[0], fact_map[g_name]))
            elif pred == "communicated_rock_data":
                self.goal_list.append(("rock", args[0], fact_map[g_name]))
            elif pred == "communicated_image_data":
                self.goal_list.append(("image", args[0], args[1], fact_map[g_name]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network) -> int:
        # 1. State-based Goal term (h_data)
        h_data = 0
        for goal in self.goal_list:
            g_type = goal[0]
            if not ((state >> goal[-1]) & 1): # Goal not reached
                h_data += 1 # At least one communication action needed
                
                # Determine lookup key for 'have_data'
                if g_type == "soil" or g_type == "rock":
                    key = (g_type, goal[1]) # type, waypoint
                else: # image
                    key = ("image", goal[1], goal[2]) # type, objective, mode
                
                # Check if any rover holds this analysis
                held = False
                for rid in self.have_analysis_ids.get(key, []):
                    if (state >> rid) & 1:
                        held = True
                        break
                if not held:
                    h_data += 1 # At least one acquisition action needed

        # 2. HTN Task Network term (h_tn)
        h_tn = 0
        for task in task_network:
            base_name = task.name.split('[')[0]
            h_tn += self.task_min_costs.get(base_name, 1 if base_name != "nop" else 0)
            
        return max(h_data, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
