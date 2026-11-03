import collections
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Rover GTOHP domain.
    
    Bottleneck Analysis:
    The main bottleneck is navigation. Only one rover can traverse an edge at a time (implicitly serialized by the HTN structure),
    and rovers must move between distinct waypoints to sample soil/rock and reach visibility points to communicate data to landers.
    
    Admissibility Sketches:
    1. Goal Lower Bound: Each communicated_data goal that is not yet true requires at least one 'communicate' action (+1).
    2. Sampling Lower Bound: If no rover has the required analysis for a goal, a 'sample' action is mandatory (+1).
    3. Navigation Lower Bound: Moving a rover between waypoints costs 3*d-2 actions for d > 0 (navigate, visit, unvisit structure).
       Each target in the task network or goal set implies a minimum travel distance from the rover's current location.
    """

    def __init__(self, rover_gemini3flash_heuristic_17="rover_gemini3flash_heuristic_17"):
        super().__init__(rover_gemini3flash_heuristic_17)
        self.at_ids = {} # rover -> {wp -> gid}
        self.have_ids = {'soil': {}, 'rock': {}, 'image': {}} # type -> rover -> {target -> gid}
        self.goal_info = [] # list of (type, target, gid)
        self.rover_dists = {} # rover -> {wp1 -> {wp2 -> dist}}
        
        self.rovers = []
        self.waypoints = []
        self.lander_wps = []
        self.comm_points = set() # waypoints from which some lander is visible
        self.equipped = {'soil': set(), 'rock': set(), 'image': set()}

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
        # 1. Identify objects
        temp_rovers = set()
        temp_waypoints = set()
        
        # 2. Extract connectivity and capabilities from operators (static facts are stripped)
        adj = collections.defaultdict(lambda: collections.defaultdict(set))
        
        for op in model.operators:
            name = op.name.lower()
            _, args = self._parse_fact_name(op.name)
            
            if 'navigate' in name and len(args) >= 3:
                # navigate[rover, from, to]
                r, w1, w2 = args[0], args[1], args[2]
                temp_rovers.add(r)
                temp_waypoints.add(w1)
                temp_waypoints.add(w2)
                adj[r][w1].add(w2)
            
            if 'sample_soil' in name and len(args) >= 1:
                self.equipped['soil'].add(args[0])
            if 'sample_rock' in name and len(args) >= 1:
                self.equipped['rock'].add(args[0])
            if 'take_image' in name and len(args) >= 1:
                self.equipped['image'].add(args[0])
                
            if 'communicate' in name:
                # communicate_soil_data1[rover, lander, from, w1, w2] or communicate_soil_data2[rover, lander, from, w1]
                # w1 is usually the lander's waypoint or the rover's waypoint visible from it
                # Let's extract lander visibility via common patterns
                if 'communicate_soil_data1' in name or 'communicate_rock_data1' in name:
                    self.comm_points.add(args[3])
                elif 'communicate_soil_data2' in name or 'communicate_rock_data2' in name:
                    self.comm_points.add(args[2])
                elif 'communicate_image_data' in name:
                    self.comm_points.add(args[4])

        self.rovers = list(temp_rovers)
        self.waypoints = list(temp_waypoints)

        # 3. Precompute shortest paths per rover
        for r in self.rovers:
            self.rover_dists[r] = {w: {w2: float('inf') for w2 in self.waypoints} for w in self.waypoints}
            for w in self.waypoints:
                self.rover_dists[r][w][w] = 0
                for neighbor in adj[r][w]:
                    self.rover_dists[r][w][neighbor] = 1
            # Floyd-Warshall
            for k in self.waypoints:
                for i in self.waypoints:
                    for j in self.waypoints:
                        if self.rover_dists[r][i][j] > self.rover_dists[r][i][k] + self.rover_dists[r][k][j]:
                            self.rover_dists[r][i][j] = self.rover_dists[r][i][k] + self.rover_dists[r][k][j]

        # 4. Map predicates
        for r in self.rovers:
            self.at_ids[r] = {}
            for type_ in self.have_ids: self.have_ids[type_][r] = {}
            
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'at' and args[0] in self.at_ids:
                self.at_ids[args[0]][args[1]] = f.global_id
            elif pred == 'have_soil_analysis' and args[0] in self.have_ids['soil']:
                self.have_ids['soil'][args[0]][args[1]] = f.global_id
            elif pred == 'have_rock_analysis' and args[0] in self.have_ids['rock']:
                self.have_ids['rock'][args[0]][args[1]] = f.global_id
            elif pred == 'have_image' and args[0] in self.have_ids['image']:
                self.have_ids['image'][args[0]][args[1], args[2]] = f.global_id

        # 5. Index goals
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            fid = next(f.global_id for f in model.facts if f.name == g_name)
            if 'soil' in pred: self.goal_info.append(('soil', args[0], fid))
            elif 'rock' in pred: self.goal_info.append(('rock', args[0], fid))
            elif 'image' in pred: self.goal_info.append(('image', (args[0], args[1]), fid))

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        # A. Baseline for goals remaining
        h_goals = 0
        
        # B. Get current rover positions
        rover_pos = {}
        for r in self.rovers:
            for wp, gid in self.at_ids[r].items():
                if (state >> gid) & 1:
                    rover_pos[r] = wp
                    break
        
        # C. Process unsatisfied goals
        unsatisfied_goal_wps = []
        for g_type, g_target, g_fid in self.goal_info:
            if not (state >> g_fid) & 1:
                h_goals += 1 # At least one communicate action
                # Check if any rover already has it
                has_it = False
                for r in self.rovers:
                    gid = self.have_ids[g_type][r].get(g_target)
                    if gid is not None and (state >> gid) & 1:
                        has_it = True
                        break
                if not has_it:
                    h_goals += 1 # At least one sample/take_image action
                    target_wp = g_target if g_type != 'image' else None # objectives require visible_from wp
                    if target_wp:
                        unsatisfied_goal_wps.append(target_wp)
        
        # D. Process task network (P2: State-aware cost refinement)
        h_tasks = 0
        for task in task_network:
            name = task.name.lower()
            if 'navigate' in name:
                h_tasks += 1
            elif 'get_soil' in name or 'get_rock' in name:
                h_tasks += 3 # Base sequence: navigate, sample, communicate
            elif 'get_image' in name:
                h_tasks += 4 # Base sequence: calibrate, navigate, take_image, communicate

        # E. Navigation penalty (P4: Bottleneck travel cost)
        h_nav = 0
        if unsatisfied_goal_wps:
            # Add distance to nearest unsatisfied goal for each rover (very loose lower bound)
            min_navs = []
            for r, pos in rover_pos.items():
                dists = [self.rover_dists[r][pos][wp] for wp in unsatisfied_goal_wps if wp in self.rover_dists[r][pos]]
                if dists:
                    d = min(dists)
                    if d > 0: min_navs.append(3 * d - 2)
            if min_navs: h_nav = min(min_navs)

        # P1: Combine terms using max and conservative sum
        # Goals + tasks represent structural actions; h_nav represents travel required to start them.
        return max(h_goals, h_tasks) + h_nav
