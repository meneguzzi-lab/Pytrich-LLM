from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic10(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Towers domain is the serial constraint on ring movements: 
    only the top-most ring of a tower can be moved, and it can only be placed on a larger 
    object (either a larger ring or a tower base). This creates a recursive dependency 
    where moving a large ring requires clearing all smaller rings above it.
    
    Admissibility Sketches:
    1. Misplaced Rings (State-based): A ring is 'unstable' if it is not currently on its 
       goal object or if the object it is on is itself unstable. Every unstable ring 
       must be moved at least once to reach the goal state. Thus, the count of 
       unstable rings is an admissible lower bound on the number of moves.
    2. Task Network Complexity: Every task currently in the task network represents 
       future work. In this domain, every high-level task (shiftTower, rotateTower, etc.) 
       eventually decomposes into one or more primitive 'move' actions (unless the 
       stack is already empty). Counting the tasks provides a simple admissible bound 
       on the remaining decomposition steps.
    """

    def __init__(self, towers_gemini3flash_heuristic_10="TowersGemini3FlashHeuristic10"):
        super().__init__(towers_gemini3flash_heuristic_10)
        self.ring_goal = {}  # Map: ring_name -> target_obj_name
        self.on_fact_ids = {}  # Map: (ring_name, obj_name) -> global_id
        self.rings = []  # List of identified ring names
        self.towers = set()  # Set of identified tower base names

    def _parse_fact_name(self, name: str):
        """Parses grounded fact names into (predicate, [args])"""
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
        """Precomputes goal destinations and fact mappings."""
        self.model = model
        temp_rings = set()
        
        # 1. Parse all facts to identify rings, towers, and map 'on' predicates
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                temp_rings.add(args[0])
            elif pred == "towerTop":
                # towerTop(obj, tower) -> second arg is the tower base
                self.towers.add(args[1])
        
        self.rings = list(temp_rings)

        # 2. Extract goal 'on' requirements from the bitwise goal state
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                # Goal requires ring args[0] to be on args[1]
                self.ring_goal[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_stable(self, ring, state_bitwise, stable_cache):
        """
        Recursively determines if a ring is correctly placed.
        A ring is stable if it is on its goal object AND that object is stable.
        Towers are base cases for stability.
        """
        if ring in stable_cache:
            return stable_cache[ring]
        
        goal_obj = self.ring_goal.get(ring)
        if goal_obj is None:
            # Conservative: if no goal specified, assume stable
            stable_cache[ring] = True
            return True
        
        # Check if currently on the correct object
        on_id = self.on_fact_ids.get((ring, goal_obj))
        if on_id is None or not ((state_bitwise >> on_id) & 1):
            stable_cache[ring] = False
            return False
            
        # If on a tower base, it is stable
        if goal_obj in self.towers:
            stable_cache[ring] = True
            return True
            
        # If on another ring, that ring must also be stable
        res = self._is_stable(goal_obj, state_bitwise, stable_cache)
        stable_cache[ring] = res
        return res

    def _compute(self, state_bitwise, task_network):
        """Calculates the combined heuristic value."""
        # Bound 1: Misplaced/Unstable Rings
        stable_cache = {}
        misplaced_count = 0
        for r in self.rings:
            if not self._is_stable(r, state_bitwise, stable_cache):
                misplaced_count += 1
        
        # Bound 2: Task Network Size
        # Every task represents at least one move needed, though complex tasks represent many more.
        task_count = len(task_network)
        
        # Admissible combination: max of independent lower bounds
        return max(misplaced_count, task_count)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """Evaluates the heuristic for the current node."""
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
