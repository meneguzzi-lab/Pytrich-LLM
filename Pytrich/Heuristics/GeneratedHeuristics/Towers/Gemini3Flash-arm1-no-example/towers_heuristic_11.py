from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic11(Heuristic):
    """
    Towers Domain Heuristic
    
    1. Bottleneck Analysis:
    The primary bottleneck in the Towers (Tower of Hanoi) domain is the sequential stack dependency. 
    A ring can only be moved if it is at the top of a tower, and it can only be placed on a larger 
    ring or an empty tower. This creates a recursive structure where moving a large ring 
    requires first moving all smaller rings to a temporary location.

    2. Admissible Lower Bounds:
    - h_unstable: A ring is considered 'unstable' if it is not in its goal position OR if the ring 
      below it (its target) is not yet stable. Every unstable ring must be moved at least once. 
      This is a lower bound on actions required to fix the state.
    - h_tasks: Each task in the task network represents at least one future action or a decomposition 
      step. In the Towers domain, every task (like shiftTower, move_abstract, rotateTower) eventually 
      leads to at least one 'move' operator or serves as a necessary step for one. This is a lower 
      bound on the remaining effort to clear the hierarchy.

    3. Design Strategy:
    The heuristic uses a bottom-up goal stability check to count misplaced work and combines it 
    with the task network length. This provides a strong gradient for both state-based goal 
    achievement and hierarchical progress.
    """

    def __init__(self, towers_gemini3flash_heuristic_11="TowersGemini3FlashHeuristic11"):
        super().__init__(towers_gemini3flash_heuristic_11)
        self.on_ids = {}      # ring_name -> {target_name: global_id}
        self.goal_on = {}     # ring_name -> target_name
        self.towers = set()   # set of tower object names
        self.ring_goal_order = [] # List of rings ordered from bottom to top based on goals

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Preprocess fact IDs and identify objects
        all_on_sources = set()
        all_objs = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            all_objs.update(args)
            if pred == "on":
                r, o = args[0], args[1]
                all_on_sources.add(r)
                if r not in self.on_ids:
                    self.on_ids[r] = {}
                self.on_ids[r][o] = f.global_id

        # Parse goals from model
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # Distinguish rings from towers: Towers are objects that never appear as 'on' sources
        self.towers = all_objs - all_on_sources

        # Precompute goal dependencies (bottom-up stack ordering)
        # Map target -> source to climb the stack
        current_map = {target: r for r, target in self.goal_on.items()}
        bottom_rings = [r for r, target in self.goal_on.items() if target in self.towers]
        
        processed = set()
        for br in bottom_rings:
            curr = br
            while curr and curr not in processed:
                self.ring_goal_order.append(curr)
                processed.add(curr)
                curr = current_map.get(curr)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state_bitwise, task_network):
        # 1. State term: Count unstable rings
        h_unstable = 0
        stable_objects = set(self.towers)
        
        # Iterate bottom-up: if target is stable and current is on target, current is stable
        for r in self.ring_goal_order:
            target = self.goal_on.get(r)
            
            # Bitwise search for current target of ring r
            current_target = None
            if r in self.on_ids:
                for t_name, fid in self.on_ids[r].items():
                    if (state_bitwise >> fid) & 1:
                        current_target = t_name
                        break
            
            if current_target == target and target in stable_objects:
                stable_objects.add(r)
            else:
                h_unstable += 1
        
        # 2. Task Network term: Count remaining abstract and primitive tasks
        h_tasks = len(task_network)
        
        # Combine terms. We add them to create a search gradient, 
        # though max() would be strictly admissible if goal fixing and TN clearing overlap.
        return h_unstable + h_tasks

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:] # Remove +/-
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
