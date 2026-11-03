from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

This domain is a "Lights Out" style puzzle where flipping one lamp can cause a chain reaction, flipping adjacent lamps.

**Bottleneck Analysis:**
The primary bottleneck is the complex, cascading effect of a single `flip` action. The cost of a `turn` is not constant; it depends on how far the propagation effect travels. This makes it difficult to estimate the true cost of any high-level choice.

**Admissibility Strategy (Max of 1 Term):**
This heuristic uses a single, admissible lower bound on the remaining cost.
1.  **Mismatched Lamps Count (`h_mismatch`):** The heuristic value is the number of lamps whose current state (Y/N) does not match their required goal state.

    - **Admissibility Sketch:** The only primitive action that can change a lamp's state is `flip`, which has a cost of 1. To change the state of a mismatched lamp, it must be the target of a `flip` action at some point, either directly via a `turn` or indirectly via propagation. Each `flip` action can, at best, fix the state of the lamp it is applied to. Therefore, the number of currently mismatched lamps is a lower bound on the minimum number of `flip` actions required to reach the goal.
"""

class LampsGemini25ProHeuristic6(Heuristic):
    """
    Heuristic for the Lamps domain.

    Calculates the number of lamps that are not in their goal state.
    This is an admissible lower bound on the number of 'flip' actions required.
    """

    def __init__(self, lamps_gemini25pro_heuristic_6=None, name="LampsGemini25ProHeuristic6"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Precomputed data from initialize()
        self.coords = []
        self.goal_on_status = {}
        self.on_fact_ids = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search starts. Preprocesses the domain.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into (predicate, [args]).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Extracts coordinates, goal states, and fact IDs from the model.
        This is done once to make the __call__ method fast.
        """
        all_coords = set()

        # 1. Parse goals to find goal configuration and relevant coordinates
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.goal_on_status[(x, y)] = status
                all_coords.add((x, y))

        # 2. Parse all facts to map (coord, status) to fact IDs
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord_tuple = (x, y)
                all_coords.add(coord_tuple)
                if coord_tuple not in self.on_fact_ids:
                    self.on_fact_ids[coord_tuple] = {}
                self.on_fact_ids[coord_tuple][status] = fact.global_id
        
        self.coords = sorted(list(all_coords))

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Computes the heuristic value for a given state.
        Counts the number of lamps not matching their goal status.
        """
        # If the task network is empty but goals are not satisfied, it's a dead end.
        if not task_network and not ((self.model.goals & state_bitwise) == self.model.goals):
            return float('inf')

        mismatched_lamps = 0
        for coord in self.coords:
            goal_status = self.goal_on_status.get(coord)
            if goal_status is None:
                # This lamp is not part of the goal definition, so we don't count it.
                continue

            # Determine the current status of the lamp at 'coord'
            current_status = None
            fact_id_map = self.on_fact_ids.get(coord)
            
            if fact_id_map:
                y_fact_id = fact_id_map.get('Y')
                n_fact_id = fact_id_map.get('N')

                if y_fact_id is not None and (state_bitwise >> y_fact_id) & 1:
                    current_status = 'Y'
                elif n_fact_id is not None and (state_bitwise >> n_fact_id) & 1:
                    current_status = 'N'
            
            if current_status != goal_status:
                mismatched_lamps += 1
                
        return float(mismatched_lamps)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate heuristic value for a node during search.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
