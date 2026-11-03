import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for snake domain to be used as an example when prompting LLMs to generate heuristics
class SnakeExampleDomainHeuristic(Heuristic):
    def __init__(self, distance_penalty=2, movement_penalty=1, name="snake_example_heuristic"):
        super().__init__(name=name)
        self.distance_penalty = distance_penalty
        self.movement_penalty = movement_penalty
        self.method_scores = {}
        self.preprocessing_time = 0
        self.grid_size = 10  # Assuming 10x10 grid based on problem file

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)
        
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        print("DEBUG: Starting _precompute_method_info")
        
        if not self.model:
            print("DEBUG: No model found")
            return
            
        # Try different possible attributes for methods
        methods = []
        
        print("DEBUG: Checking for methods attribute")
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
            print(f"DEBUG: Found {len(methods)} methods via .methods")
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
            print(f"DEBUG: Found {len(methods)} methods via .decompositions")
        elif hasattr(self.model, 'tasks'):
            print(f"DEBUG: Found {len(self.model.tasks)} tasks")
            # Get methods from abstract tasks
            for i, task in enumerate(self.model.tasks):
                print(f"DEBUG: Processing task {i}")
                if hasattr(task, 'decompositions'):
                    task_methods = task.decompositions
                    print(f"DEBUG: Task {i} has {len(task_methods)} decompositions")
                    methods.extend(task_methods)
                if i > 100:  # Safety break
                    print("DEBUG: Breaking after 100 tasks to prevent infinite loop")
                    break
        
        print(f"DEBUG: Total methods found: {len(methods)}")
        
        if not methods:
            print("DEBUG: No methods found, returning")
            return
        
        print("DEBUG: Processing methods...")    
        for i, method in enumerate(methods):
            if i > 50:  # Safety break
                print("DEBUG: Breaking after 50 methods to prevent infinite loop")
                break
                
            method_name = method.name if hasattr(method, 'name') else str(method)
            print(f"DEBUG: Processing method {i}: {method_name}")
            
            # Count different types of subtasks
            movement_count = 0
            strike_count = 0
            subtask_count = 0
            
            # Try different possible attributes for subtasks
            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition
            
            if subtasks:
                subtask_count = len(subtasks)
                print(f"DEBUG: Method {method_name} has {subtask_count} subtasks")
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)
                    
                    subtask_lower = subtask_name.lower()
                    
                    if any(move_term in subtask_lower for move_term in ['move-long', 'move-short', 'move ']):
                        movement_count += 1
                    if 'strike' in subtask_lower:
                        strike_count += 1
            
            self.method_scores[method_name] = {
                'movement_count': movement_count,
                'strike_count': strike_count,
                'subtask_count': subtask_count,
                'is_hunt_method': self._is_hunt_method(method_name),
                'is_move_method': self._is_move_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': movement_count + strike_count
            }
        
        print(f"DEBUG: Finished processing. Created {len(self.method_scores)} method scores")

    def _is_hunt_method(self, method_name):
        """Check if method is for hunting (main goal)."""
        return 'hunt' in method_name.lower()

    def _is_move_method(self, method_name):
        """Check if method is for movement."""
        return 'move' in method_name.lower()

    def _is_base_case_method(self, method_name):
        """Check if method is a base case (no further decomposition needed)."""
        base_indicators = ['hunt_done', 'move-base']
        return any(indicator in method_name.lower() for indicator in base_indicators)

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly achieves goals (striking mice)."""
        return 'hunt_all' in method_name.lower() or 'strike' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        Return very small values to ensure admissibility.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0
        
        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on various factors.
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0  # Start with 0 cost
        
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()
        
        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            
            # Very small penalties for complexity
            score += info['movement_count'] * 0.1
            score += info['subtask_count'] * 0.05
            
            # Small bonuses (negative cost) for productive methods
            if info['is_goal_achieving']:
                score -= 0.2  # Small bonus for hunting mice
            
            if info['is_base_case']:
                score -= 0.1  # Small bonus for termination
        
        # Method-specific scoring based on name patterns
        if 'hunt_done' in method_lower:
            score = 0  # No cost for completion
        elif 'strike' in method_lower:
            score += 0.1  # Very small cost for striking (productive action)
        elif 'hunt_all' in method_lower:
            score += 0.2  # Small cost for hunting (involves movement)
        elif 'move-base' in method_lower:
            score = 0  # No cost for base case (already at destination)
        elif 'move-long' in method_lower or 'move-short' in method_lower:
            score += 0.3  # Small cost for movement
        
        return max(score, 0)

    def _estimate_remaining_mice(self, node):
        """
        Try to estimate remaining mice, but since we can't inspect state,
        return a conservative estimate.
        """
        # Since we can't inspect the state, assume moderate progress
        return 3  # Conservative estimate of remaining mice

    def _manhattan_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions.
        Positions are assumed to be in format like 'px3y2'.
        """
        if not isinstance(pos1, str) or not isinstance(pos2, str):
            return 1  # Default distance
        
        try:
            # Extract coordinates from position strings like 'px3y2'
            x1 = int(pos1.split('x')[1].split('y')[0])
            y1 = int(pos1.split('y')[1])
            x2 = int(pos2.split('x')[1].split('y')[0])
            y2 = int(pos2.split('y')[1])
            
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            return 1  # Default if parsing fails

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        Keep values very small for admissibility.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure very small values
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value
    
    def __repr__(self):
        return f"SnakeHeuristic(dist_penalty={self.distance_penalty}, move_penalty={self.movement_penalty})"
    
    def __str__(self):
        return f"SnakeHeuristic(dist_penalty={self.distance_penalty}, move_penalty={self.movement_penalty})"

    def __output__(self):
        return (
            f"Snake Heuristic info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tDistance penalty: {self.distance_penalty}\n"
            f"\tMovement penalty: {self.movement_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tGrid size: {self.grid_size}x{self.grid_size}\n"
        )