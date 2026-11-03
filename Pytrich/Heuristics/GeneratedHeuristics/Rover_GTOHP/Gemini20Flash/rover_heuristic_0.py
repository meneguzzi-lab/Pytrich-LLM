import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic0(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_0=1):
        super().__init__(name='RoverGemini20FlashHeuristic0')
        self.preprocessed = False
        self.waypoint_connections = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._preprocess()  # Moved preprocessing to a separate method
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _preprocess(self):
        """Precomputes information about the domain.
        Currently only calculates waypoint connections"""

        if self.preprocessed:
            return

        self.waypoint_connections = {}
        for op in self.model.operators:
            if op.name == 'navigate':
                p1 = op.name.split()[1] # changed indexing
                p2 = op.name.split()[2]

                if p1 not in self.waypoint_connections:
                    self.waypoint_connections[p1] = []
                self.waypoint_connections[p1].append(p2)

        self.preprocessed = True

    def _estimate_distance(self, start, end):
        """Estimates the distance between two waypoints.  This is imperfect since can_traverse isn't transitive and it doens't look at the plan, but better than blind search"""
        if start == end:
            return 0
        if start in self.waypoint_connections and end in self.waypoint_connections.get(start,[]): #Imperfect but helps 
            return 1 #Direct connection
        else:
            return 2 # No known direct connection, assign higher cost


    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        h_value = 0

        #Give a higher cost for non primitive tasks to encourage making plans        
        if isinstance(node.task, tuple):
           h_value += 1

        #If there is a current abstract task running, score it    
        current_tasks = node.current_tasks()
        for task in current_tasks:
          task_string = str(task)

          if   'get_soil_data' in task_string: h_value +=1
          elif 'send_soil_data' in task_string: h_value +=1
          elif 'get_rock_data' in task_string: h_value +=1
          elif 'send_rock_data' in task_string: h_value +=1
          elif 'get_image_data' in task_string: h_value +=1
          elif 'send_image_data' in task_string: h_value +=1
          elif 'do_navigate'   in task_string: h_value +=1
          elif 'empty_store'   in task_string: h_value +=1


        #Estimate the number of pending tasks as additional cost        
        pending_tasks = len(node.pending_tasks)
        h_value += pending_tasks *0.5


        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f"RoverGemini20FlashHeuristic0 with preprocessed waypoint connections: {len(self.waypoint_connections)} and preprocessing time {self.preprocessing_time:.4f}"