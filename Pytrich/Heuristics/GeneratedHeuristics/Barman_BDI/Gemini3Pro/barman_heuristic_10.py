from Pytrich.core.heuristic import Heuristic
import sys

class BarmanGemini3ProHeuristic10(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_5=None):
        super().__init__(model)
        self.fact_name_to_id = {f.name: f.global_id for f in self.model.facts}
        self.task_info = {}
        self._precompute_task_info()

    def _get_fact_id(self, predicate, args):
        """Constructs the grounded fact name and retrieves its ID."""
        # Format: +predicate[arg1,arg2]
        name = f"+{predicate}[{','.join(args)}]"
        return self.fact_name_to_id.get(name, -1)

    def _parse_task_name(self, name):
        """Parses the task string into task name and arguments."""
        parts = name.replace("(", "").replace(")", "").split()
        if not parts:
            return "", []
        return parts[0], parts[1:]

    def _precompute_task_info(self):
        """
        Precomputes base costs and discount conditions for all abstract tasks in the model.
        This avoids string parsing and dictionary lookups during the heuristic evaluation.
        """
        # Base costs for high-level tasks (estimated number of primitive actions)
        # Tuning these to be admissible but informative lower bounds.
        COST_MAP = {
            "AchieveContainsShotCocktail": 15,
            "AchieveContainsShakerCocktail": 12,
            "AchieveContainsShakerIngredient": 6,
            "AchieveContainsShotIngredient": 6,
            "AchieveCleanShaker": 4,
            "AchieveCleanShot": 4,
            "AchieveHolding": 2,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

        for task in self.model.abstract_tasks:
            t_name, args = self._parse_task_name(task.name)
            
            base_cost = COST_MAP.get(t_name, 1)
            discounts = []

            # Logic to determine when a task is effectively 'done' or 'cheaper'
            if t_name == "AchieveContainsShotCocktail" and len(args) >= 2:
                # Args: ?shot ?cocktail
                shot, cocktail = args[0], args[1]
                # If shot already contains cocktail, cost is 0
                fid = self._get_fact_id("contains", [shot, cocktail])
                discounts.append((fid, base_cost))
                
            elif t_name == "AchieveContainsShakerCocktail" and len(args) >= 2:
                # Args: ?shaker ?cocktail
                shaker, cocktail = args[0], args[1]
                fid = self._get_fact_id("contains", [shaker, cocktail])
                discounts.append((fid, base_cost))

            elif t_name == "AchieveContainsShakerIngredient" and len(args) >= 2:
                # Args: ?shaker ?ingredient
                shaker, ing = args[0], args[1]
                fid = self._get_fact_id("contains", [shaker, ing])
                discounts.append((fid, base_cost))

            elif t_name == "AchieveContainsShotIngredient" and len(args) >= 2:
                # Args: ?shot ?ingredient
                shot, ing = args[0], args[1]
                fid = self._get_fact_id("contains", [shot, ing])
                discounts.append((fid, base_cost))

            elif t_name == "AchieveCleanShot" and len(args) >= 1:
                # Args: ?shot
                shot = args[0]
                fid = self._get_fact_id("clean", [shot])
                discounts.append((fid, base_cost))
                # If not clean but empty, it's cheaper than full
                fid_empty = self._get_fact_id("empty", [shot])
                # If empty, we save the emptying cost (~1 action)
                discounts.append((fid_empty, 1))

            elif t_name == "AchieveCleanShaker" and len(args) >= 1:
                # Args: ?shaker
                shaker = args[0]
                fid = self._get_fact_id("clean", [shaker])
                discounts.append((fid, base_cost))
                fid_empty = self._get_fact_id("empty", [shaker])
                discounts.append((fid_empty, 1))

            elif t_name == "AchieveHolding" and len(args) >= 2:
                # Args: ?hand ?obj
                hand, obj = args[0], args[1]
                fid = self._get_fact_id("holding", [hand, obj])
                discounts.append((fid, base_cost))

            elif t_name == "AchieveHandEmpty" and len(args) >= 1:
                # Args: ?hand
                hand = args[0]
                fid = self._get_fact_id("handEmpty", [hand])
                discounts.append((fid, base_cost))
                
            elif t_name == "AchieveOnTable" and len(args) >= 1:
                obj = args[0]
                fid = self._get_fact_id("ontable", [obj])
                discounts.append((fid, base_cost))

            # Store the computed info
            self.task_info[task.global_id] = (base_cost, discounts)

    def h(self, node):
        """
        Calculates the heuristic value for the given node.
        Sum of estimated costs of tasks in the task network.
        """
        h_val = 0
        state = node.state
        
        # Iterate through the task network
        # node.task_network contains a list of Operator or AbstractTask objects
        for task in node.task_network:
            # Check if it is an AbstractTask (has decompositions)
            if hasattr(task, 'decompositions'):
                info = self.task_info.get(task.global_id)
                if info:
                    base_cost, discounts = info
                    current_cost = base_cost
                    
                    # Apply discounts based on current state
                    for fid, discount_amount in discounts:
                        if fid != -1 and ((state >> fid) & 1):
                            current_cost -= discount_amount
                    
                    # Ensure cost doesn't go below 0
                    if current_cost < 0:
                        current_cost = 0
                    
                    h_val += current_cost
                else:
                    # Fallback for unknown tasks
                    h_val += 1
            else:
                # It is an Operator
                # Use the operator's cost (default 1)
                h_val += getattr(task, 'cost', 1)
                
        return h_val