from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic16(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_11):
        super().__init__(barman_gemini3pro_heuristic_11)
        self.model = barman_gemini3pro_heuristic_11
        
        # Lookup maps: predicate -> { args_tuple : global_id }
        self.fact_ids = {}
        self.shakers = set()
        
        # 1. Parse static model facts to build indices and identify objects
        for fact in self.model.facts:
            name = fact.name.strip()
            if name.startswith('+'):
                name = name[1:]
            elif name.startswith('-'):
                continue
                
            # Expected format: predicate[arg1,arg2]
            idx = name.find('[')
            if idx != -1 and name.endswith(']'):
                pred = name[:idx]
                args_str = name[idx+1:-1]
                args = tuple(args_str.split(',')) if args_str else ()
                
                if pred not in self.fact_ids:
                    self.fact_ids[pred] = {}
                self.fact_ids[pred][args] = fact.global_id
                
                # Identify shakers by the 'shakerLevel' predicate which is unique to shakers
                if pred == 'shakerLevel':
                    self.shakers.add(args[0])

        # 2. Pre-compute goal tasks from the model.goals bitmask
        # This handles cases where the task network is empty (e.g. final state check)
        self.goal_tasks = []
        if hasattr(self.model, 'goals') and self.model.goals:
            for pred, arg_map in self.fact_ids.items():
                for args, fid in arg_map.items():
                    if (self.model.goals >> fid) & 1:
                        # Map low-level goal facts to high-level task estimates
                        if pred == 'contains':
                            if args[0] in self.shakers:
                                self.goal_tasks.append(('AchieveContainsShakerCocktail', args))
                            else:
                                self.goal_tasks.append(('AchieveContainsShotCocktail', args))
                        elif pred == 'clean':
                            if args[0] in self.shakers:
                                self.goal_tasks.append(('AchieveCleanShaker', args))
                            else:
                                self.goal_tasks.append(('AchieveCleanShot', args))
                        elif pred == 'handEmpty':
                            self.goal_tasks.append(('AchieveHandEmpty', args))
                        elif pred == 'ontable':
                            self.goal_tasks.append(('AchieveOnTable', args))
                        elif pred == 'holding':
                            self.goal_tasks.append(('AchieveHolding', args))

    def _get_fid(self, pred, args):
        return self.fact_ids.get(pred, {}).get(args)

    def _is_true(self, state, pred, args):
        fid = self._get_fid(pred, args)
        if fid is not None:
            return (state >> fid) & 1
        return False

    def compute(self, state, task_network):
        h = 0.0
        
        # Determine which tasks to evaluate cost for.
        # If the HTN task network is not empty, we estimate the cost to complete it.
        # If it is empty, we fall back to checking unsatisfied goal conditions.
        tasks_to_eval = []
        
        if task_network:
            for t in task_network:
                # If it's a primitive Operator, add its cost (usually 1)
                if hasattr(t, 'cost'):
                    h += getattr(t, 'cost', 1)
                    continue
                
                # If it's an AbstractTask, parse its name to get type and arguments
                # Name format assumption: "TaskName arg1 arg2 ..."
                if hasattr(t, 'name'):
                    parts = t.name.split()
                    if parts:
                        tasks_to_eval.append((parts[0], tuple(parts[1:])))
        else:
            tasks_to_eval = self.goal_tasks

        # Evaluate cost for each pending task
        for t_name, args in tasks_to_eval:
            
            if t_name == 'AchieveContainsShotCocktail':
                # Args: (?shot, ?cocktail)
                shot, cocktail = args
                
                # If already achieved, cost is 0
                if not self._is_true(state, 'contains', (shot, cocktail)):
                    # Minimum 1 step to pour
                    h += 1 
                    
                    # Check if cocktail is already prepared in a shaker
                    cocktail_found_in_shaker = False
                    cocktail_ready_to_pour = False
                    
                    for sh in self.shakers:
                        if self._is_true(state, 'contains', (sh, cocktail)):
                            cocktail_found_in_shaker = True
                            if self._is_true(state, 'shaked', (sh,)):
                                cocktail_ready_to_pour = True
                            break
                    
                    if cocktail_found_in_shaker:
                        if not cocktail_ready_to_pour:
                            h += 1 # Needs shaking
                    else:
                        # Not in any shaker: Need to add 2 ingredients + shake
                        # Cost: 2 * (Fill Shot + Pour to Shaker) + Shake
                        # Fill(1) + Pour(1) = 2 per ingredient.
                        h += 5 
                    
                    # Check if the destination shot is clean
                    if not self._is_true(state, 'clean', (shot,)) and not self._is_true(state, 'empty', (shot,)):
                        h += 2 # Empty + Clean
                        
            elif t_name == 'AchieveContainsShakerCocktail':
                # Args: (?shaker, ?cocktail)
                shaker, cocktail = args
                if not self._is_true(state, 'contains', (shaker, cocktail)):
                    # Assuming empty/clean shaker needed or adding to used one
                    # Cost to add ingredients and shake
                    h += 5 
                    
                    # If shaker is dirty and not containing parts, might need cleaning
                    if not self._is_true(state, 'clean', (shaker,)) and not self._is_true(state, 'empty', (shaker,)):
                        # Simple check: if not empty and not clean, cleaning cost
                        h += 2
                elif not self._is_true(state, 'shaked', (shaker,)):
                    h += 1 # Just shake

            elif t_name == 'AchieveContainsShakerIngredient':
                # Args: (?shaker, ?ingredient)
                shaker, ing = args
                if not self._is_true(state, 'contains', (shaker, ing)):
                    # Fill shot (1) + Pour to shaker (1)
                    h += 2

            elif t_name == 'AchieveContainsShotIngredient':
                # Args: (?shot, ?ingredient)
                shot, ing = args
                if not self._is_true(state, 'contains', (shot, ing)):
                    h += 1 # Fill shot

            elif t_name == 'AchieveCleanShot':
                shot = args[0]
                if not self._is_true(state, 'clean', (shot,)):
                    # If not empty, empty it first
                    if not self._is_true(state, 'empty', (shot,)):
                        h += 2 # Empty + Clean
                    else:
                        h += 1 # Just Clean
                        
            elif t_name == 'AchieveCleanShaker':
                shaker = args[0]
                if not self._is_true(state, 'clean', (shaker,)):
                    if not self._is_true(state, 'empty', (shaker,)):
                        h += 2
                    else:
                        h += 1

            elif t_name == 'AchieveHandEmpty':
                hand = args[0]
                if not self._is_true(state, 'handEmpty', (hand,)):
                    h += 1 # Drop or put down

            elif t_name == 'AchieveHolding':
                hand, obj = args
                if not self._is_true(state, 'holding', (hand, obj)):
                    h += 1 # Grasp (assuming hand empty or +1 for empty)
            
            elif t_name == 'DoPourShakerToShot':
                h += 1

        return h
