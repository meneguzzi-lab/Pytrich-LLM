from typing import List, Union, Set, Dict, Tuple
import sys

class BlocksworldGemini3ProHeuristic11(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_8, model):
        super().__init__(blocksworld_gemini3pro_heuristic_8, model)
        self.model = model
        
        # Precompute mappings for efficient bitwise state checks
        # Reverse mapping: bottom_block -> list of (top_block, global_id_of_on_fact)
        self.on_map_rev: Dict[str, List[Tuple[str, int]]] = {}
        # Direct mapping: (top, bottom) -> global_id
        self.on_map_dir: Dict[Tuple[str, str], int] = {}
        # Status mappings: block -> global_id
        self.ontable_map: Dict[str, int] = {}
        self.clear_map: Dict[str, int] = {}
        self.holding_map: Dict[str, int] = {}
        self.handempty_id: int = -1
        
        # Parse all grounded facts in the model to populate mappings
        for fact in self.model.facts:
            name = fact.name.strip()
            # Handle +/- prefix for positive/negative facts
            if name.startswith("+"):
                name = name[1:]
            elif name.startswith("-"):
                continue
                
            # Parse format: predicate[arg1,arg2,...]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                
                if predicate == "on" and len(args) == 2:
                    top, bottom = args[0], args[1]
                    if bottom not in self.on_map_rev:
                        self.on_map_rev[bottom] = []
                    self.on_map_rev[bottom].append((top, fact.global_id))
                    self.on_map_dir[(top, bottom)] = fact.global_id
                    
                elif predicate == "ontable" and len(args) == 1:
                    self.ontable_map[args[0]] = fact.global_id
                    
                elif predicate == "clear" and len(args) == 1:
                    self.clear_map[args[0]] = fact.global_id
                    
                elif predicate == "holding" and len(args) == 1:
                    self.holding_map[args[0]] = fact.global_id
                    
                elif predicate == "handempty":
                    self.handempty_id = fact.global_id

    def _get_on_block(self, block: str, state: int) -> Union[str, None]:
        """Returns the block directly on top of the given block, or None."""
        # Optimization: Check clear first. If clear, nothing is on top.
        if block in self.clear_map:
            if (state >> self.clear_map[block]) & 1:
                return None
        
        # Check all possible blocks that could be on this block
        if block in self.on_map_rev:
            for top_block, gid in self.on_map_rev[block]:
                if (state >> gid) & 1:
                    return top_block
        return None

    def _get_clear_cost(self, block: str, state: int, memo: Dict[str, int]) -> int:
        """Recursively calculates the cost to clear a block."""
        if block in memo:
            return memo[block]
            
        # Base case 1: Block is already clear
        if block in self.clear_map and ((state >> self.clear_map[block]) & 1):
            memo[block] = 0
            return 0
            
        # Base case 2: Block is being held (conceptually clear/available to be placed)
        if block in self.holding_map and ((state >> self.holding_map[block]) & 1):
            memo[block] = 0
            return 0
            
        # Recursive step: Find block on top and cost to remove it
        top = self._get_on_block(block, state)
        if top is None:
            # Inconsistent state or static logic implies clear but predicate not set.
            # Assume 0 to be admissible.
            memo[block] = 0
            return 0
            
        # Cost = Cost to clear the top block + 2 actions (unstack top, putdown top)
        # We assume 'clearing' involves moving the obstructing block to the table.
        cost = self._get_clear_cost(top, state, memo) + 2
        memo[block] = cost
        return cost

    def _get_ontable_cost(self, block: str, state: int, memo: Dict[str, int]) -> int:
        """Calculates cost to put a block on the table."""
        # Case 1: Already on table
        if block in self.ontable_map and ((state >> self.ontable_map[block]) & 1):
            return 0
            
        # Case 2: Holding the block
        if block in self.holding_map and ((state >> self.holding_map[block]) & 1):
            return 1 # Just put-down
            
        # Case 3: Block is on something else
        # Cost = Cost to clear it + 2 (unstack it, put it down)
        return self._get_clear_cost(block, state, memo) + 2

    def h(self, node):
        """
        Domain-dependent heuristic for Blocksworld.
        Estimates cost by analyzing the decomposition of the task network.
        Considers the 'Move to Table' strategy enforced by the domain methods.
        """
        state = node.state
        tn = node.task_network
        
        if not tn:
            return 0
            
        total_cost = 0
        memo_clear = {}
        
        for task in tn:
            # Handle Primitive Tasks (Operators)
            if hasattr(task, 'add_effects'): 
                # 'nop' has 0 cost
                if task.name.startswith("nop"):
                    continue
                # Primitive actions have unit cost
                total_cost += 1
                continue
            
            # Handle Abstract Tasks
            # Expecting grounded name like "do_put_on b1 b2"
            parts = task.name.split()
            if not parts:
                total_cost += 1
                continue
            
            task_type = parts[0]
            args = parts[1:]
            
            if task_type == "do_put_on" and len(args) == 2:
                x, y = args[0], args[1]
                
                # Check if goal already satisfied (Method m0)
                if (x, y) in self.on_map_dir and ((state >> self.on_map_dir[(x, y)]) & 1):
                    continue
                    
                # Method m1 enforces: clear x, clear y, on_table y, move x y
                # Heuristic Estimate:
                # 1. Cost to clear X
                # 2. Cost to put Y on table (implies clearing Y first)
                # 3. Cost to move X to Y (Pick/Unstack + Stack)
                
                cost_clear_x = self._get_clear_cost(x, state, memo_clear)
                cost_ontable_y = self._get_ontable_cost(y, state, memo_clear)
                
                # Move cost: 2 (Unstack+Stack or Pickup+Stack) usually
                # If already holding X, only 1 (Stack)
                move_cost = 2
                if x in self.holding_map and ((state >> self.holding_map[x]) & 1):
                    move_cost = 1
                    
                total_cost += cost_clear_x + cost_ontable_y + move_cost
                
            elif task_type == "do_on_table" and len(args) == 1:
                total_cost += self._get_ontable_cost(args[0], state, memo_clear)
                
            elif task_type == "do_clear" and len(args) == 1:
                total_cost += self._get_clear_cost(args[0], state, memo_clear)
                
            elif task_type == "do_move" and len(args) >= 2:
                # Primitive move sequence cost
                x = args[0]
                if x in self.holding_map and ((state >> self.holding_map[x]) & 1):
                    total_cost += 1
                else:
                    total_cost += 2
            else:
                # Unknown abstract task, assume minimal cost
                total_cost += 1
                
        return total_cost
