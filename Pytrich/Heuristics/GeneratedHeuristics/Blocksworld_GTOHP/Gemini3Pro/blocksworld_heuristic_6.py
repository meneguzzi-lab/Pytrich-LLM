import sys
from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic6(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_3, model):
        super().__init__(blocksworld_gemini3pro_heuristic_3, model)
        self.fact_map = {}
        self.goal_map = {}  # block_name -> parent_name (or 'TABLE')
        self.goal_targets_rev = {}  # parent_name -> block_name (the block that wants to be on parent)

        # --- Pre-process facts for efficient state decoding ---
        # We map the global_id of each fact to a tuple (type, arg1, arg2)
        # This allows O(1) decoding during the heuristic evaluation.
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on' and len(args) == 2:
                # (type, child, parent)
                self.fact_map[fact.global_id] = ('on', args[0], args[1])
            elif pred == 'ontable' and len(args) == 1:
                # (type, child)
                self.fact_map[fact.global_id] = ('ontable', args[0])
            elif pred == 'holding' and len(args) == 1:
                # (type, block)
                self.fact_map[fact.global_id] = ('holding', args[0])

        # --- Pre-process goals ---
        # The goals are stored as a bitmask in self.model.goals.
        # We extract them once to build a fast lookup map.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on' and len(args) == 2:
                child, parent = args[0], args[1]
                self.goal_map[child] = parent
            elif pred == 'ontable' and len(args) == 1:
                child = args[0]
                self.goal_map[child] = 'TABLE'

        # Build reverse map: Parent -> Child (who wants to be on top)
        # This is used to detect 'blocking' blocks.
        for child, parent in self.goal_map.items():
            if parent != 'TABLE':
                self.goal_targets_rev[parent] = child

    def _parse_fact_name(self, name: str):
        """Parses grounded fact strings like +on[b1,b2] into predicate and args."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def h(self, node):
        """
        Calculates the heuristic value for a given node.
        Strategy: Count 'misplaced' blocks.
        A block is considered misplaced (needs move) if:
        1. It is currently in a position different from its goal position.
        2. The block underneath it needs to move (unstable support).
        3. It is blocking a position required by another block (it is on P, but Z wants to be on P).
        """
        
        # 1. Decode the current state bitmask into a map
        current_parent = {}  # block -> parent (or 'TABLE' or 'HAND')
        
        state = node.state
        # Iterate over set bits efficiently
        while state:
            # Get lowest set bit
            t = state & -state
            # Calculate index (global_id)
            idx = t.bit_length() - 1
            # Clear bit
            state ^= t
            
            info = self.fact_map.get(idx)
            if info:
                itype = info[0]
                if itype == 'on':
                    current_parent[info[1]] = info[2]
                elif itype == 'ontable':
                    current_parent[info[1]] = 'TABLE'
                elif itype == 'holding':
                    current_parent[info[1]] = 'HAND'

        # 2. Build support structure for iterative propagation
        # supports[p] = list of blocks currently on p
        supports = {}
        blocks_on_table_or_hand = []
        
        all_blocks = list(current_parent.keys())
        
        for block in all_blocks:
            parent = current_parent[block]
            if parent == 'TABLE' or parent == 'HAND':
                blocks_on_table_or_hand.append(block)
            else:
                if parent not in supports:
                    supports[parent] = []
                supports[parent].append(block)

        # 3. Iterative evaluation of 'needs_move'
        # We process from bottom (Table/Hand) up to Top.
        # This ensures when we process block B, we know if its parent needs to move.
        
        needs_move = {} # block -> bool
        
        # Queue for topological processing (Bottom-Up)
        queue = blocks_on_table_or_hand
        
        # We iterate through the queue. If we encounter a cycle (unreachable from table/hand),
        # those blocks are implicitly misplaced, but standard BW states are trees.
        # Blocks not reached by this BFS are part of 'floating' cycles (unlikely in valid states) or just missed.
        # We will handle unvisited blocks at the end.
        
        idx = 0
        while idx < len(queue):
            block = queue[idx]
            idx += 1
            
            parent = current_parent[block]
            
            # Determine if 'block' needs to move
            must_move = False
            
            if parent == 'HAND':
                # Block in hand implies it's not in goal position (goals are on table or block)
                must_move = True
            else:
                # Parent is TABLE or another Block
                
                # Factor A: Parent itself needs to move?
                if parent == 'TABLE':
                    parent_moving = False
                else:
                    # Since we process bottom-up, parent should be in needs_move
                    parent_moving = needs_move.get(parent, True)
                
                if parent_moving:
                    must_move = True
                else:
                    # Factor B: Does block match its specific goal?
                    goal_parent = self.goal_map.get(block)
                    if goal_parent is not None:
                        if parent != goal_parent:
                            must_move = True
                    
                    # Factor C: Is block blocking someone else?
                    # Only relevant if block is currently stable otherwise (parent not moving, block matches goal or no goal)
                    if not must_move and parent != 'TABLE':
                        # If parent is the goal target for some other block Z
                        target_z = self.goal_targets_rev.get(parent)
                        if target_z and target_z != block:
                            must_move = True
            
            needs_move[block] = must_move
            
            # Add children to queue
            if block in supports:
                queue.extend(supports[block])

        # 4. Count misplaced blocks
        count = 0
        for block in all_blocks:
            if block not in needs_move:
                # Block was not reachable from Table/Hand -> Cycle -> Unstable
                count += 1
            elif needs_move[block]:
                count += 1
        
        return count
