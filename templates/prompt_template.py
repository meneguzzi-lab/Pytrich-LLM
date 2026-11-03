"""
Shared prompt template for generating domain-dependent heuristics.

The prompt is split into named sections so an ablation runner can disable any
of them via the `components` dict passed to `generate_prompt()`. Default is
all-on, preserving original behavior for non-ablation callers.

Section keys (each maps to part of the original monolithic prompt):
    domain_hints           -> §2.5  (domain-specific hints from domains.yaml)
    critical_fact_format   -> §§3-5 (grounded fact format, goal state, bitwise ops)
    example_heuristic      -> §7    (working example heuristic)
    pytrich_api            -> §§8-9.1 (Fact / Operator / Task / Decomposition / Model)
    admissibility          -> §10   (admissibility vs guidance)
    win_lose_patterns      -> §11   (winning patterns + anti-patterns)
    design_procedure       -> §12   (the 5-step required design procedure)

Sections always included (cannot be ablated, output won't parse without them):
    §1  domain HDDL
    §2  training instances
    §6  required imports + base class signatures

"""

import textwrap


# States what h estimates. Matches astar_search.py: an applied operator adds
# 1 to g whatever its declared cost, a decomposition adds 0.
_SECTION_COST_MODEL = """
## Cost Model

Pytrich's A* charges one unit of path cost for every primitive action applied, including the no-op and
`__method_precondition_*` operators the grounder produces, and zero for applying a decomposition method. Your
heuristic estimates the remaining cost under this model.
"""

DEFAULT_COMPONENTS = {
    'domain_hints': True,
    'critical_fact_format': True,
    'example_heuristic': True,
    'pytrich_api': True,
    'admissibility': True,
    'win_lose_patterns': True,
    'design_procedure': True,
}


# Weak stubs used when ablating sections that would otherwise produce zero-coverage
# output if fully removed (mirrors the Correa et al. ablation pattern of replacing
# rather than removing for catastrophic cases).

# §7 stub: a blind heuristic that returns 0. Establishes shape of __call__ /
# initialize, no actual guidance. The LLM gets a syntactically valid template
# but no domain-specific reasoning to imitate.
_STUB_EXAMPLE_HEURISTIC = """
## 7. Working Example Heuristic (minimal stub)

The following is a structurally valid but uninformative example. It returns 0
for every node, providing no search guidance. Use it only as a syntactic
template for inheriting from `Heuristic` and matching the required signatures.

```python
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TrivialBlindHeuristic(Heuristic):
    def initialize(self, model: Model, initial_node: HTNNode):
        return super().initialize(model, 0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        super().update_info(0)
        return 0
```
"""


# §§8-9.1 stub: minimal API surface. Only the attributes and methods the
# heuristic actually needs; no class bodies, no bitwise examples.
_STUB_PYTRICH_API = """
## 8. Pytrich Model and Fact API (minimal reference)

Your heuristic interacts with the Pytrich model through the following surface.
This is a reference list, NOT full source — read carefully:

- `model.facts` — list of `Fact` objects in the grounded problem.
- `model.operators` — list of `Operator` objects.
- `model.goals` — bitwise integer; 1-bits at positions corresponding to goal-fact `global_id`s.
- `model.initial_state` — bitwise integer for the initial state.
- `model.state_explicit_repr(state_int)` — returns the list of fact names true in `state_int`.
- `Fact.name` — string, e.g. "+on[r1,r2]" or "+at[truck,depot]".
- `Fact.global_id` — integer used in state bitmasks.
- `Operator.name`, `Operator.pos_precons`, `Operator.neg_precons`, `Operator.add_effects`, `Operator.del_effects` — bitwise integers.
- `Operator.applicable(state)` — bool. `Operator.apply(state)` — successor state (bitwise).
- `node.state` — bitwise integer for the current state.
- `node.task_network` — list of `Operator` and `AbstractTask` references still to be resolved.

You must infer any further structure (e.g. `AbstractTask.decompositions`) from the
example heuristic and the imports.
"""


# §§3-5: CRITICAL grounded-fact / goal / bitwise sections (pure literal, no interpolation)
_SECTION_CRITICAL_FACT_FORMAT = r"""
## 3. CRITICAL: Grounded Fact Format

**IMPORTANT**: After grounding, facts do NOT have HDDL format. They have this format:
- `+predicate[arg1,arg2]` or `-predicate[arg1,arg2]`
- Examples: `+on[r1,r2]`, `+towerTop[r1,t1]`, `+smallerThan[r1,t2]`

To parse a grounded fact name:
```python
def _parse_fact_name(self, name: str):
    name = name.strip()
    # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
    if name.startswith("+") or name.startswith("-"):
        name = name[1:]  # Remove +/- prefix
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
    return None, []
```

## 4. CRITICAL: Accessing Goal State

**IMPORTANT**: Goals are stored in `model.goals` as a bitwise integer, NOT as separate predicates.

To get the goal facts:
```python
# model.goals is a bitwise integer representing goal state
goal_fact_names = self.model.state_explicit_repr(self.model.goals)
# Returns: ['+on[r1,r2]', '+on[r2,r3]', '+on[r3,t3]']
```

## 5. CRITICAL: Checking State with Bitwise Operations

Each fact has a `global_id`. To check if a fact is true in the current state:
```python
# CORRECT - Fast bitwise check
is_true = (state_bitwise >> fact.global_id) & 1

# WRONG - Too slow, will timeout on large problems
fact.name in self.model.state_explicit_repr(state)  # DO NOT USE
```
"""


# §6: Required imports + base class. ALWAYS INCLUDED — output won't parse without it.
_SECTION_REQUIRED_INTERFACE = r"""
## 6. CRITICAL: Required Imports and Base Class Interface

Your heuristic file MUST begin with these exact imports. Do NOT shorten the paths.

```python
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
```

**WRONG** (will crash at load time):
```python
from Heuristics.heuristic import Heuristic   # MISSING "Pytrich." prefix — DO NOT USE
```

### Method Signatures (MUST match exactly)

Your class inherits from `Heuristic`. You MUST implement these two methods with these exact signatures:

```python
def initialize(self, model: Model, initial_node: HTNNode):
    '''Called ONCE before search starts. Do ALL preprocessing here.'''
    # ... build fact maps, precompute distances, etc. ...
    h_value = self._compute(initial_node.state, initial_node.task_network)
    return super().initialize(model, h_value)

def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
    '''Called for EVERY node expansion. parent_node is the parent, node is the child.'''
    h_value = self._compute(node.state, node.task_network)
    super().update_info(h_value)
    return h_value
```

**WRONG** signatures (will crash at runtime):
```python
def initialize(self, model):                  # MISSING initial_node — crashes: "takes 2 args but 3 given"
def __call__(self, node):                     # MISSING parent_node — crashes: "takes 2 args but 3 given"
super().initialize(model)                     # MISSING h_value — crashes: "missing required arg 'initial_h'"
```

The base `Heuristic` class for reference:
```python
class Heuristic:
    def __init__(self, name="blind"):
        self.model = None
        self.name = name
        self.calls = 0
        self.total_hvalue = 0
        self.min_hvalue = 1000000
        self.initial_h = 0

    def initialize(self, model, initial_h):
        self.model = model
        self.initial_h = initial_h
        self.update_info(initial_h)
        return initial_h

    def update_info(self, h_value):
        self.calls += 1
        self.total_hvalue += h_value
        self.min_hvalue = min(self.min_hvalue, h_value)

    def __call__(self, parent_node, node):
        pass
```
"""


# §§8-9.1: Pytrich classes (pure literal, no interpolation)
_SECTION_PYTRICH_API = r"""
## 8. Pytrich Classes and Structures

### Fact class
```python
class Fact:
    def __init__(self, name, local_id, global_id):
        self.name = name
        self.hash_name = hash(name)
        self.global_id:int = global_id
        self.local_id:int  = local_id

    def __eq__(self, other):
        return (
            self.name == other.name
        )

    def __hash__(self):
        return  self.hash_name

    def __str__(self):
        return f"F(self.name (self.global_id))"

    def __repr__(self):
        return f"<Fself.global_id:self.name >"
```

### Operator class
```python
class Operator:
    def __init__(self, global_id, local_id, name, cost, pos_precons, neg_precons, add_effects, del_effects):
        self.name = name
        self.hash_name = hash(name)
        self.global_id:int = global_id
        self.local_id:int  = local_id

        self.cost:int = cost
        self.pos_precons:int = pos_precons
        self.neg_precons:int = neg_precons
        self.del_effects:int = del_effects
        self.add_effects:int = add_effects


    def applicable(self, state_bitwise):
        return ((state_bitwise & self.pos_precons) == self.pos_precons) and \
               ((state_bitwise & self.neg_precons) == 0)

    def apply(self, state_bitwise):
        return (state_bitwise & ~self.del_effects) | self.add_effects

    def relaxed_apply(self, state_bitwise):
        return state_bitwise | self.add_effects

    def get_add_effects(self):
        bitwise_value = self.add_effects
        i = 0
        while bitwise_value:
            if bitwise_value & 1:
                yield i
            bitwise_value >>= 1
            i += 1

    def get_precons(self):
        bitwise_value = self.pos_precons
        i = 0
        while bitwise_value:
            if bitwise_value & 1:
                yield i
            bitwise_value >>= 1
            i += 1

    def __eq__(self, other):
        return (
            self.name == other.name
        )

    def __hash__(self):
        return  self.hash_name

    def __str__(self):
        return f"OP(self.name bin(self.pos_precons) bin(self.neg_precons))"

    def __repr__(self):
        return f"<Op self.global_id:self.name >"
```

## 9. Python AbstractTask and Decomposition classes
```python
class AbstractTask:
    def __init__(self, global_id, local_id, decompositions, name):
        self.name = name
        self.hash_name = hash(name)
        self.decompositions: List[Decomposition] = decompositions
        self.global_id:int = global_id
        self.local_id:int  = local_id

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return f'GT(self.name arity len(self.decompositions))'
    def __repr__(self):
        return f'<Gt %s>' % self.name
    def __hash__(self):
        return hash(self.hash_name)

class Decomposition:
    def __init__(self, name, global_id, local_id, pos_precons, neg_precons, compound_task, task_network):
        self.name = name
        self.hash_name = hash(name)
        self.global_id:int = global_id
        self.local_id:int  = local_id

        self.compound_task:AbstractTask = compound_task
        self.task_network:List[Union[Operator, AbstractTask]] = task_network
        self.pos_precons:int = pos_precons
        self.neg_precons:int = neg_precons

    def applicable(self, state_bitwise):
        return ((state_bitwise & self.pos_precons) == self.pos_precons) and \
               ((state_bitwise & self.neg_precons) == 0)

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return self.hash_name

    def __repr__(self):
        return f"<D self.name self.task_network >"

    def __str__(self):
        s = f"DECOMPOSISITION self.name"
        s += f"Decomposed Task: self.compound_task"
        s += f"Task Network: self.task_network[0:min(5, len(self.task_network))]"
        return s

## 9.1 Python Model class
```python
class Model:
    def __init__(self, facts: set, initial_state: set, initial_tn: List[Union[Operator, AbstractTask]],
                 goals: set, operators: List[Operator], decompositions: List[Decomposition],
                 abstract_tasks: List[AbstractTask]):
        self.facts = facts
        self.initial_state = initial_state
        self.goals = goals
        self.initial_tn = initial_tn
        self.operators = operators
        self.decompositions = decompositions
        self.abstract_tasks = abstract_tasks

        self.desc = Descriptions()

        # Global ID info: initial (init) and final (end) global indixes for facts, operators, abstract_tasks, and decompositions
        self.ifacts_init = 0
        self.ifacts_end  = len(self.facts) - 1
        self.iop_init  = self.ifacts_end+1
        self.iop_end   = self.iop_init + len(self.operators)-1
        self.iabt_init = self.iop_end + 1
        self.iabt_end  = self.iabt_init + len(self.abstract_tasks)-1
        self.idec_init = self.iabt_end+1
        self.idec_end  = self.idec_init + len(self.decompositions)-1

        #self._remove_panda_top()

    def get_component(self, component_id):
        if component_id <= self.ifacts_end:
            fact=self.facts[component_id]
            assert fact.global_id == component_id, (
                f'FACT COMPONENT INDEXING FAILED: expected component_id, got fact.global_id'
            )
            return self.facts[component_id]

        if component_id >= self.iop_init and component_id <= self.iop_end:
            operator = self.operators[component_id - self.iop_init]
            assert operator.global_id == component_id, (
                f'OPERATOR COMPONENT INDEXING FAILED: expected component_id, got operator.global_id'
            )
            return operator

        if component_id >= self.iabt_init and component_id <= self.iabt_end:
            abstract_task = self.abstract_tasks[component_id - self.iabt_init]
            assert abstract_task.global_id == component_id, (
                f'ABSTRACT TASK COMPONENT INDEXING FAILED: expected component_id, got abstract_task.global_id'
            )
            return abstract_task

        if component_id >= self.idec_init and component_id <= self.idec_end:
            decomposition = self.decompositions[component_id - self.idec_init]
            assert decomposition.global_id == component_id, (
                f'DECOMPOSITION COMPONENT INDEXING FAILED: expected component_id, got decomposition.global_id'
            )
            return decomposition

        raise ValueError(f'Invalid component_id: component_id')

    def state_explicit_repr(self, state):
        return [self.facts[bit_pos].name for bit_pos in range(state.bit_length()) if state & 1<<bit_pos]

    def goal_reached(self, state, task_network=[]):
        return self.goals <= state and len(task_network) == 0

    def problem_info(self):
        model_info = (
            f"Model info:"
            f"self.desc('fact_model', len(self.facts))"
            f"self.desc('abstract_task_model', len(self.abstract_tasks))"
            f"self.desc('operator_model', len(self.operators))"
            f"self.desc('decomposition_model', len(self.decompositions))"
        )
        return model_info

    def __str__(self):
        memory_info = (
            f"Memory Usage:"
            f"Facts: sys.getsizeof(self.facts) bytes"
            f"Initial State: sys.getsizeof(self.initial_state) bytes"
            f"Goals: sys.getsizeof(self.goals) bytes"
            f"tOperators: sys.getsizeof(self.operators) bytes"
            f"Initial Task Network: sys.getsizeof(self.initial_tn) bytes"
            f"Decompositions: sys.getsizeof(self.decompositions) bytes"
        )
        return memory_info
```
"""


# §10: Admissibility (pure literal)
_SECTION_ADMISSIBILITY = r"""
## 10. Admissibility vs. Guidance (Important Distinction)

**Admissibility** (`h(n) <= h*(n)`) is required if you want A* to return optimal plans. For fast satisficing search
(our default), you can layer small **tie-breaking penalties** on top of an admissible base to break symmetries:

```python
h = admissible_max_of_terms + 0.001 * symmetry_penalty
```

Keep tie-breaking values sub-unit so they never dominate the admissible terms. Admissibility + tie-breaking are not
in conflict — they compose.
"""


# §11: Win/lose patterns (pure literal — note `{}` not `{{}}` since this is NOT an f-string)
_SECTION_WIN_LOSE = r"""
## 11. Patterns That Win vs. Patterns That Lose

These patterns are distilled from heuristics that beat the Panda baseline on Pytrich benchmarks and from those that
did not. Apply them deliberately.

### Winning Patterns (use these)

**P1 — Multi-component max heuristic.** Compute 2–3 independent admissible lower bounds and return `max(...)`.
Each term should capture a different kind of unavoidable cost (e.g., structural travel, decomposition count,
bottleneck resource). `max` is admissible iff each term is.
```python
def __call__(self, node):
    t1 = self._travel_lower_bound(node.state)
    t2 = self._decomposition_lower_bound(node.task_network)
    t3 = self._bottleneck_lower_bound(node.state)
    return max(t1, t2, t3)
```

**P2 — State-aware per-task cost estimation.** Do NOT return a value that depends only on the task network, and
do NOT assign a flat "+1" for every non-trivial task. Instead, use current-state bit tests to compute each task's
realistic minimum decomposition cost. The difference between a weak heuristic and a strong one is usually this:
the weak one says "this task costs 1" while the strong one says "this task costs 13 because 3 blocks must be
cleared first (7 actions) + 2 more cleared (5 actions) + 1 final move."
```python
# Per-task cost using current state (GOOD — realistic lower bound)
clear_x_bit = (state >> self.clear_fact_id[x]) & 1
cost = 0 if clear_x_bit else (1 + 2 * self._stack_height(x, state))

# Flat constant (BAD — wastes the heuristic's potential)
cost = 0 if clear_x_bit else 1  # DO NOT DO THIS
```

**P3 — Preprocess once, lookup fast.** Do all parsing, fact-map building, goal indexing, and distance-matrix work
inside `initialize(model)`. Keep `__call__` to O(1) bit tests per term — never iterate `self.model.facts` or
`self.model.operators` in `__call__`.
```python
def initialize(self, model):
    super().initialize(model)
    self.on_fact_id = {}  # predicate -> args -> global_id
    for f in model.facts:
        pred, args = self._parse_fact_name(f.name)
        if pred == "on":
            self.on_fact_id[(args[0], args[1])] = f.global_id
```

**P4 — Identify the domain bottleneck and cost it explicitly.** Most HTN domains have one serializing constraint
that dominates plan length (stack clearing in Blocksworld, single-carry in Robot, holding-shot in Barman, rover
navigation in Rover). Name the bottleneck in a top-of-file comment and add a dedicated `max` term for it.

**P5 — Goal-stability / false-trivial detection.** A goal fact currently true is NOT free if a supporting fact
will be undone on the way to another goal. Count these as non-trivial. In Blocksworld this single idea took
p10 from 2.37M expanded nodes to 5,813.

**P6 — Symmetry-breaking penalties for parameter-binding domains.** When multiple constants are interchangeable
(Barman shots, Robot containers, Rover waypoints that look alike), assign increasing ordered penalties to
canonicalize choices. Use **large** per-task penalties (~100s) — small penalties (<50) get washed out by the
dominant admissible terms. Add them as sub-unit tie-breakers only if you need to preserve optimality; otherwise
add them directly.

### Anti-Patterns (avoid these)

**A1 — Task-count heuristics with no state context.** "1 point per task type present in the network" is
unresponsive to progress and almost always loses to TDG. The fix is P2 (state-aware refinement).

**A2 — Pure forward simulation instead of summation.** Simulating task execution with state updates can produce
*lower* h-values than additive estimates because it avoids double-counting shared work. Counterintuitively, this
expands MORE nodes. Prefer disjoint-additive or max-composite bounds.

**A3 — Slow `__call__`.** Any loop over `self.model.facts`, `self.model.operators`, or nested recursion without
memoization in `__call__` will time out on large instances. Push that work into `initialize()` (see P3).

**A4 — Flat "+1 per non-nop task" costing.** Assigning cost=1 for every non-trivial task ignores the actual
work required. A task that needs 13 actions to decompose should contribute ~13 to the heuristic, not 1. Use
current-state facts to compute realistic per-task lower bounds (see P2). This is the single most common mistake
that makes a structurally correct heuristic (multi-component max, preprocessing, etc.) still perform poorly.
"""


# §12: Design procedure — needs domain_name interpolation, kept as a template
_SECTION_DESIGN_PROCEDURE_TEMPLATE = r"""
## 12. Your Task: Required Design Procedure

Follow these steps and **write the justification in top-of-file comments** of your generated heuristic:

1. **Name the bottleneck.** Read the {domain_name} HDDL and identify the single constraint that most serializes
   work. One-sentence comment.
2. **List 2–3 independent admissible lower bounds.** For each, write a one-line admissibility sketch: why is
   this term a lower bound on unavoidable cost?
3. **Implement `initialize(self, model)`.** Parse fact names, build per-predicate id maps, precompute goal
   structures, compute distance matrices or per-block stacks if relevant. Preprocessing here is free.
4. **Implement `__call__(self, node)` to return `max(term1, term2, term3)`.** Each term must use O(1) bit tests.
   No iteration over model-sized collections.
5. **Add tie-breaking penalties** (symmetry-breaking, goal ordering) ONLY if the domain has interchangeable
   parameters. Use sub-unit magnitudes (`* 0.001`) if preserving admissibility matters; otherwise add directly.

Return the complete heuristic as a Python class inheriting from `Heuristic`, with docstrings and comments
explaining your bottleneck analysis and admissibility sketches.
"""


_INT_CALL = "node: HTNNode) -> int:"


def _as_float_annotation(section: str, expected: int = 1) -> str:
    """
    Annotate __call__ -> float, matching the sub-unit tie-breakers §10 asks for.

    expected=None for a worked example, which may hold any number of __call__
    signatures. Helper methods such as
    _calculate_heuristic keep their own annotations: only __call__ is the interface.
    """
    if expected is not None and section.count(_INT_CALL) != expected:
        raise ValueError(f"expected {expected} '{_INT_CALL}' in the section, found {section.count(_INT_CALL)}")
    return section.replace(_INT_CALL, "node: HTNNode) -> float:")


def generate_prompt(
    domain_name: str,
    domain_content: str,
    smallest_instance_name: str,
    smallest_instance_content: str,
    largest_instance_name: str,
    largest_instance_content: str,
    example_heuristic_content: str,
    class_name: str,
    param_name: str,
    counter: int,
    domain_hints: str = "",
    components: dict = None,
    example_domain: str = None,
) -> str:
    """
    Generate the complete prompt for LLM heuristic generation.

    Args:
        domain_name: Display name of the domain (e.g., "Towers", "Barman")
        domain_content: Content of the domain.hddl file
        smallest_instance_name: Name of smallest instance file
        smallest_instance_content: Content of smallest instance
        largest_instance_name: Name of largest instance file
        largest_instance_content: Content of largest instance
        example_heuristic_content: Content of a working example heuristic
        class_name: Name for the generated class (e.g., "TowersGemini25ProHeuristic5")
        param_name: Name for __init__ parameter (e.g., "towers_gemini25pro_heuristic_5")
        counter: Heuristic number
        domain_hints: Optional domain-specific hints from domains.yaml
        components: Optional dict gating ablatable sections. See DEFAULT_COMPONENTS
                    for keys. Missing keys default to True (include).
        example_domain: Display name of the domain the §7 example was written for,
                        or None when unknown or not applicable (stub, placeholder).

    Returns:
        Complete prompt string
    """
    if components is None:
        components = DEFAULT_COMPONENTS

    def included(key: str) -> bool:
        return components.get(key, True)

    # §2.5 — domain hints (only non-empty when both flag is on AND hints exist)
    if included('domain_hints') and domain_hints:
        section_25 = (
            "## 2.5 Domain-Specific Hints (from prior analysis)\n\n"
            "These insights were discovered through extensive experimentation on this domain. Use them.\n\n"
            f"{domain_hints}\n"
        )
    else:
        section_25 = ""

    # §§3-5
    section_critical = _SECTION_CRITICAL_FACT_FORMAT if included('critical_fact_format') else ""

    # §6 always included
    section_interface = _SECTION_REQUIRED_INTERFACE
    section_interface = _as_float_annotation(section_interface)

    # §7 — example heuristic (uses example_heuristic_content). When ablated, falls
    # back to a weak stub rather than empty string (paper's "replace when catastrophic"
    # pattern — full removal collapses coverage to zero).
    if included('example_heuristic'):
        other_domain = ""
        if example_domain and example_domain.lower() != domain_name.lower():
            these = "These examples were" if " and " in example_domain else "This example was"
            noun = "domains" if " and " in example_domain else "domain"
            other_domain = (f"{these} written for the **{example_domain}** {noun}, not for "
                            f"**{domain_name}**, the domain you are writing a heuristic for.\n")
        example_heuristic_content = _as_float_annotation(example_heuristic_content, expected=None)
        section_example = (
            "\n## 7. Working Example Heuristic\n\n"
            f"{other_domain}"
            "This is a WORKING heuristic that correctly parses facts and reduces node expansion.\n"
            "Study this example carefully - it demonstrates all the patterns you need.\n\n"
            "```python\n"
            f"{example_heuristic_content}\n"
            "```\n"
        )
    else:
        section_example = _STUB_EXAMPLE_HEURISTIC
        section_example = _as_float_annotation(section_example)

    # §§8-9.1 — Pytrich API. When ablated, replaced with a minimal API reference
    # rather than empty (the LLM cannot write valid code without knowing the
    # attribute/method names it must call).
    section_api = _SECTION_PYTRICH_API if included('pytrich_api') else _STUB_PYTRICH_API

    # §10
    section_admissibility = _SECTION_ADMISSIBILITY if included('admissibility') else ""

    # §11
    section_win_lose = _SECTION_WIN_LOSE if included('win_lose_patterns') else ""

    # §12 (uses domain_name)
    if included('design_procedure'):
        section_design = _SECTION_DESIGN_PROCEDURE_TEMPLATE.format(domain_name=domain_name)
    else:
        section_design = ""

    section_cost = _SECTION_COST_MODEL

    prompt_template = f"""
# Task: Generate a Domain-Dependent Heuristic for HDDL Planning

You are an expert in hierarchical planning and heuristic design. Your task is to analyze the provided domain and create an effective domain-dependent heuristic for the **{domain_name}** domain.

Name the class as `{class_name}` and ensure it inherits from `Heuristic`. The heuristic should be designed to work with the Pytrich Planner framework.
Name the __init__ parameter name all lower case `{param_name}`.
Any variable in the code named {domain_name.lower()}_heuristic_{counter} should be renamed to {param_name}.
{section_cost}
## 1. Target Domain Definition

### Domain File: {domain_name.lower()}.hddl
```hddl
{domain_content}
```

## 2. Training Set Instances

### Smallest Instance: {smallest_instance_name}
```hddl
{smallest_instance_content}
```

### Largest Instance: {largest_instance_name}
```hddl
{largest_instance_content}
```

{section_25}{section_critical}{section_interface}{section_example}{section_api}{section_admissibility}{section_win_lose}{section_design}
"""

    return textwrap.dedent(prompt_template).strip()


def generate_refinement_prompt(
    base_prompt: str,
    iteration: int,
    previous_code: str,
    test_results: dict,
    baseline_results: dict,
    instance_name: str,
    new_class_name: str,
    new_param_name: str,
) -> str:
    """
    Generate a refinement prompt that wraps the base prompt with feedback.

    Args:
        base_prompt: The complete base prompt (from generate_prompt)
        iteration: Current iteration number (0-indexed, so this is being asked to produce iteration+1)
        previous_code: Source code of the previous heuristic version
        test_results: Dict with keys: status, nodes_expanded, search_time, solution_size, error
        baseline_results: Dict with same keys for TDG baseline (or None if unavailable)
        instance_name: Name of the test instance
        new_class_name: Class name for the new version
        new_param_name: Parameter name for the new version

    Returns:
        Complete refinement prompt string
    """
    # Build the results table
    your_status = test_results.get('status', 'UNKNOWN')
    your_nodes = test_results.get('nodes_expanded', 'N/A')
    your_time = test_results.get('search_time', 'N/A')
    your_sol = test_results.get('solution_size', 'N/A')

    if baseline_results:
        bl_status = baseline_results.get('status', 'N/A')
        bl_nodes = baseline_results.get('nodes_expanded', 'N/A')
        bl_time = baseline_results.get('search_time', 'N/A')
        bl_sol = baseline_results.get('solution_size', 'N/A')
    else:
        bl_status = bl_nodes = bl_time = bl_sol = 'N/A'

    results_table = (
        f"| Metric | Your Heuristic | TDG Baseline |\n"
        f"|--------|---------------|---------------|\n"
        f"| Status | {your_status} | {bl_status} |\n"
        f"| Nodes Expanded | {your_nodes} | {bl_nodes} |\n"
        f"| Search Time | {your_time}s | {bl_time}s |\n"
        f"| Solution Size | {your_sol} | {bl_sol} |"
    )

    # Build error/timeout section
    error_section = ""
    error_msg = test_results.get('error', '')
    if error_msg:
        error_section = f"\n**ERROR**: Your heuristic failed with:\n```\n{error_msg}\n```\n"
    elif your_status == 'TIMEOUT':
        error_section = (
            "\n**TIMEOUT**: Your heuristic timed out (60s limit). "
            "The `__call__` method is likely too slow or the heuristic values are not guiding search effectively.\n"
        )
    elif your_status == 'OUT OF MEMORY':
        error_section = (
            "\n**OUT OF MEMORY**: Your heuristic caused the search to exhaust memory. "
            "The heuristic values may not be pruning the search space effectively.\n"
        )

    # Detect structural smells in the previous code to tailor diagnostics
    iterates_full_model = any(
        pattern in previous_code
        for pattern in (
            "for f in self.model.facts",
            "for fact in self.model.facts",
            "for op in self.model.operators",
            "for o in self.model.operators",
        )
    )
    uses_max_composition = "max(" in previous_code and previous_code.count("max(") >= 1
    checks_state_bits = (">> " in previous_code and ") & 1" in previous_code) or "state_explicit_repr" in previous_code

    # Build auto-guidance based on what went wrong
    guidance_lines = []
    if error_msg:
        guidance_lines.append("- Fix the error shown above. Ensure the code is syntactically valid and handles all edge cases.")
    elif your_status == 'TIMEOUT':
        guidance_lines.append("- Improve heuristic accuracy so search expands fewer nodes, or reduce per-call computation time.")
        if iterates_full_model:
            guidance_lines.append(
                "- STRUCTURAL FIX: Your `__call__` iterates over `self.model.facts` or `self.model.operators`. "
                "That is O(model size) per node and will time out on large instances. Move that work into "
                "`initialize(model)` (build per-predicate id maps there) and keep `__call__` to O(1) bit tests like "
                "`(state >> fact_id) & 1`."
            )
    elif your_status == 'OUT OF MEMORY':
        guidance_lines.append("- Provide more informative heuristic values to prune the search space.")
    elif your_status == 'GOAL' and baseline_results and baseline_results.get('status') == 'GOAL':
        try:
            your_n = int(your_nodes)
            bl_n = int(bl_nodes)
            if your_n > bl_n:
                guidance_lines.append(f"- Your heuristic solved it but expanded {your_n} nodes vs TDG's {bl_n}. Improve heuristic accuracy to reduce node expansions.")
                diagnose = [
                    "- DIAGNOSE why you underperform TDG:",
                    f"  (a) State-awareness: {'MISSING — you never read current-state bits. ' if not checks_state_bits else 'present. '}"
                    "Does your h depend only on the task network? That is anti-pattern A1. Read state facts via "
                    "`(state >> fact.global_id) & 1` to refine static estimates.",
                    f"  (b) Multi-component max: {'MISSING — you return a single expression. ' if not uses_max_composition else 'present. '}"
                    "Consider 2–3 independent admissible lower bounds combined via `max(...)`, each capturing a "
                    "different unavoidable cost (travel, decomposition count, bottleneck).",
                    "  (c) Domain bottleneck: name the single constraint that serializes work (stack clearing, "
                    "single-carry, holding-shot, etc.) and add a dedicated `max` term for it.",
                ]
                guidance_lines.extend(diagnose)
            else:
                guidance_lines.append(f"- Good: you expanded fewer nodes ({your_n}) than TDG ({bl_n}). Push further:")
                if not uses_max_composition:
                    guidance_lines.append("  - Add another independent admissible lower bound and `max` it with your current term.")
                if not checks_state_bits:
                    guidance_lines.append("  - Your heuristic does not read state bits. Tighten an existing term by refining it with current-state information.")
                if uses_max_composition and checks_state_bits:
                    guidance_lines.append("  - Can you add a third `max` term, or tighten an existing term using goal-stability (goal facts currently true but whose support will be undone)?")
        except (ValueError, TypeError):
            guidance_lines.append("- Your heuristic found a solution. Try to reduce nodes expanded further.")
    else:
        guidance_lines.append("- Analyze the test results and improve heuristic quality.")

    guidance = "\n".join(guidance_lines) if guidance_lines else "- Improve overall heuristic quality."

    refinement_section = f"""

---

## REFINEMENT: Iteration {iteration + 1}

You are refining a previous heuristic attempt. Do NOT start from scratch — modify and improve the previous version.

**IMPORTANT**: Name the new class `{new_class_name}` and parameter `{new_param_name}`.

### Previous Attempt (v{iteration})

```python
{previous_code}
```

### Test Results on {instance_name}

{results_table}
{error_section}

### Refinement Instructions

Based on the results above, generate an improved version. Focus on:
{guidance}

Provide the complete improved heuristic code.
"""

    return base_prompt + textwrap.dedent(refinement_section)
