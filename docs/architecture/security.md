# Security Model

Safe evaluation of LLM-generated code.

## The Problem

The RAG pipeline uses an LLM to generate heuristic conditions:

```python
"ctx['avg_neighbor_recovery'] > 0.5"
```

These conditions need to be evaluated at runtime. Using Python's `eval()` on untrusted input is dangerous:

```python
# DANGEROUS - never do this!
eval("__import__('os').system('rm -rf /')")
```

## The Solution: AST-Based Safe Evaluation

Instead of `eval()`, we:
1. Parse the expression into an Abstract Syntax Tree (AST)
2. Walk the tree, validating each node against a whitelist
3. Only execute if all nodes are safe

```python
import ast

tree = ast.parse(expression, mode='eval')
# tree.body is now a validated AST that can be safely evaluated
```

## Allowed Operations

### Context Access

Only `ctx['key']` with whitelisted keys:

```python
ALLOWED_CTX_KEYS = {
    'avg_neighbor_recovery',
    'avg_infra_func',
    'avg_business_avail',
    'num_neighbors',
    'resilience',
    'resilience_category',
    'household_income',
    'income_level',
    'perception_type',
    'damage_severity',
    'recovery_state',
    'is_feasible',
    'is_adequate',
    'is_habitable',
    'repair_cost',
    'available_resources',
    'time_step',
    'months_since_disaster',
    'avg_neighbor_recovered_binary',
}
```

### Comparison Operators

```python
COMPARISON_OPS = {
    ast.Lt: operator.lt,      # <
    ast.LtE: operator.le,     # <=
    ast.Gt: operator.gt,      # >
    ast.GtE: operator.ge,     # >=
    ast.Eq: operator.eq,      # ==
    ast.NotEq: operator.ne,   # !=
}
```

### Boolean Operators

```python
BOOL_OPS = {
    ast.And: all,  # and
    ast.Or: any,   # or
}

UNARY_OPS = {
    ast.Not: operator.not_,   # not
    ast.USub: operator.neg,   # -
}
```

### Arithmetic Operators

```python
BINARY_OPS = {
    ast.Add: operator.add,    # +
    ast.Sub: operator.sub,    # -
    ast.Mult: operator.mul,   # *
    ast.Div: operator.truediv,  # /
}
```

### Literals

```python
# Allowed constant types
(int, float, str, bool, type(None))
```

## Disallowed Operations

### Function Calls

```python
# BLOCKED
"len(ctx['avg_neighbor_recovery'])"
"eval('1+1')"
"open('/etc/passwd')"
"print('hello')"
```

### Attribute Access

```python
# BLOCKED
"ctx.get('key', 0)"
"''.join([])"
"[].append(1)"
```

### Import Statements

```python
# BLOCKED
"__import__('os')"
"import os"
```

### Unknown Variables

```python
# BLOCKED
"unknown_var > 0.5"
"ctx['unknown_key'] > 0.5"
```

## Implementation

### Validation Phase

```python
def _validate_node(self, node: ast.AST) -> None:
    if isinstance(node, ast.Compare):
        self._validate_node(node.left)
        for op in node.ops:
            if type(op) not in COMPARISON_OPS:
                raise UnsafeExpressionError(f"Disallowed: {type(op)}")
        for comparator in node.comparators:
            self._validate_node(comparator)

    elif isinstance(node, ast.Subscript):
        # Only allow ctx['key']
        if isinstance(node.value, ast.Name) and node.value.id == 'ctx':
            if isinstance(node.slice, ast.Constant):
                key = node.slice.value
                if key not in self.allowed_keys:
                    raise UnsafeExpressionError(f"Disallowed key: {key}")

    elif isinstance(node, ast.Call):
        # ALL function calls are blocked
        raise UnsafeExpressionError("Function calls not allowed")

    # ... other node types
```

### Evaluation Phase

Only after validation succeeds:

```python
def _evaluate_node(self, node: ast.AST, ctx: dict) -> Any:
    if isinstance(node, ast.Compare):
        left = self._evaluate_node(node.left, ctx)
        result = True
        for op, comp in zip(node.ops, node.comparators):
            right = self._evaluate_node(comp, ctx)
            result = result and COMPARISON_OPS[type(op)](left, right)
            left = right
        return result

    elif isinstance(node, ast.Subscript):
        key = node.slice.value
        return ctx.get(key)

    # ... other node types
```

## Usage

### Compiling Conditions

```python
from household_recovery.safe_eval import compile_condition

# Safe - compiles successfully
func = compile_condition("ctx['avg_neighbor_recovery'] > 0.5")
result = func({'avg_neighbor_recovery': 0.7})  # True

# Unsafe - raises UnsafeExpressionError
func = compile_condition("__import__('os').system('ls')")
```

### Validating Without Compiling

```python
from household_recovery.safe_eval import validate_condition

is_valid, error = validate_condition("ctx['unknown_key'] > 0.5")
# is_valid = False
# error = "Disallowed context key: 'unknown_key'. Allowed keys: [...]"
```

## Error Messages

### Unknown Key

```
UnsafeExpressionError: Disallowed context key: 'unknown_key'.
Allowed keys: ['avg_business_avail', 'avg_infra_func', ...]
```

### Function Call

```
UnsafeExpressionError: Disallowed AST node type: Call.
Only comparisons, boolean ops, and ctx subscripts are allowed.
```

### Disallowed Operator

```
UnsafeExpressionError: Disallowed comparison operator: In
```

## Security Guarantees

### What's Guaranteed

1. **No arbitrary code execution** - Only whitelisted operations
2. **No file access** - No `open()`, no `__import__()`
3. **No network access** - No socket operations
4. **No system commands** - No `os.system()`, no `subprocess`
5. **Bounded computation** - No loops, no recursion

### What's NOT Guaranteed

1. **Correctness** - Expressions might have logical errors
2. **Performance** - Complex expressions could be slow
3. **Denial of Service** - Very long expressions could parse slowly

## Testing

```python
# Test valid expressions
valid = [
    "ctx['avg_neighbor_recovery'] > 0.5",
    "ctx['resilience'] < 0.35 and ctx['income_level'] == 'low'",
    "ctx['num_neighbors'] > 4 or ctx['avg_infra_func'] > 0.8",
]

for expr in valid:
    func = compile_condition(expr)
    assert callable(func)

# Test invalid expressions
invalid = [
    "__import__('os').system('rm -rf /')",
    "ctx['unknown_key'] > 0.5",
    "eval('1+1')",
    "open('/etc/passwd').read()",
]

for expr in invalid:
    is_valid, error = validate_condition(expr)
    assert not is_valid
```

## Best Practices

### For LLM Prompts

Include the allowed keys explicitly:

```
condition: valid Python expression using ONLY these ctx keys:
'avg_neighbor_recovery', 'avg_infra_func', 'avg_business_avail',
'num_neighbors', 'resilience', 'resilience_category',
'household_income', 'income_level'
```

### For Validation

Always validate before use:

```python
is_valid, error = validate_condition(expression)
if not is_valid:
    logger.warning(f"Invalid condition: {error}")
    continue  # Skip this heuristic
```

### For Debugging

Enable logging to see validation failures:

```python
import logging
logging.getLogger('household_recovery.safe_eval').setLevel(logging.DEBUG)
```

## Comparison to Alternatives

| Approach | Safety | Flexibility | Complexity |
|----------|--------|-------------|------------|
| `eval()` | None | Full | None |
| Regex validation | Medium | Low | Medium |
| AST parsing (this) | High | Medium | Medium |
| Custom DSL | High | Low | High |
| Sandboxed execution | High | High | High |

AST parsing provides a good balance of safety and flexibility for our use case.

## Next Steps

- [RAG Architecture](rag-architecture.md) - How heuristics are generated
- [Agent Model](agent-model.md) - How heuristics are applied
- [Module Relationships](module-relationships.md) - Where safe_eval fits
