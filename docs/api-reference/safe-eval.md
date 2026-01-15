# Safe Eval Module

Secure expression evaluation for LLM-generated heuristic conditions.

```python
from household_recovery.safe_eval import (
    SafeExpressionEvaluator,
    compile_condition,
    validate_condition,
    UnsafeExpressionError,
    ALLOWED_CTX_KEYS
)
```

## Why Safe Evaluation?

LLM-generated heuristics include condition expressions like:
```python
ctx['avg_neighbor_recovery'] > 0.5
```

Using Python's `eval()` on untrusted input is dangerous. This module uses AST (Abstract Syntax Tree) parsing to validate expressions before execution, preventing code injection attacks.

---

## SafeExpressionEvaluator

Evaluates Python expressions safely using AST parsing.

### Allowed Operations

- **Subscript access** on 'ctx' dict with allowed keys
- **Numeric and string literals**
- **Comparison operations**: `<`, `>`, `<=`, `>=`, `==`, `!=`
- **Boolean operations**: `and`, `or`, `not`
- **Basic arithmetic**: `+`, `-`, `*`, `/`
- **Ternary expressions**: `x if condition else y`

### Disallowed Operations

- Function calls
- Attribute access (except on ctx)
- Import statements
- Assignment
- Any other operations

### Constructor

```python
evaluator = SafeExpressionEvaluator(
    allowed_keys=ALLOWED_CTX_KEYS
)
```

### Methods

#### `compile(expression) -> Callable[[dict], bool]`

Compile an expression string into a safe callable.

```python
evaluator = SafeExpressionEvaluator()
func = evaluator.compile("ctx['avg_neighbor_recovery'] > 0.5")

result = func({'avg_neighbor_recovery': 0.7})  # True
result = func({'avg_neighbor_recovery': 0.3})  # False
```

**Raises:**
- `UnsafeExpressionError`: If expression contains disallowed operations
- `SyntaxError`: If expression is not valid Python

---

## compile_condition

Convenience function to compile a condition expression.

```python
func = compile_condition("ctx['avg_neighbor_recovery'] > 0.5")
result = func({'avg_neighbor_recovery': 0.7})  # True
```

---

## validate_condition

Check if a condition expression is safe without compiling.

```python
is_valid, error = validate_condition("ctx['avg_neighbor_recovery'] > 0.5")
if is_valid:
    print("Expression is safe")
else:
    print(f"Unsafe: {error}")
```

**Returns:** `tuple[bool, str | None]` - (is_valid, error_message_or_none)

---

## UnsafeExpressionError

Exception raised when an expression contains disallowed operations.

```python
try:
    func = compile_condition("__import__('os').system('rm -rf /')")
except UnsafeExpressionError as e:
    print(f"Blocked: {e}")
```

---

## ALLOWED_CTX_KEYS

Whitelist of allowed context keys that heuristics can reference.

```python
ALLOWED_CTX_KEYS = frozenset({
    'avg_neighbor_recovery',    # Average neighbor recovery (0-1)
    'avg_infra_func',           # Average infrastructure functionality (0-1)
    'avg_business_avail',       # Average business availability (0-1)
    'num_neighbors',            # Number of neighbors (int)
    'resilience',               # Household resilience (0-1)
    'resilience_category',      # 'low', 'medium', 'high'
    'household_income',         # Annual income (float)
    'income_level',             # 'low', 'middle', 'high'
})
```

---

## Valid Expression Examples

```python
# Simple comparison
"ctx['avg_neighbor_recovery'] > 0.5"

# Multiple conditions with AND
"ctx['resilience'] < 0.35 and ctx['income_level'] == 'low'"

# Multiple conditions with OR
"ctx['num_neighbors'] > 4 or ctx['avg_infra_func'] > 0.8"

# Arithmetic in comparison
"ctx['avg_neighbor_recovery'] + ctx['avg_infra_func'] > 1.0"

# Ternary expression
"ctx['resilience'] if ctx['income_level'] == 'high' else 0.5"
```

---

## Invalid Expression Examples

These will raise `UnsafeExpressionError`:

```python
# Function calls
"len(ctx['avg_neighbor_recovery'])"  # No function calls

# Unknown variables
"ctx['unknown_key'] > 0.5"  # Key not in whitelist

# Import statements
"__import__('os').system('ls')"  # No imports

# Attribute access
"ctx.get('key', 0)"  # No attribute access

# File operations
"open('/etc/passwd').read()"  # No function calls
```

---

## Comparison Operators

| Operator | Function |
|----------|----------|
| `<` | `operator.lt` |
| `<=` | `operator.le` |
| `>` | `operator.gt` |
| `>=` | `operator.ge` |
| `==` | `operator.eq` |
| `!=` | `operator.ne` |

---

## Boolean Operators

| Operator | Function |
|----------|----------|
| `and` | `all(values)` |
| `or` | `any(values)` |
| `not` | `operator.not_` |

---

## Binary Operators

| Operator | Function |
|----------|----------|
| `+` | `operator.add` |
| `-` | `operator.sub` |
| `*` | `operator.mul` |
| `/` | `operator.truediv` |

---

## How It Works

1. **Parse**: Convert expression string to AST using `ast.parse()`
2. **Validate**: Walk the AST tree, checking each node against whitelist
3. **Compile**: Create a function that safely evaluates the validated tree
4. **Execute**: Call the compiled function with a context dictionary

This approach is much safer than `eval()` because:
- Only whitelisted operations are allowed
- The expression is validated before any execution
- Unknown keys or operations are rejected
- No way to escape to arbitrary code execution
