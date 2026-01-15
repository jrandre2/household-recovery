"""
Safe Expression Evaluator using AST parsing.

This module provides secure evaluation of condition strings from LLM output.
Only allows a whitelist of safe operations to prevent code injection attacks.

Educational Note:
-----------------
The Python `ast` module lets us parse Python code into an Abstract Syntax Tree
without executing it. We can then walk the tree and verify each node is safe
before evaluation. This is much safer than `eval()` on untrusted input.

Example:
    >>> evaluator = SafeExpressionEvaluator()
    >>> expr = "ctx['avg_neighbor_recovery'] > 0.5"
    >>> func = evaluator.compile(expr)
    >>> result = func({'avg_neighbor_recovery': 0.7})
    True
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Whitelist of allowed context keys (variables the LLM can reference)
ALLOWED_CTX_KEYS = frozenset({
    'avg_neighbor_recovery',
    'avg_infra_func',
    'avg_business_avail',
    'num_neighbors',
    'resilience',
    'resilience_category',
    'household_income',
    'income_level',
})

# Allowed comparison operators
COMPARISON_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# Allowed boolean operators
BOOL_OPS = {
    ast.And: lambda values: all(values),
    ast.Or: lambda values: any(values),
}

# Allowed unary operators
UNARY_OPS = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
}

# Allowed binary operators (for arithmetic if needed)
BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


class UnsafeExpressionError(Exception):
    """Raised when an expression contains disallowed operations."""
    pass


class SafeExpressionEvaluator:
    """
    Evaluates Python expressions safely using AST parsing.

    Only allows:
    - Subscript access on 'ctx' dict with allowed keys
    - Numeric and string literals
    - Comparison operations (<, >, <=, >=, ==, !=)
    - Boolean operations (and, or, not)
    - Basic arithmetic (+, -, *, /)

    Disallows:
    - Function calls
    - Attribute access (except on ctx)
    - Import statements
    - Assignment
    - Any other operations
    """

    def __init__(self, allowed_keys: frozenset[str] = ALLOWED_CTX_KEYS):
        self.allowed_keys = allowed_keys

    def compile(self, expression: str) -> Callable[[dict], bool]:
        """
        Compile an expression string into a safe callable.

        Args:
            expression: Python expression string like "ctx['key'] > 0.5"

        Returns:
            A function that takes a context dict and returns bool

        Raises:
            UnsafeExpressionError: If expression contains disallowed operations
            SyntaxError: If expression is not valid Python
        """
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid expression syntax: {expression}") from e

        # Validate the AST
        self._validate_node(tree.body)

        # Create a safe evaluator function
        def safe_eval(ctx: dict) -> bool:
            return self._evaluate_node(tree.body, ctx)

        return safe_eval

    def _validate_node(self, node: ast.AST) -> None:
        """Recursively validate that all AST nodes are allowed."""

        if isinstance(node, ast.Expression):
            self._validate_node(node.body)

        elif isinstance(node, ast.Compare):
            self._validate_node(node.left)
            for op in node.ops:
                if type(op) not in COMPARISON_OPS:
                    raise UnsafeExpressionError(f"Disallowed comparison operator: {type(op).__name__}")
            for comparator in node.comparators:
                self._validate_node(comparator)

        elif isinstance(node, ast.BoolOp):
            if type(node.op) not in BOOL_OPS:
                raise UnsafeExpressionError(f"Disallowed boolean operator: {type(node.op).__name__}")
            for value in node.values:
                self._validate_node(value)

        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in UNARY_OPS:
                raise UnsafeExpressionError(f"Disallowed unary operator: {type(node.op).__name__}")
            self._validate_node(node.operand)

        elif isinstance(node, ast.BinOp):
            if type(node.op) not in BINARY_OPS:
                raise UnsafeExpressionError(f"Disallowed binary operator: {type(node.op).__name__}")
            self._validate_node(node.left)
            self._validate_node(node.right)

        elif isinstance(node, ast.Subscript):
            # Only allow ctx['key'] pattern
            if isinstance(node.value, ast.Name) and node.value.id == 'ctx':
                if isinstance(node.slice, ast.Constant):
                    key = node.slice.value
                    if key not in self.allowed_keys:
                        raise UnsafeExpressionError(
                            f"Disallowed context key: '{key}'. "
                            f"Allowed keys: {sorted(self.allowed_keys)}"
                        )
                else:
                    raise UnsafeExpressionError("Context subscript must be a string literal")
            else:
                raise UnsafeExpressionError("Subscript only allowed on 'ctx' variable")

        elif isinstance(node, ast.Constant):
            # Allow numeric and string literals
            if not isinstance(node.value, (int, float, str, bool, type(None))):
                raise UnsafeExpressionError(f"Disallowed constant type: {type(node.value)}")

        elif isinstance(node, ast.Name):
            # Only allow 'ctx' as a standalone name
            if node.id != 'ctx':
                raise UnsafeExpressionError(f"Disallowed variable name: '{node.id}'")

        elif isinstance(node, ast.IfExp):
            # Allow ternary: x if condition else y
            self._validate_node(node.test)
            self._validate_node(node.body)
            self._validate_node(node.orelse)

        else:
            raise UnsafeExpressionError(
                f"Disallowed AST node type: {type(node).__name__}. "
                "Only comparisons, boolean ops, and ctx subscripts are allowed."
            )

    def _evaluate_node(self, node: ast.AST, ctx: dict) -> Any:
        """Recursively evaluate a validated AST node."""

        if isinstance(node, ast.Compare):
            left = self._evaluate_node(node.left, ctx)
            result = True
            current = left
            for op, comparator in zip(node.ops, node.comparators):
                right = self._evaluate_node(comparator, ctx)
                op_func = COMPARISON_OPS[type(op)]
                result = result and op_func(current, right)
                current = right
            return result

        elif isinstance(node, ast.BoolOp):
            values = [self._evaluate_node(v, ctx) for v in node.values]
            return BOOL_OPS[type(node.op)](values)

        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand, ctx)
            return UNARY_OPS[type(node.op)](operand)

        elif isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left, ctx)
            right = self._evaluate_node(node.right, ctx)
            return BINARY_OPS[type(node.op)](left, right)

        elif isinstance(node, ast.Subscript):
            # We know from validation this is ctx['key']
            key = node.slice.value
            return ctx.get(key)

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            if node.id == 'ctx':
                return ctx
            raise UnsafeExpressionError(f"Unknown name: {node.id}")

        elif isinstance(node, ast.IfExp):
            test = self._evaluate_node(node.test, ctx)
            if test:
                return self._evaluate_node(node.body, ctx)
            else:
                return self._evaluate_node(node.orelse, ctx)

        else:
            raise UnsafeExpressionError(f"Cannot evaluate node type: {type(node).__name__}")


# Module-level evaluator instance
_evaluator = SafeExpressionEvaluator()


def compile_condition(expression: str) -> Callable[[dict], bool]:
    """
    Convenience function to compile a condition expression.

    Args:
        expression: Python expression like "ctx['avg_neighbor_recovery'] > 0.5"

    Returns:
        A callable that takes a context dict and returns bool
    """
    return _evaluator.compile(expression)


def validate_condition(expression: str) -> tuple[bool, str | None]:
    """
    Check if a condition expression is safe without compiling.

    Args:
        expression: Python expression to validate

    Returns:
        Tuple of (is_valid, error_message_or_none)
    """
    try:
        _evaluator.compile(expression)
        return True, None
    except (UnsafeExpressionError, SyntaxError) as e:
        return False, str(e)


# Self-test when run directly
if __name__ == "__main__":
    print("Testing SafeExpressionEvaluator...\n")

    # Test valid expressions
    valid_tests = [
        ("ctx['avg_neighbor_recovery'] > 0.5", {'avg_neighbor_recovery': 0.7}, True),
        ("ctx['avg_neighbor_recovery'] > 0.5", {'avg_neighbor_recovery': 0.3}, False),
        ("ctx['resilience'] < 0.35 and ctx['income_level'] == 'low'",
         {'resilience': 0.2, 'income_level': 'low'}, True),
        ("ctx['num_neighbors'] > 4 or ctx['avg_infra_func'] > 0.8",
         {'num_neighbors': 3, 'avg_infra_func': 0.9}, True),
    ]

    for expr, ctx, expected in valid_tests:
        func = compile_condition(expr)
        result = func(ctx)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {expr}")
        print(f"       ctx={ctx}")
        print(f"       expected={expected}, got={result}\n")

    # Test invalid expressions (should raise errors)
    invalid_tests = [
        ("__import__('os').system('rm -rf /')", "Import/call not allowed"),
        ("ctx['unknown_key'] > 0.5", "Unknown key"),
        ("eval('1+1')", "Function call not allowed"),
        ("open('/etc/passwd').read()", "Function call not allowed"),
    ]

    print("\nTesting rejection of unsafe expressions:")
    for expr, reason in invalid_tests:
        is_valid, error = validate_condition(expr)
        status = "PASS (rejected)" if not is_valid else "FAIL (should reject)"
        print(f"[{status}] {expr}")
        print(f"       Reason: {reason}")
        if error:
            print(f"       Error: {error}\n")
