"""Agent tools. The calculator is an AST-whitelist evaluator: no eval(), no names, no calls."""
from __future__ import annotations
import ast, operator

class ToolError(Exception):
    pass

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def _eval(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return n.value
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        v = _eval(n.operand)
        return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        l, r = _eval(n.left), _eval(n.right)
        if isinstance(n.op, ast.Pow) and (abs(r) > 64 or abs(l) > 1e6):
            raise ToolError("exponent too large")
        try:
            return _BIN[type(n.op)](l, r)
        except ZeroDivisionError:
            raise ToolError("division by zero")
    raise ToolError("unsupported expression (only numbers and + - * / // % ** are allowed)")


def calculator(expression: str) -> str:
    expr = str(expression).replace(",", "").replace("^", "**").replace("×", "*").replace("÷", "/").strip()
    if not expr or len(expr) > 200:
        raise ToolError("empty or too long expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolError("syntax error")
    v = _eval(tree.body)
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.10g}" if isinstance(v, float) else str(v)


TOOLS = {"calculator": calculator}
