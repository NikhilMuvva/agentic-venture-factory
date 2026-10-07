from __future__ import annotations

import ast
import re
from typing import Any

import sympy as sp

from models import QuestionDraft, ResponseType, VerificationResult, XYGraphVisual


_ALLOWED_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left ** right,
}
_ALLOWED_UNARY_OPERATORS = {ast.UAdd: lambda value: value, ast.USub: lambda value: -value}


def _parse_expression(expression: str, variables: dict[str, Any] | None = None) -> Any:
    expression = expression.strip()
    expression = expression.replace("×", "*").replace("^", "**")
    expression = re.sub(r"(?<=\d)\s*(?=[A-Za-z])", "*", expression)
    symbols = {name: sp.Symbol(name, real=True) for name in (variables or {})}
    for name in ("x", "y", "z", "t", "n", "r", "h", "m", "a", "b", "c", "k"):
        symbols.setdefault(name, sp.Symbol(name, real=True))
    symbols["pi"] = sp.pi
    tree = ast.parse(expression, mode="eval")

    def convert(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINARY_OPERATORS:
            return _ALLOWED_BINARY_OPERATORS[type(node.op)](convert(node.left), convert(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPERATORS:
            return _ALLOWED_UNARY_OPERATORS[type(node.op)](convert(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
            return sp.sqrt(convert(node.args[0]))
        raise ValueError("Expression contains unsupported syntax.")

    value = convert(tree)
    for name, numeric_value in (variables or {}).items():
        value = value.subs(symbols[name], sp.Rational(str(numeric_value)))
    return sp.simplify(value)


def _parse_equation(equation: str, variables: dict[str, Any] | None = None) -> sp.Equality:
    parts = equation.split("=")
    if len(parts) != 2:
        raise ValueError("An equation must contain exactly one equals sign.")
    return sp.Eq(_parse_expression(parts[0], variables), _parse_expression(parts[1], variables))


def _parse_answer(value: str) -> Any:
    cleaned = value.strip().replace("$", "")
    cleaned = cleaned.replace("°", "")
    cleaned = re.sub(
        r"\s*(?:degrees?|units?|square\s+(?:units?|[a-z]+)|cubic\s+(?:units?|[a-z]+)|"
        r"(?:mm|cm|m|km|inches?|feet|ft|meters?|kilometers?)(?:\^?[23])?)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^[a-zA-Z]\s*=\s*", "", cleaned)
    cleaned = cleaned.replace("%", "/100")
    parts = re.split(r"\s+(?:and|or)\s+", cleaned)
    if len(parts) > 1:
        return tuple(_parse_expression(re.sub(r"^[a-zA-Z]\s*=\s*", "", part)) for part in parts)
    if cleaned.startswith("(") and cleaned.endswith(")") and "," in cleaned:
        inner = cleaned[1:-1]
        return tuple(_parse_expression(part) for part in inner.split(","))
    cleaned = cleaned.replace(",", "")
    return _parse_expression(cleaned)


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    try:
        return bool(sp.simplify(left - right) == 0)
    except (TypeError, AttributeError):
        return left == right


def numeric_answers_equivalent(answer: str, accepted_answers: list[str]) -> bool:
    try:
        computed = _parse_answer(answer)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return False
    for accepted in accepted_answers:
        try:
            if _equivalent(computed, _parse_answer(accepted)):
                return True
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
            continue
    return False


def _format_answer(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return ", ".join(_format_answer(item) for item in value)
    return str(sp.simplify(value))


def _real_solutions(equation: sp.Equality, variable: sp.Symbol) -> list[Any]:
    solutions = sp.solve(equation, variable)
    return [solution for solution in solutions if solution.is_real is not False]


def inequality_point_state(expression: str, x_value: float, y_value: float) -> str:
    parts = re.split(r"(<=|>=|<|>)", expression, maxsplit=1)
    if len(parts) != 3:
        raise ValueError("Inequality must contain <, <=, >, or >=.")
    left = _parse_expression(parts[0])
    right = _parse_expression(parts[2])
    relation = parts[1]
    relation_factory = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[relation]
    x_symbol, y_symbol = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    substitutions = {
        x_symbol: sp.Rational(str(x_value)),
        y_symbol: sp.Rational(str(y_value)),
    }
    left_value = left.subs(substitutions)
    right_value = right.subs(substitutions)
    if sp.simplify(left_value - right_value) == 0:
        return "boundary"
    comparison = relation_factory(left_value, right_value)
    return "satisfies" if bool(comparison) else "outside"


def inequality_point_satisfies(expression: str, x_value: float, y_value: float) -> bool:
    return inequality_point_state(expression, x_value, y_value) == "satisfies" or (
        inequality_point_state(expression, x_value, y_value) == "boundary"
        and "=" in expression
    )


def _compute(draft: QuestionDraft) -> tuple[Any, str]:
    spec = draft.verification_spec
    variables = {item.name: item.value for item in spec.variables}
    method = spec.method

    if method == "graph_line":
        if not isinstance(draft.visual_spec, XYGraphVisual):
            raise ValueError("Graph-line verification requires an xy_graph visual.")
        lines = [series for series in draft.visual_spec.series if series.kind == "line"]
        if len(lines) != 1:
            raise ValueError("Graph-line verification requires exactly one line series.")
        series = lines[0]
        return (sp.Rational(str(series.slope)), sp.Rational(str(series.intercept))), "visual graph slope/intercept"

    if method == "linear_equation":
        if not spec.expression:
            raise ValueError("Equation verification requires an equation expression.")
        symbol = sp.Symbol(spec.variable, real=True)
        solutions = _real_solutions(_parse_equation(spec.expression, variables), symbol)
        if len(solutions) != 1:
            raise ValueError("Expected exactly one real solution.")
        return solutions[0], "sympy.solve(linear_equation)"

    if method == "triangle_angle":
        if len(spec.equations) != 1 or not spec.answer_expression:
            raise ValueError("Triangle-angle verification requires one angle-sum equation and an answer expression.")
        symbol = sp.Symbol(spec.variable, real=True)
        solutions = _real_solutions(_parse_equation(spec.equations[0], variables), symbol)
        if len(solutions) != 1:
            raise ValueError("Expected one real variable value from the triangle angle sum.")
        answer = _parse_expression(spec.answer_expression, {spec.variable: solutions[0]})
        return answer, "sympy triangle angle equation and expression"

    if method == "linear_system":
        equations = [_parse_equation(item, variables) for item in spec.equations]
        if len(equations) < 2:
            raise ValueError("A linear system requires at least two equations.")
        symbols = sorted(set().union(*(equation.free_symbols for equation in equations)), key=str)
        solutions = sp.solve(equations, symbols, dict=True)
        if len(solutions) != 1:
            raise ValueError("Expected one unique real system solution.")
        return tuple(solutions[0][symbol] for symbol in symbols), "sympy.solve(linear_system)"

    if method == "quadratic":
        if not spec.expression:
            raise ValueError("Quadratic verification requires an equation expression.")
        symbol = sp.Symbol(spec.variable, real=True)
        equation = _parse_equation(spec.expression, variables)
        if spec.operation == "discriminant":
            polynomial = sp.Poly(equation.lhs - equation.rhs, symbol)
            return sp.discriminant(polynomial.as_expr(), symbol), "sympy.discriminant"
        roots = _real_solutions(equation, symbol)
        if spec.operation == "sum_roots":
            return sp.simplify(sum(roots)), "sympy.solve(quadratic_roots_sum)"
        if spec.operation == "roots":
            return tuple(sorted(roots, key=lambda root: float(sp.N(root)))), "sympy.solve(quadratic_roots)"
        if len(roots) != 1:
            raise ValueError("Expected one real root for this quadratic answer.")
        return roots[0], "sympy.solve(quadratic)"

    if method == "inequality":
        if not spec.expression:
            raise ValueError("Inequality verification requires an expression.")
        relation_parts = re.split(r"(<=|>=|<|>)", spec.expression, maxsplit=1)
        if len(relation_parts) != 3:
            raise ValueError("Inequality must contain <, <=, >, or >=.")
        left = _parse_expression(relation_parts[0])
        right = _parse_expression(relation_parts[2])
        relation = relation_parts[1]
        if not isinstance(draft.visual_spec, XYGraphVisual):
            raise ValueError("Inequality verification requires an XY graph visual.")
        boundaries = [
            series for series in draft.visual_spec.series
            if series.kind == "inequality_boundary"
        ]
        if len(boundaries) != 1:
            raise ValueError("Inequality graph must contain exactly one inequality boundary.")
        boundary = boundaries[0]
        x_symbol, y_symbol = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
        solved_boundary = sp.solve(sp.Eq(left, right), y_symbol)
        if len(solved_boundary) != 1:
            raise ValueError("Inequality boundary must be expressible as one line y=mx+b.")
        boundary_expression = sp.expand(solved_boundary[0])
        expected_slope = sp.diff(boundary_expression, x_symbol)
        expected_intercept = boundary_expression.subs(x_symbol, 0)
        if (
            not _equivalent(boundary.slope, expected_slope)
            or not _equivalent(boundary.intercept, expected_intercept)
        ):
            raise ValueError("Rendered inequality boundary does not match its expression.")
        below_state = inequality_point_state(spec.expression, 0, float(expected_intercept) - 1)
        above_state = inequality_point_state(spec.expression, 0, float(expected_intercept) + 1)
        if below_state == "satisfies" and above_state == "outside":
            expected_relation = "<=" if "=" in relation else "<"
        elif above_state == "satisfies" and below_state == "outside":
            expected_relation = ">=" if "=" in relation else ">"
        else:
            raise ValueError("Inequality solution is not a single half-plane.")
        if boundary.relation != expected_relation:
            raise ValueError("Rendered inequality shading does not match the inequality solution side.")
        if not spec.choice_points:
            raise NotImplementedError("Inequality verification requires labeled choice_points.")
        if draft.choices is None:
            raise ValueError("Inequality MCQ verification requires visible answer choices.")
        visible_choices = draft.choices.model_dump()
        for point in spec.choice_points:
            displayed_point = _parse_answer(visible_choices[point.label])
            specified_point = (sp.Rational(str(point.x)), sp.Rational(str(point.y)))
            if not _equivalent(displayed_point, specified_point):
                raise ValueError(f"Choice {point.label} text does not match its verified coordinate.")
        relation_factory = {
            "<": sp.Lt,
            "<=": sp.Le,
            ">": sp.Gt,
            ">=": sp.Ge,
        }[relation]
        satisfying_choices = []
        for point in spec.choice_points:
            candidate = relation_factory(left, right, evaluate=False).subs({
                x_symbol: sp.Rational(str(point.x)),
                y_symbol: sp.Rational(str(point.y)),
            })
            if bool(candidate):
                satisfying_choices.append(point.label)
        return tuple(satisfying_choices), "sympy inequality point evaluation"

    if method in ("function_evaluation", "expression_evaluation", "formula", "ratio", "percentage"):
        if not spec.expression:
            raise ValueError("Evaluation requires an expression.")
        return _parse_expression(spec.expression, variables), f"sympy expression evaluation ({method})"

    if method in ("mean", "median", "range"):
        if not spec.values:
            raise ValueError("Statistics verification requires data values.")
        values = [sp.Rational(str(value)) for value in spec.values]
        if method == "mean":
            return sp.Rational(sum(values), len(values)), "python/sympy arithmetic mean"
        if method == "median":
            ordered_values = sorted(values)
            middle = len(ordered_values) // 2
            if len(ordered_values) % 2:
                median = ordered_values[middle]
            else:
                median = sp.simplify((ordered_values[middle - 1] + ordered_values[middle]) / 2)
            return median, "exact rational median"
        return max(values) - min(values), "python/sympy range"

    if method == "probability":
        if spec.numerator is None or spec.denominator is None or spec.denominator <= 0:
            raise ValueError("Probability verification requires a positive denominator and numerator.")
        return sp.Rational(str(spec.numerator)) / sp.Rational(str(spec.denominator)), "exact probability ratio"

    raise NotImplementedError("This verification method is not implemented.")


def verify_draft(draft: QuestionDraft) -> VerificationResult:
    method = draft.verification_spec.method
    if method == "llm_only":
        return VerificationResult(
            verified=False,
            notes="No deterministic verifier spec was provided for this item.",
            verification_method="llm_only",
        )

    try:
        computed, verification_method = _compute(draft)
        if method == "graph_line":
            computed_slope, computed_intercept = computed
            matching_labels = []
            x_symbol, y_symbol = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
            for label, answer in draft.choices.model_dump().items():
                try:
                    equation = _parse_equation(answer.replace("f(x)", "y"))
                    expression = sp.solve(equation, y_symbol)
                    if len(expression) != 1:
                        continue
                    polynomial = sp.Poly(expression[0], x_symbol)
                    slope = polynomial.coeff_monomial(x_symbol)
                    intercept = polynomial.coeff_monomial(1)
                    if _equivalent(slope, computed_slope) and _equivalent(intercept, computed_intercept):
                        matching_labels.append(label)
                except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
                    continue
        elif method == "inequality":
            matching_labels = list(computed)
        elif computed is sp.true or computed is sp.false:
            matching_labels = [
                label for label, answer in draft.choices.model_dump().items()
                if bool(_parse_answer(answer)) is bool(computed)
            ] if draft.choices else []
        elif draft.response_type == ResponseType.multiple_choice:
            matching_labels = []
            for label, answer in draft.choices.model_dump().items():
                try:
                    if _equivalent(_parse_answer(answer), computed):
                        matching_labels.append(label)
                except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
                    continue
        else:
            matching_labels = []
            for answer in draft.accepted_answers:
                try:
                    if _equivalent(_parse_answer(answer), computed):
                        matching_labels.append(answer)
                except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
                    continue

        expected = draft.correct_answer if draft.response_type == ResponseType.multiple_choice else None
        if draft.response_type == ResponseType.student_produced_response:
            matches = len(matching_labels) >= 1
        else:
            matches = len(matching_labels) == 1 and matching_labels[0] == expected
        return VerificationResult(
            verified=matches,
            computed_answer=_format_answer(computed),
            notes=(
                f"Computed result matches {expected or 'an accepted SPR answer'}."
                if matches else f"Computed result maps to {matching_labels}, not {expected or draft.accepted_answers}."
            ),
            verification_method=verification_method,
        )
    except NotImplementedError as error:
        return VerificationResult(verified=False, notes=str(error), verification_method="llm_only")
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, sp.SympifyError) as error:
        return VerificationResult(
            verified=False,
            notes=f"Deterministic verification could not evaluate this spec: {error}",
            verification_method="llm_only",
        )
