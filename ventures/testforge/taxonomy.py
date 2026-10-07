from typing import Final


TAXONOMY: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "algebra": {
        "linear_equations_one_variable": (
            "solving linear equations", "distribution", "combining like terms",
            "equations containing fractions", "zero solutions", "one solution",
            "infinitely many solutions", "interpreting variables and constants",
        ),
        "linear_functions": (
            "function notation", "inputs and outputs", "rate of change", "slope",
            "intercept", "tables", "graphs", "equations",
            "average rate of change for linear functions", "contextual interpretation",
        ),
        "linear_equations_two_variables": (
            "Ax + By = C", "slope-intercept form", "slope", "y-intercept", "x-intercept",
            "parallel lines", "perpendicular lines", "graph/equation/table connections",
            "contextual interpretation",
        ),
        "systems_linear_equations": (
            "substitution", "elimination", "graphical intersection", "one solution",
            "zero solutions", "infinitely many solutions",
            "parameter values producing specific solution counts", "contextual systems",
        ),
        "linear_inequalities": (
            "single-variable inequalities", "reversing inequality after division by negative",
            "compound reasoning", "inequalities in two variables", "boundary lines",
            "solution regions", "systems of inequalities", "viable coordinate points",
        ),
    },
    "advanced_math": {
        "equivalent_expressions": (
            "polynomial factoring", "polynomial expansion", "polynomial operations",
            "difference of squares", "quadratic factoring", "rational expressions",
            "rational exponents", "radical forms", "structural substitution",
        ),
        "nonlinear_equations_and_systems": (
            "quadratics", "factoring", "quadratic formula", "discriminant",
            "number of real solutions", "absolute value equations", "radical equations",
            "extraneous solutions", "rational equations", "polynomial equations",
            "linear/quadratic systems", "nonlinear systems", "real-number solutions only",
        ),
        "nonlinear_functions": (
            "quadratics", "vertex", "maximum/minimum", "transformations",
            "exponential growth", "exponential decay", "polynomial functions",
            "rational functions", "absolute value functions", "graphs", "tables",
            "equations", "contextual models",
        ),
    },
    "problem_solving_and_data_analysis": {
        "ratios_rates_units": (
            "ratios", "proportional relationships", "unit rates", "derived units",
            "unit conversions", "scale drawings", "density", "multistep proportions",
        ),
        "percentages": (
            "percent increase", "percent decrease", "consecutive percent changes",
            "discounts", "taxes", "tips", "simple interest", "growth factors",
        ),
        "one_variable_data": (
            "mean", "median", "range", "distributions", "outliers",
            "standard deviation comparisons", "frequency tables", "dot plots",
            "histograms", "box plots", "comparing distributions",
        ),
        "two_variable_data": (
            "scatterplots", "positive/negative association", "linear models",
            "quadratic models", "exponential models", "line of best fit", "predictions",
            "residual-style reasoning", "interpreting model slope/intercept",
        ),
        "probability_conditional_probability": (
            "basic probability", "conditional probability", "two-way tables",
            "area models", "correct denominator selection",
        ),
        "inference_margin_error": (
            "sample mean", "sample proportion", "population estimate", "sample size",
            "margin of error", "effects of increasing sample size",
        ),
        "statistical_claims": (
            "random samples", "random assignment", "observational studies", "experiments",
            "population generalization", "association vs causation",
        ),
    },
    "geometry_and_trigonometry": {
        "area_volume": (
            "perimeter", "area", "surface area", "volume", "composite figures",
            "scale factors", "area scaling", "volume scaling", "prisms", "cylinders",
            "cones", "pyramids", "spheres", "sector area",
        ),
        "lines_angles_triangles": (
            "parallel lines", "transversals", "vertical angles", "supplementary angles",
            "triangle angle sum", "exterior angle theorem", "congruence", "similarity",
            "corresponding sides", "scale factors",
        ),
        "right_triangles_trigonometry": (
            "Pythagorean theorem", "sine", "cosine", "tangent", "special right triangles",
            "30-60-90", "45-45-90", "complementary trig relationships",
            "degree/radian reasoning",
        ),
        "circles": (
            "radius", "diameter", "tangent", "chord", "arcs", "central angles",
            "inscribed angles", "sector area", "arc length", "degree/radian conversion",
            "circle equations", "completing the square", "coordinate geometry of circles",
        ),
    },
}


DOMAIN_LABELS: Final[dict[str, str]] = {
    "algebra": "Algebra",
    "advanced_math": "Advanced Math",
    "problem_solving_and_data_analysis": "Problem-Solving and Data Analysis",
    "geometry_and_trigonometry": "Geometry and Trigonometry",
}


def get_skills(domain: str) -> dict[str, tuple[str, ...]]:
    try:
        return TAXONOMY[domain]
    except KeyError as error:
        raise ValueError(f"Unknown SAT Math domain: {domain}") from error
