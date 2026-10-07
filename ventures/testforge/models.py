from __future__ import annotations

import math
from collections import Counter
from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator
from pydantic import create_model
from taxonomy import TAXONOMY


class ResponseType(str, Enum):
    multiple_choice = "multiple_choice"
    student_produced_response = "student_produced_response"


class ContextType(str, Enum):
    abstract = "abstract"
    real_world = "real_world"
    science = "science"
    social_science = "social_science"


class VisualType(str, Enum):
    none = "none"
    table = "table"
    xy_graph = "xy_graph"
    scatterplot = "scatterplot"
    bar_chart = "bar_chart"
    histogram = "histogram"
    dot_plot = "dot_plot"
    box_plot = "box_plot"
    geometry = "geometry"
    circle = "circle"
    solid_3d = "solid_3d"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class AnswerChoices(BaseModel):
    A: str
    B: str
    C: str
    D: str

    @model_validator(mode="after")
    def choices_are_nonempty_and_distinct(self) -> AnswerChoices:
        values = [value.strip() for value in self.model_dump().values()]
        if any(not value for value in values):
            raise ValueError("Answer choices must not be empty.")
        if len(set(values)) != 4:
            raise ValueError("Multiple-choice answers must be distinct.")
        return self


Point2D: TypeAlias = Annotated[list[float], Field(min_length=2, max_length=2)]


def _validate_finite(values: list[float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Visual coordinates and values must be finite.")


class TableVisual(BaseModel):
    visual_type: Literal["table"] = "table"
    title: str | None = None
    column_headers: list[str] = Field(min_length=1)
    rows: list[list[str | float | int]] = Field(min_length=1)

    @model_validator(mode="after")
    def rows_match_headers(self) -> TableVisual:
        if any(len(row) != len(self.column_headers) for row in self.rows):
            raise ValueError("Every table row must match the header count.")
        for row in self.rows:
            _validate_finite([float(value) for value in row if isinstance(value, (int, float))])
        return self


class GraphSeries(BaseModel):
    kind: Literal["line", "quadratic", "exponential", "points", "inequality_boundary"]
    label: str | None = None
    slope: float | None = None
    intercept: float | None = None
    a: float | None = None
    b: float | None = None
    c: float | None = None
    initial_value: float | None = None
    base: float | None = None
    points: list[Point2D] = Field(default_factory=list)
    relation: Literal["<", "<=", ">", ">="] | None = None

    @model_validator(mode="after")
    def validate_series_parameters(self) -> GraphSeries:
        numeric_values = [
            value for value in (
                self.slope, self.intercept, self.a, self.b, self.c,
                self.initial_value, self.base,
            ) if value is not None
        ]
        _validate_finite(numeric_values)
        for point in self.points:
            _validate_finite(list(point))
        if self.kind in ("line", "inequality_boundary"):
            if self.slope is None or self.intercept is None:
                raise ValueError("Line series require slope and intercept.")
        if self.kind == "quadratic" and None in (self.a, self.b, self.c):
            raise ValueError("Quadratic series require coefficients a, b, and c.")
        if self.kind == "exponential":
            if self.initial_value is None or self.base is None or self.base <= 0:
                raise ValueError("Exponential series require an initial value and positive base.")
        if self.kind == "points" and not self.points:
            raise ValueError("Point series require at least one coordinate.")
        if self.kind == "inequality_boundary" and self.relation is None:
            raise ValueError("Inequality boundaries require a relation.")
        return self


class XYGraphVisual(BaseModel):
    visual_type: Literal["xy_graph"] = "xy_graph"
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_label: str | None = None
    y_label: str | None = None
    grid: bool = True
    series: list[GraphSeries] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_axes(self) -> XYGraphVisual:
        _validate_finite([self.x_min, self.x_max, self.y_min, self.y_max])
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Graph axis minima must be less than maxima.")
        for item in self.series:
            if item.kind == "points":
                for x_value, y_value in item.points:
                    if not self.x_min <= x_value <= self.x_max or not self.y_min <= y_value <= self.y_max:
                        raise ValueError("Graph points must lie within the configured axes.")
        return self


class BestFitLine(BaseModel):
    slope: float
    intercept: float

    @model_validator(mode="after")
    def finite_coefficients(self) -> BestFitLine:
        _validate_finite([self.slope, self.intercept])
        return self


class ScatterPlotVisual(BaseModel):
    visual_type: Literal["scatterplot"] = "scatterplot"
    points: list[Point2D] = Field(min_length=1)
    best_fit_line: BestFitLine | None = None
    x_label: str = "x"
    y_label: str = "y"

    @model_validator(mode="after")
    def finite_points(self) -> ScatterPlotVisual:
        for point in self.points:
            _validate_finite(list(point))
        return self


class BarChartVisual(BaseModel):
    visual_type: Literal["bar_chart"] = "bar_chart"
    labels: list[str] = Field(min_length=1)
    values: list[float] = Field(min_length=1)
    x_label: str = "Category"
    y_label: str = "Value"

    @model_validator(mode="after")
    def matching_values(self) -> BarChartVisual:
        if len(self.labels) != len(self.values):
            raise ValueError("Bar chart labels and values must have equal lengths.")
        _validate_finite(self.values)
        return self


class HistogramVisual(BaseModel):
    visual_type: Literal["histogram"] = "histogram"
    bin_edges: list[float] = Field(min_length=2)
    frequencies: list[int] = Field(min_length=1)
    x_label: str = "Value"
    y_label: str = "Frequency"

    @model_validator(mode="after")
    def increasing_bins(self) -> HistogramVisual:
        _validate_finite(self.bin_edges)
        if len(self.bin_edges) != len(self.frequencies) + 1:
            raise ValueError("Histogram requires one more bin edge than frequency.")
        if any(right <= left for left, right in zip(self.bin_edges, self.bin_edges[1:])):
            raise ValueError("Histogram bin edges must strictly increase.")
        if any(value < 0 for value in self.frequencies):
            raise ValueError("Histogram frequencies cannot be negative.")
        return self


class ValueFrequency(BaseModel):
    value: float
    frequency: int

    @model_validator(mode="after")
    def positive_frequency(self) -> ValueFrequency:
        _validate_finite([self.value])
        if self.frequency <= 0:
            raise ValueError("Dot-plot frequency must be positive.")
        return self


class DotPlotVisual(BaseModel):
    visual_type: Literal["dot_plot"] = "dot_plot"
    values: list[float] = Field(default_factory=list)
    value_frequencies: list[ValueFrequency] = Field(default_factory=list)
    x_label: str = "Value"

    @model_validator(mode="after")
    def has_dot_values(self) -> DotPlotVisual:
        if self.values and self.value_frequencies:
            expanded_frequencies = [
                item.value for item in self.value_frequencies for _ in range(item.frequency)
            ]
            if Counter(self.values) != Counter(expanded_frequencies):
                raise ValueError("Dot-plot values and value_frequencies must match exactly when both are provided.")
            self.values = []
        elif not self.values and not self.value_frequencies:
            raise ValueError("A dot plot requires values or value_frequencies.")
        _validate_finite(self.values + [item.value for item in self.value_frequencies])
        return self


class BoxPlotVisual(BaseModel):
    visual_type: Literal["box_plot"] = "box_plot"
    min: float
    q1: float
    median: float
    q3: float
    max: float
    x_label: str = "Value"

    @model_validator(mode="after")
    def ordered_quartiles(self) -> BoxPlotVisual:
        values = [self.min, self.q1, self.median, self.q3, self.max]
        _validate_finite(values)
        if values != sorted(values):
            raise ValueError("Box plot values must satisfy min <= q1 <= median <= q3 <= max.")
        return self


class GeometryPoint(BaseModel):
    kind: Literal["point"] = "point"
    x: float
    y: float
    name: str | None = None


class Segment(BaseModel):
    kind: Literal["segment"] = "segment"
    start: Point2D
    end: Point2D
    label: str | None = None


class Polygon(BaseModel):
    kind: Literal["polygon"] = "polygon"
    vertices: list[Point2D] = Field(min_length=3)
    label: str | None = None


class CirclePrimitive(BaseModel):
    kind: Literal["circle"] = "circle"
    center: Point2D
    radius: float
    label: str | None = None

    @model_validator(mode="after")
    def positive_radius(self) -> CirclePrimitive:
        _validate_finite([*self.center, self.radius])
        if self.radius <= 0:
            raise ValueError("Circle radius must be positive.")
        return self


class Arc(BaseModel):
    kind: Literal["arc"] = "arc"
    center: Point2D
    radius: float
    start_degrees: float
    end_degrees: float
    label: str | None = None

    @model_validator(mode="after")
    def positive_radius(self) -> Arc:
        _validate_finite([*self.center, self.radius, self.start_degrees, self.end_degrees])
        if self.radius <= 0:
            raise ValueError("Arc radius must be positive.")
        return self


class AngleMarker(BaseModel):
    kind: Literal["angle"] = "angle"
    vertex: Point2D
    ray1_point: Point2D
    ray2_point: Point2D
    label: str | None = None

    @model_validator(mode="after")
    def rays_have_length(self) -> AngleMarker:
        for point in (self.vertex, self.ray1_point, self.ray2_point):
            _validate_finite(list(point))
        if self.vertex == self.ray1_point or self.vertex == self.ray2_point:
            raise ValueError("Angle-marker rays must have nonzero length.")
        return self


class Label(BaseModel):
    kind: Literal["label"] = "label"
    position: Point2D
    text: str


class RightAngleMarker(BaseModel):
    kind: Literal["right_angle"] = "right_angle"
    vertex: Point2D
    along_first_ray: Point2D
    along_second_ray: Point2D
    size: float = 0.2

    @model_validator(mode="after")
    def rays_have_length(self) -> RightAngleMarker:
        _validate_finite([*self.vertex, *self.along_first_ray, *self.along_second_ray, self.size])
        if self.size <= 0:
            raise ValueError("Right-angle marker size must be positive.")
        if self.vertex == self.along_first_ray or self.vertex == self.along_second_ray:
            raise ValueError("Right-angle marker rays must have nonzero length.")
        return self


class GeometryPrimitive(BaseModel):
    kind: Literal["point", "segment", "polygon", "circle", "arc", "angle", "label", "right_angle"]
    x: float | None = None
    y: float | None = None
    name: str | None = None
    start: Point2D | None = None
    end: Point2D | None = None
    vertices: list[Point2D] = Field(default_factory=list)
    center: Point2D | None = None
    radius: float | None = None
    start_degrees: float | None = None
    end_degrees: float | None = None
    vertex: Point2D | None = None
    ray1_point: Point2D | None = None
    ray2_point: Point2D | None = None
    position: Point2D | None = None
    text: str | None = None
    along_first_ray: Point2D | None = None
    along_second_ray: Point2D | None = None
    size: float | None = 0.2
    label: str | None = None

    @model_validator(mode="after")
    def validate_kind_fields(self) -> GeometryPrimitive:
        required = {
            "point": (self.x is not None and self.y is not None),
            "segment": (self.start is not None and self.end is not None),
            "polygon": (len(self.vertices) >= 3),
            "circle": (self.center is not None and self.radius is not None),
            "arc": (self.center is not None and self.radius is not None and self.start_degrees is not None and self.end_degrees is not None),
            "angle": (self.vertex is not None and self.ray1_point is not None and self.ray2_point is not None),
            "label": (self.position is not None and self.text is not None),
            "right_angle": (self.vertex is not None and self.along_first_ray is not None and self.along_second_ray is not None),
        }
        if not required[self.kind]:
            raise ValueError(f"Geometry primitive '{self.kind}' is missing required coordinates or labels.")
        if self.kind in ("circle", "arc") and self.radius is not None and self.radius <= 0:
            raise ValueError("Geometry circle/arc radius must be positive.")
        if self.kind == "right_angle" and self.size is not None and self.size <= 0:
            raise ValueError("Right-angle marker size must be positive.")
        for value in self.model_dump().values():
            if isinstance(value, (int, float)):
                _validate_finite([float(value)])
            elif isinstance(value, (list, tuple)):
                if all(isinstance(item, (int, float)) for item in value):
                    _validate_finite([float(item) for item in value])
                else:
                    for point in value:
                        if isinstance(point, (list, tuple)) and all(isinstance(item, (int, float)) for item in point):
                            _validate_finite([float(item) for item in point])
        if self.kind in ("angle", "right_angle") and self.vertex is not None:
            rays = (self.ray1_point, self.ray2_point) if self.kind == "angle" else (self.along_first_ray, self.along_second_ray)
            if any(point == self.vertex for point in rays):
                raise ValueError("Angle-marker rays must have nonzero length.")
        return self


class GeometryVisual(BaseModel):
    visual_type: Literal["geometry"] = "geometry"
    primitives: list[GeometryPrimitive] = Field(min_length=1)
    to_scale: bool = False

    @model_validator(mode="after")
    def finite_geometry(self) -> GeometryVisual:
        for primitive in self.primitives:
            for value in primitive.model_dump().values():
                if isinstance(value, (int, float)):
                    _validate_finite([float(value)])
                elif isinstance(value, (list, tuple)):
                    if all(isinstance(item, (int, float)) for item in value):
                        _validate_finite([float(item) for item in value])
                    else:
                        for point in value:
                            if isinstance(point, (list, tuple)) and all(isinstance(item, (int, float)) for item in point):
                                _validate_finite([float(item) for item in point])
        return self


class NamedPoint(BaseModel):
    name: str
    x: float
    y: float


class CircleVisual(BaseModel):
    visual_type: Literal["circle"] = "circle"
    center: Point2D
    radius: float
    equation: str | None = None
    points: list[NamedPoint] = Field(default_factory=list)
    radii: list[Segment] = Field(default_factory=list)
    chords: list[Segment] = Field(default_factory=list)
    tangents: list[Segment] = Field(default_factory=list)
    arcs: list[Arc] = Field(default_factory=list)
    angle_markers: list[AngleMarker] = Field(default_factory=list)
    to_scale: bool = False

    @model_validator(mode="after")
    def valid_circle_data(self) -> CircleVisual:
        _validate_finite([*self.center, self.radius])
        if self.radius <= 0:
            raise ValueError("Circle radius must be positive.")
        for point in self.points:
            _validate_finite([point.x, point.y])
        for segment in [*self.radii, *self.chords, *self.tangents]:
            _validate_finite([*segment.start, *segment.end])
        for arc in self.arcs:
            _validate_finite([*arc.center, arc.radius, arc.start_degrees, arc.end_degrees])
        for marker in self.angle_markers:
            _validate_finite([*marker.vertex, *marker.ray1_point, *marker.ray2_point])
        return self


class Solid3DVisual(BaseModel):
    visual_type: Literal["solid_3d"] = "solid_3d"
    solid_type: Literal["rectangular_prism", "cylinder", "cone", "pyramid", "sphere"]
    width: float | None = None
    depth: float | None = None
    height: float | None = None
    radius: float | None = None
    label: str | None = None

    @model_validator(mode="after")
    def positive_dimensions(self) -> Solid3DVisual:
        dimensions = [value for value in (self.width, self.depth, self.height, self.radius) if value is not None]
        _validate_finite(dimensions)
        if any(value <= 0 for value in dimensions):
            raise ValueError("Solid dimensions must be positive when supplied.")
        required_dimensions = {
            "rectangular_prism": (self.width, self.depth, self.height),
            "cylinder": (self.radius, self.height),
            "cone": (self.radius, self.height),
            "pyramid": (self.width, self.depth, self.height),
            "sphere": (self.radius,),
        }[self.solid_type]
        if any(value is None for value in required_dimensions):
            raise ValueError(f"{self.solid_type} visual is missing a required dimension.")
        return self


VisualSpec: TypeAlias = Annotated[
    TableVisual | XYGraphVisual | ScatterPlotVisual | BarChartVisual | HistogramVisual |
    DotPlotVisual | BoxPlotVisual | GeometryVisual | CircleVisual | Solid3DVisual,
    Field(discriminator="visual_type"),
]


class QuestionWorkOrder(BaseModel):
    id: str
    domain: Literal[
        "algebra",
        "advanced_math",
        "problem_solving_and_data_analysis",
        "geometry_and_trigonometry",
    ]
    skill: str
    concept: str
    difficulty: Difficulty
    response_type: ResponseType
    visual_type: VisualType
    context_type: ContextType

    @model_validator(mode="after")
    def validate_taxonomy(self) -> QuestionWorkOrder:
        domain_skills = TAXONOMY.get(self.domain, {})
        if self.skill not in domain_skills:
            raise ValueError(f"Unknown skill '{self.skill}' for domain '{self.domain}'.")
        if self.concept not in domain_skills[self.skill]:
            raise ValueError(f"Unknown concept '{self.concept}' for skill '{self.skill}'.")
        return self


class VariableValue(BaseModel):
    name: str
    value: float


class ChoicePoint(BaseModel):
    label: Literal["A", "B", "C", "D"]
    x: float
    y: float


class MathVerificationSpec(BaseModel):
    method: Literal[
        "graph_line", "linear_equation", "triangle_angle", "function_evaluation", "linear_system", "quadratic",
        "inequality", "expression_evaluation", "ratio", "percentage",
        "mean", "median", "range", "probability", "formula", "llm_only",
    ]
    expression: str | None = None
    equations: list[str] = Field(default_factory=list)
    variable: str = "x"
    answer_expression: str | None = None
    variables: list[VariableValue] = Field(default_factory=list)
    operation: Literal["value", "solve", "sum_roots", "roots", "discriminant"] = "value"
    numerator: float | None = None
    denominator: float | None = None
    values: list[float] = Field(default_factory=list)
    choice_points: list[ChoicePoint] = Field(default_factory=list)


class QuestionDraft(BaseModel):
    domain: Literal[
        "algebra",
        "advanced_math",
        "problem_solving_and_data_analysis",
        "geometry_and_trigonometry",
    ]
    stem: str
    response_type: ResponseType
    choices: AnswerChoices | None = None
    correct_answer: Literal["A", "B", "C", "D"] | None = None
    accepted_answers: list[str] = Field(default_factory=list)
    explanation: str
    visual_spec: VisualSpec | None = None
    skill: str
    concept: str
    difficulty: Difficulty
    context_type: ContextType
    visual_type: VisualType
    verification_spec: MathVerificationSpec

    @model_validator(mode="after")
    def validate_response_and_visual(self) -> QuestionDraft:
        domain_skills = TAXONOMY.get(self.domain, {})
        if self.skill not in domain_skills:
            returned_words = self.skill.lower().replace("-", " ").split()
            matching_skills = [
                skill_id for skill_id in domain_skills
                if skill_id.replace("_", " ").split()[-len(returned_words):] == returned_words
            ] if returned_words else []
            if len(matching_skills) == 1:
                self.skill = matching_skills[0]
        if self.skill not in domain_skills or self.concept not in domain_skills[self.skill]:
            raise ValueError("Question skill and concept must match the central taxonomy.")
        if self.response_type == ResponseType.multiple_choice:
            if self.choices is None or self.correct_answer is None:
                raise ValueError("Multiple-choice drafts require choices and correct_answer.")
            if self.accepted_answers:
                raise ValueError("Multiple-choice drafts must not include accepted_answers.")
        else:
            if self.choices is not None or self.correct_answer is not None:
                raise ValueError("Student-produced responses must not have choices or a choice answer.")
            if not self.accepted_answers:
                raise ValueError("Student-produced responses require accepted numeric answers.")
        actual_visual = self.visual_spec.visual_type if self.visual_spec else VisualType.none.value
        expected_visual = self.visual_type.value
        if actual_visual != expected_visual:
            raise ValueError("visual_type must match the supplied VisualSpec.")
        return self


VISUAL_MODEL_BY_TYPE = {
    VisualType.none: None,
    VisualType.table: TableVisual,
    VisualType.xy_graph: XYGraphVisual,
    VisualType.scatterplot: ScatterPlotVisual,
    VisualType.bar_chart: BarChartVisual,
    VisualType.histogram: HistogramVisual,
    VisualType.dot_plot: DotPlotVisual,
    VisualType.box_plot: BoxPlotVisual,
    VisualType.geometry: GeometryVisual,
    VisualType.circle: CircleVisual,
    VisualType.solid_3d: Solid3DVisual,
}


def question_draft_response_model(visual_type: VisualType) -> type[QuestionDraft]:
    visual_model = VISUAL_MODEL_BY_TYPE[visual_type]
    visual_field = (None, ...) if visual_model is None else (visual_model, ...)
    return create_model(
        f"GeneratedQuestionDraft{visual_type.value.title().replace('_', '')}",
        __base__=QuestionDraft,
        visual_spec=visual_field,
    )


class StudentSolverResult(BaseModel):
    selected_choice: Literal["A", "B", "C", "D"] | None = None
    numeric_answer: str | None = None
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class VisualReviewResult(BaseModel):
    pass_review: bool
    score: int = Field(ge=0, le=100)
    mathematical_validity: int = Field(ge=0, le=100)
    skill_alignment_score: int = Field(ge=0, le=100)
    clarity: int = Field(ge=0, le=100)
    distractor_quality: int = Field(ge=0, le=100)
    difficulty_accuracy: int = Field(ge=0, le=100)
    difficulty_matches: bool
    visual_quality_score: int | None = Field(default=None, ge=0, le=100)
    visual_matches_question: bool
    originality_risk: Literal["low", "medium", "high"]
    issues: list[str]
    reviewer_notes: str


class VerificationResult(BaseModel):
    verified: bool
    computed_answer: str | None = None
    notes: str
    verification_method: str


class APIInfrastructureFailure(RuntimeError):
    def __init__(
        self,
        reason: str,
        message: str,
        stage: str,
        model: str | None = None,
        status_code: int | None = None,
        classification: str | None = None,
    ) -> None:
        super().__init__(message)
        self.infrastructure_failure = True
        self.reason = reason
        self.stage = stage
        self.model = model
        self.status_code = status_code
        self.classification = classification
        self.candidate_record: dict | None = None
