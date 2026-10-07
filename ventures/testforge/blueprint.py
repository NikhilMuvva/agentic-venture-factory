from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from models import ContextType, Difficulty, QuestionWorkOrder, ResponseType, VisualType


class BlueprintConfig(BaseModel):
    total_math_questions: int = 44
    domain_targets: dict[str, tuple[int, int]] = {
        "algebra": (13, 15),
        "advanced_math": (13, 15),
        "problem_solving_and_data_analysis": (5, 7),
        "geometry_and_trigonometry": (5, 7),
    }
    multiple_choice_fraction: float = Field(default=0.75, ge=0, le=1)
    student_produced_response_fraction: float = Field(default=0.25, ge=0, le=1)
    contextual_fraction: float = Field(default=0.30, ge=0, le=1)

    @model_validator(mode="after")
    def validate_response_mix(self) -> BlueprintConfig:
        if abs(self.multiple_choice_fraction + self.student_produced_response_fraction - 1) > 0.01:
            raise ValueError("Response-type fractions should sum to 1.")
        return self


FULL_TEST_BLUEPRINT = BlueprintConfig()


VISUAL_SMOKE_TEST: tuple[QuestionWorkOrder, ...] = (
    QuestionWorkOrder(
        id="SMOKE-001", domain="algebra", skill="linear_functions", concept="graphs",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.xy_graph, context_type=ContextType.real_world,
    ),
    QuestionWorkOrder(
        id="SMOKE-002", domain="algebra", skill="systems_linear_equations", concept="graphical intersection",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.xy_graph, context_type=ContextType.abstract,
    ),
    QuestionWorkOrder(
        id="SMOKE-003", domain="algebra", skill="linear_inequalities", concept="solution regions",
        difficulty=Difficulty.hard, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.xy_graph, context_type=ContextType.abstract,
    ),
    QuestionWorkOrder(
        id="SMOKE-004", domain="advanced_math", skill="nonlinear_functions", concept="graphs",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.xy_graph, context_type=ContextType.science,
    ),
    QuestionWorkOrder(
        id="SMOKE-005", domain="problem_solving_and_data_analysis", skill="one_variable_data", concept="dot plots",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.dot_plot, context_type=ContextType.abstract,
    ),
    QuestionWorkOrder(
        id="SMOKE-006", domain="problem_solving_and_data_analysis", skill="two_variable_data", concept="scatterplots",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.scatterplot, context_type=ContextType.social_science,
    ),
    QuestionWorkOrder(
        id="SMOKE-007", domain="problem_solving_and_data_analysis", skill="probability_conditional_probability", concept="two-way tables",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.table, context_type=ContextType.abstract,
    ),
    QuestionWorkOrder(
        id="SMOKE-008", domain="geometry_and_trigonometry", skill="lines_angles_triangles", concept="triangle angle sum",
        difficulty=Difficulty.medium, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.geometry, context_type=ContextType.abstract,
    ),
    QuestionWorkOrder(
        id="SMOKE-009", domain="geometry_and_trigonometry", skill="right_triangles_trigonometry", concept="Pythagorean theorem",
        difficulty=Difficulty.medium, response_type=ResponseType.student_produced_response,
        visual_type=VisualType.geometry, context_type=ContextType.real_world,
    ),
    QuestionWorkOrder(
        id="SMOKE-010", domain="geometry_and_trigonometry", skill="circles", concept="central angles",
        difficulty=Difficulty.hard, response_type=ResponseType.multiple_choice,
        visual_type=VisualType.circle, context_type=ContextType.abstract,
    ),
)
