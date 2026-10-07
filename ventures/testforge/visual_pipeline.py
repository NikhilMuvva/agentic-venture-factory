from __future__ import annotations

import json
import re
import numpy as np
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from google.genai import types
from PIL import Image
from pydantic import BaseModel, TypeAdapter, ValidationError

from blueprint import VISUAL_SMOKE_TEST
from models import (
    QuestionDraft,
    QuestionWorkOrder,
    APIInfrastructureFailure,
    ResponseType,
    StudentSolverResult,
    VerificationResult,
    VisualReviewResult,
    VisualSpec,
    VisualType,
    question_draft_response_model,
)
from renderer import render_visual
from renderer import student_visible_visual_description
from storage import (
    APPROVED_DIR,
    REJECTED_DIR,
    RUNS_DIR,
    VISUALS_DIR,
    ensure_pipeline_directories,
    save_json_record,
)
from taxonomy import DOMAIN_LABELS, TAXONOMY
from verifier import inequality_point_state, numeric_answers_equivalent, verify_draft
from models import XYGraphVisual


StructuredCaller = Callable[[Any, type[BaseModel], str, str, list[dict]], tuple[BaseModel, str]]
MAX_TOTAL_ATTEMPTS = 60
SUPPORTED_SMOKE_SKILLS = {work_order.skill for work_order in VISUAL_SMOKE_TEST}
VERIFICATION_GUIDANCE = {
    "linear_functions": "Ask which equation matches the single graphed line. Use method graph_line; the VisualSpec supplies its slope and intercept.",
    "systems_linear_equations": "Graph exactly two lines and ask for their intersection as an ordered pair. Use method linear_system and put both line equations in equations.",
    "linear_inequalities": "Use one strict inequality in the form y < m*x+b with one dashed line boundary and the region below shaded. Ask which listed coordinate satisfies it. Use method inequality, set expression to the exact y < m*x+b relation, and include all four labeled coordinates in choice_points. Use integer points not on the boundary. Substitute each point explicitly: exactly the claimed correct choice must satisfy the inequality and the other three must not.",
    "nonlinear_functions": "Graph one quadratic and ask for its value at a specified x. Use method function_evaluation, expression for the quadratic, and variables with that x value.",
    "one_variable_data": "Ask for the median of the dot-plot data. Use method median and list every plotted observation in values, repeating values according to dot frequency.",
    "two_variable_data": "Include a scatterplot with best_fit_line and ask for the model estimate at a specified x. Use method function_evaluation, expression m*x+b from the best-fit line, and variables with the requested x.",
    "probability_conditional_probability": "Use a complete two-way table and ask a single conditional probability. Use method probability with the exact favorable count as numerator and the requested condition total as denominator.",
    "lines_angles_triangles": "For medium difficulty, label all three triangle angles with linear expressions in x, give one angle-sum equation, and ask for the measure of an angle expression that requires solving for x first. Use method triangle_angle; put the full sum-to-180 equation in equations, variable x, and the requested angle expression in answer_expression. Label the same expressions in the geometry visual.",
    "right_triangles_trigonometry": "For medium SPR difficulty, give the two legs of a right triangle and ask for its perimeter. This requires finding the hypotenuse with the Pythagorean theorem and then adding all three sides. Use method formula with an exact sqrt(a^2+b^2)+a+b expression and include equivalent numeric SPR answers.",
    "circles": "For hard difficulty, combine an inscribed-angle relationship with sector area: give the radius and an inscribed angle, ask for the area of the sector intercepted by the same arc, and require using twice the inscribed angle as the central angle. Use method formula with the exact pi*r^2*(2*inscribed_angle/360) expression. Include a readable angle marker and radius label.",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _similarity(stem: str, approved_stems: list[str]) -> float:
    normalized = _normalize(stem)
    return max((SequenceMatcher(None, normalized, _normalize(item)).ratio() for item in approved_stems), default=0.0)


def audit_existing_approvals() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    ensure_pipeline_directories()
    work_orders = {item.id: item for item in VISUAL_SMOKE_TEST}
    valid: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, str]] = []

    for path in sorted(APPROVED_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            work_order = QuestionWorkOrder.model_validate(record["work_order"])
            expected = work_orders.get(work_order.id)
            if expected is None:
                continue
            if work_order.model_dump(mode="json") != expected.model_dump(mode="json"):
                raise ValueError("Stored work order differs from the current smoke blueprint.")
            question = QuestionDraft.model_validate(record["question"])
            solver = StudentSolverResult.model_validate(record["solver_output"])
            verification = VerificationResult.model_validate(record["deterministic_verifier_output"])
            review = VisualReviewResult.model_validate(record["reviewer_output"])
            visual = TypeAdapter(VisualSpec).validate_python(record["visual_spec"])
            if not _matches_work_order(question, expected):
                raise ValueError("Stored question does not match its work order.")
            if not _solver_matches(question, solver):
                raise ValueError("Stored Solver output does not match the question answer.")
            if not verification.verified or verification.verification_method == "llm_only":
                raise ValueError("Stored deterministic verification is not valid.")
            if _rejection_reasons(question, solver, verification, review, 0.0):
                raise ValueError("Stored Reviewer output no longer satisfies approval thresholds.")
            actual_visual = getattr(visual.visual_type, "value", visual.visual_type)
            if actual_visual != expected.visual_type.value:
                raise ValueError("Stored VisualSpec type differs from the work order.")
            visual_path = Path(record["visual_image_path"])
            if not visual_path.is_absolute():
                visual_path = Path(__file__).resolve().parent / visual_path
            if not visual_path.is_file() or visual_path.stat().st_size <= 1000:
                raise ValueError("Stored visual PNG is missing or too small.")
            pixels = np.asarray(Image.open(visual_path).convert("RGB"))
            if pixels.size == 0 or float(pixels.std()) <= 5:
                raise ValueError("Stored visual PNG is blank.")
            valid.setdefault(work_order.id, record)
        except (KeyError, OSError, TypeError, ValueError, ValidationError) as error:
            invalid.append({"file": path.name, "reason": str(error)[:240]})

    return valid, invalid


def print_resume_status(
    valid_approvals: dict[str, dict[str, Any]],
    fresh: bool = False,
) -> list[QuestionWorkOrder]:
    completed = set() if fresh else set(valid_approvals)
    remaining = [work for work in VISUAL_SMOKE_TEST if work.id not in completed]
    print("=" * 55)
    print("TESTFORGE")
    print("Fresh Mode" if fresh else "Resume Mode")
    print("=" * 55)
    print(f"Valid completed work orders: {len(completed)}/{len(VISUAL_SMOKE_TEST)}")
    print(f"Existing valid approvals: {len(valid_approvals)}/{len(VISUAL_SMOKE_TEST)}")
    print("Remaining work orders:")
    if remaining:
        for work in remaining:
            response_label = "SPR" if work.response_type == ResponseType.student_produced_response else "MCQ"
            print(f"- {work.id} - {work.skill} - {work.visual_type.value} - {response_label}")
    else:
        print("- none")
    return remaining


def _student_prompt(draft: QuestionDraft) -> str:
    choices = ""
    if draft.response_type == ResponseType.multiple_choice:
        choices = "\nAnswer choices:\n" + "\n".join(
            f"{label}. {value}" for label, value in draft.choices.model_dump().items()
        )
    image_notice = ""
    if draft.visual_spec:
        image_notice = "\nA diagram or graph is attached. Use the visual as part of the question."
    response_instruction = (
        "Choose one option A, B, C, or D." if draft.response_type == ResponseType.multiple_choice
        else "Give the numeric answer; equivalent exact fraction/decimal forms are acceptable."
    )
    return f"""You are an independent SAT-style math student solver.
Solve the problem from scratch using only what a student can see.
Do not assume the generator's answer is correct. Check the mathematics carefully.
{response_instruction}

Question:
{draft.stem}{choices}{image_notice}

Return your selected_choice for multiple choice, or numeric_answer for student-produced response, plus concise reasoning and confidence.
"""


def _review_prompt(
    work_order: QuestionWorkOrder,
    draft: QuestionDraft,
    solver: StudentSolverResult,
    verification: VerificationResult,
) -> str:
    return f"""You are a strict final QA reviewer for original SAT-style math practice.

Evaluate:
- mathematical correctness and exactly one correct response
- alignment to the requested domain, skill, and concept
- concise, unambiguous, self-contained wording
- distractor quality for MCQ; appropriate numeric response for SPR
- whether the visual is correct, useful, readable, and matches the stem
- difficulty accuracy
- originality risk; do not approve anything resembling official test wording

Difficulty rubric:
EASY: direct one-step or straightforward multi-step application, little interpretation, obvious setup, minimal conceptual traps.
MEDIUM: translating information into equations, multiple conceptual steps, less obvious setup, meaningful distractors.
HARD: combining concepts, non-obvious setup, careful reasoning, multiple plausible approaches or traps.
If the claimed difficulty is substantially wrong, difficulty_accuracy must be below 80 and difficulty_matches must be false.

Only set pass_review true when the overall score is at least 85, mathematical validity at least 95, skill alignment at least 90, difficulty matches, the visual matches the question, visual quality is at least 80 when a visual exists, and originality risk is not high.

Requested work order:
{work_order.model_dump_json()}

Complete question draft, including the source-of-truth visual spec:
{draft.model_dump_json(indent=2)}

Independent Solver output:
{solver.model_dump_json()}

Deterministic verifier output:
{verification.model_dump_json()}

Return only the requested structured review result. Explain any failure in issues and reviewer_notes.
"""


def _matches_work_order(draft: QuestionDraft, work_order: QuestionWorkOrder) -> bool:
    expected_visual = work_order.visual_type.value
    actual_visual = draft.visual_spec.visual_type if draft.visual_spec else VisualType.none.value
    if isinstance(actual_visual, VisualType):
        actual_visual = actual_visual.value
    return (
        draft.domain == work_order.domain
        and draft.skill == work_order.skill
        and draft.concept == work_order.concept
        and draft.difficulty == work_order.difficulty
        and draft.response_type == work_order.response_type
        and draft.visual_type == work_order.visual_type
        and actual_visual == expected_visual
    )


def _format_coordinate(value: float) -> str:
    return f"{value:g}"


def normalize_inequality_choices(draft: QuestionDraft) -> tuple[QuestionDraft, list[str]]:
    spec = draft.verification_spec
    if (
        spec.method != "inequality"
        or draft.response_type != ResponseType.multiple_choice
        or not isinstance(draft.visual_spec, XYGraphVisual)
        or not spec.expression
    ):
        return draft, []

    points = {point.label: (point.x, point.y) for point in spec.choice_points}
    labels = ("A", "B", "C", "D")
    visual = draft.visual_spec

    def state(point: tuple[float, float]) -> str:
        return inequality_point_state(spec.expression, point[0], point[1])

    satisfying_labels = [label for label in labels if label in points and state(points[label]) == "satisfies"]
    adjustments: list[str] = []
    correct_label = draft.correct_answer
    if correct_label not in satisfying_labels and satisfying_labels:
        correct_label = satisfying_labels[0]
        adjustments.append(f"Correct answer label recalculated as {correct_label} from the inequality.")

    used_points: set[tuple[float, float]] = set()

    def find_point(wanted_state: str) -> tuple[float, float]:
        samples = 31
        x_values = [visual.x_min + (visual.x_max - visual.x_min) * index / (samples - 1) for index in range(samples)]
        y_values = [visual.y_min + (visual.y_max - visual.y_min) * index / (samples - 1) for index in range(samples)]
        for x_value in x_values:
            for y_value in y_values:
                candidate = (round(x_value, 6), round(y_value, 6))
                if candidate not in used_points and state(candidate) == wanted_state:
                    return candidate
        raise ValueError(f"Could not find a distinct {wanted_state} point inside the graph axes.")

    correct_point = points.get(correct_label)
    if correct_point is None or state(correct_point) != "satisfies":
        correct_point = find_point("satisfies")
        adjustments.append(f"Choice {correct_label} coordinate generated from the rendered solution region.")

    normalized_points: dict[str, tuple[float, float]] = {correct_label: correct_point}
    used_points.add(correct_point)
    for label in labels:
        if label == correct_label:
            continue
        candidate = points.get(label)
        if candidate is None or candidate in used_points or state(candidate) != "outside":
            candidate = find_point("outside")
            adjustments.append(f"Choice {label} coordinate replaced with a verified non-solution.")
        normalized_points[label] = candidate
        used_points.add(candidate)

    option_values = {
        label: f"({_format_coordinate(normalized_points[label][0])}, {_format_coordinate(normalized_points[label][1])})"
        for label in labels
    }
    serialized = draft.model_dump(mode="python")
    serialized["choices"] = option_values
    serialized["correct_answer"] = correct_label
    serialized["explanation"] = (
        f"Substituting choice {correct_label}, {option_values[correct_label]}, into "
        f"{spec.expression} satisfies the inequality. Each other listed point is outside its solution region."
    )
    serialized["verification_spec"]["choice_points"] = [
        {"label": label, "x": normalized_points[label][0], "y": normalized_points[label][1]}
        for label in labels
    ]
    return QuestionDraft.model_validate(serialized), adjustments


def _format_reason(error: Exception, fallback: str) -> str:
    reason = getattr(error, "reason", None)
    if reason in {"API_FAILURE", "INVALID_OUTPUT"}:
        return reason
    if isinstance(error, ValidationError):
        return "INVALID_OUTPUT"
    return fallback


def _solver_matches(draft: QuestionDraft, solver: StudentSolverResult) -> bool:
    if draft.response_type == ResponseType.multiple_choice:
        return solver.selected_choice == draft.correct_answer and solver.numeric_answer is None
    return (
        solver.selected_choice is None
        and solver.numeric_answer is not None
        and numeric_answers_equivalent(solver.numeric_answer, draft.accepted_answers)
    )


def _rejection_reasons(
    draft: QuestionDraft,
    solver: StudentSolverResult,
    verification: VerificationResult,
    review: VisualReviewResult,
    similarity: float,
) -> list[str]:
    reasons = []
    if not _solver_matches(draft, solver):
        reasons.append("ANSWER_DISAGREEMENT")
    if verification.verification_method != "llm_only" and not verification.verified:
        reasons.append("DETERMINISTIC_VERIFICATION_FAILED")
    if draft.skill in SUPPORTED_SMOKE_SKILLS and verification.verification_method == "llm_only":
        reasons.append("DETERMINISTIC_VERIFICATION_UNAVAILABLE")
    if review.score < 85:
        reasons.append("REVIEW_SCORE_TOO_LOW")
    if review.mathematical_validity < 95:
        reasons.append("MATHEMATICAL_VALIDITY_TOO_LOW")
    if review.skill_alignment_score < 90:
        reasons.append("SKILL_ALIGNMENT_TOO_LOW")
    if not review.difficulty_matches or review.difficulty_accuracy < 80:
        reasons.append("DIFFICULTY_MISMATCH")
    if review.originality_risk == "high":
        reasons.append("HIGH_ORIGINALITY_RISK")
    if draft.visual_spec and (
        not review.visual_matches_question
        or review.visual_quality_score is None
        or review.visual_quality_score < 80
    ):
        reasons.append("VISUAL_REVIEW_FAILED")
    if not review.pass_review:
        reasons.append("REVIEWER_REJECTED")
    if similarity >= 0.82:
        reasons.append("TOO_SIMILAR_TO_EXISTING_QUESTION")
    return reasons


def _candidate_record(
    candidate_id: str,
    work_order: QuestionWorkOrder,
    timestamp: str,
    draft: QuestionDraft | None,
    solver: StudentSolverResult | None,
    verification: VerificationResult | None,
    review: VisualReviewResult | None,
    models_used: dict[str, str],
    visual_path: Path | None,
    api_model_failures: list[dict],
    generator_output: dict[str, Any] | None = None,
    deterministic_adjustments: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "work_order_id": work_order.id,
        "work_order": work_order.model_dump(mode="json"),
        "timestamp": timestamp,
        "question": draft.model_dump(mode="json") if draft else None,
        "generator_output": generator_output or (draft.model_dump(mode="json") if draft else None),
        "deterministic_adjustments": deterministic_adjustments or [],
        "solver_output": solver.model_dump(mode="json") if solver else None,
        "deterministic_verifier_output": verification.model_dump(mode="json") if verification else None,
        "reviewer_output": review.model_dump(mode="json") if review else None,
        "visual_spec": draft.visual_spec.model_dump(mode="json") if draft and draft.visual_spec else None,
        "visual_image_path": f"data/visuals/{visual_path.name}" if visual_path else None,
        "models_used": models_used,
        "human_approved": False,
        "training_rights_verified": False,
        "api_model_failures": [item for item in api_model_failures if item.get("candidate_id") == candidate_id],
    }


def _create_draft(
    call_structured: StructuredCaller,
    work_order: QuestionWorkOrder,
    candidate_id: str,
    api_model_failures: list[dict],
) -> tuple[QuestionDraft, str]:
    skill_concepts = TAXONOMY[work_order.domain][work_order.skill]
    verification_guidance = VERIFICATION_GUIDANCE.get(
        work_order.skill,
        "Use a supported deterministic method and include all numeric source data; choose llm_only only when deterministic calculation is not applicable.",
    )
    prompt = f"""Create exactly one completely original SAT-style math practice question.
Use no official or released College Board question wording or close imitation.
Use concise wording and avoid unnecessary story text. Return a Pydantic-compatible structured object only.

Work order:
Domain: {DOMAIN_LABELS[work_order.domain]} ({work_order.domain})
Skill id: {work_order.skill}
Allowed skill concepts: {', '.join(skill_concepts)}
Required concept: {work_order.concept}
Difficulty: {work_order.difficulty.value}
Response type: {work_order.response_type.value}
Context type: {work_order.context_type.value}
Visual type: {work_order.visual_type.value}

Requirements:
- Match skill, concept, difficulty, context_type, and response_type exactly.
- For multiple choice provide exactly four distinct choices and one correct_answer label; accepted_answers must be empty.
- For student-produced response provide no choices and no correct_answer label; accepted_answers must contain mathematically equivalent numeric forms.
- For visual_type none set visual_spec to null. Otherwise return a complete validated VisualSpec first; all visual data must be mathematical parameters, never SVG, code, or pixel data.
- The visual must be necessary/useful and faithfully represent the question.
- Include a concise correct explanation.
- Set verification_spec to a supported deterministic method and provide only the problem's mathematical source data, not a claimed answer. Use llm_only only when none of the supported verification methods fits.
- Deterministic verification instructions for this skill: {verification_guidance}
- Use real-number solutions only.
"""
    response_model = question_draft_response_model(work_order.visual_type)
    generated_draft, model = call_structured(
        prompt,
        response_model,
        "generator",
        candidate_id,
        api_model_failures,
    )
    draft = QuestionDraft.model_validate(generated_draft.model_dump())
    if not isinstance(draft, QuestionDraft):
        raise ValueError("Generator returned an invalid structured draft.")
    return draft, model


def _run_candidate(
    call_structured: StructuredCaller,
    work_order: QuestionWorkOrder,
    candidate_id: str,
    api_model_failures: list[dict],
) -> tuple[dict[str, Any], bool, str | None]:
    timestamp = _timestamp()
    draft = None
    solver = None
    verification = None
    review = None
    visual_path = None
    generator_output = None
    deterministic_adjustments: list[str] = []
    models_used: dict[str, str] = {}
    stage = "generator"

    try:
        print("\n[1/4] GENERATOR")
        draft, models_used["generator"] = _create_draft(call_structured, work_order, candidate_id, api_model_failures)
        generator_output = draft.model_dump(mode="json")
        if not _matches_work_order(draft, work_order):
            raise ValueError("Generated draft does not match its work order.")

        draft, deterministic_adjustments = normalize_inequality_choices(draft)
        for adjustment in deterministic_adjustments:
            print(f"Deterministic adjustment: {adjustment}")

        stage = "renderer"
        if draft.visual_spec:
            visual_path = VISUALS_DIR / f"{candidate_id}.png"
            print(f"Rendering {draft.visual_spec.visual_type} visual...")
            render_visual(draft.visual_spec, visual_path)
            if not visual_path.is_file() or visual_path.stat().st_size < 1000:
                raise ValueError("Renderer did not produce a valid PNG.")
        print("Generator output validated.")

        stage = "solver"
        print("\n[2/4] INDEPENDENT SOLVER")
        solver_prompt = _student_prompt(draft)
        solver_contents: Any = solver_prompt
        if visual_path:
            solver_contents = [
                types.Part.from_text(text=solver_prompt),
                types.Part.from_bytes(data=visual_path.read_bytes(), mime_type="image/png"),
            ]
        solver, models_used["solver"] = call_structured(
            solver_contents, StudentSolverResult, "solver", candidate_id, api_model_failures
        )
        if not isinstance(solver, StudentSolverResult):
            raise ValueError("Solver returned an invalid structured result.")
        solver_matches = _solver_matches(draft, solver)
        print(f"Solver answer: {solver.selected_choice or solver.numeric_answer}")

        stage = "verifier"
        verification = verify_draft(draft)
        print(f"Deterministic verification: {verification.verification_method} ({'PASS' if verification.verified else 'NOT VERIFIED'})")
        if verification.verification_method != "llm_only" and not verification.verified:
            raise CandidateRejected("DETERMINISTIC_VERIFICATION_FAILED")
        if work_order.skill in SUPPORTED_SMOKE_SKILLS and verification.verification_method == "llm_only":
            raise CandidateRejected("DETERMINISTIC_VERIFICATION_UNAVAILABLE")
        if not solver_matches:
            raise CandidateRejected("ANSWER_DISAGREEMENT")

        stage = "reviewer"
        print("\n[3/4] QA REVIEWER")
        review_prompt = _review_prompt(work_order, draft, solver, verification)
        review_contents: Any = review_prompt
        if visual_path:
            review_contents = [
                types.Part.from_text(text=review_prompt),
                types.Part.from_bytes(data=visual_path.read_bytes(), mime_type="image/png"),
            ]
        review, models_used["reviewer"] = call_structured(
            review_contents, VisualReviewResult, "reviewer", candidate_id, api_model_failures
        )
        if not isinstance(review, VisualReviewResult):
            raise ValueError("Reviewer returned an invalid structured result.")
        record = _candidate_record(candidate_id, work_order, timestamp, draft, solver, verification, review, models_used, visual_path, api_model_failures, generator_output, deterministic_adjustments)
        return record, solver_matches, "REVIEWED"
    except APIInfrastructureFailure as error:
        error.candidate_record = _candidate_record(
            candidate_id,
            work_order,
            timestamp,
            draft,
            solver,
            verification,
            review,
            models_used,
            visual_path,
            api_model_failures,
            generator_output,
            deterministic_adjustments,
        )
        raise
    except APIInfrastructureFailure as error:
        error.candidate_record = _candidate_record(
            candidate_id,
            work_order,
            timestamp,
            draft,
            solver,
            verification,
            review,
            models_used,
            visual_path,
            api_model_failures,
            generator_output,
            deterministic_adjustments,
        )
        raise
    except CandidateRejected as error:
        record = _candidate_record(candidate_id, work_order, timestamp, draft, solver, verification, review, models_used, visual_path, api_model_failures, generator_output, deterministic_adjustments)
        return record, False, error.reason
    except Exception as error:
        reason = _format_reason(error, "API_FAILURE")
        if stage == "renderer":
            reason = "VISUAL_RENDER_FAILURE"
        elif isinstance(error, (ValueError, ValidationError)):
            reason = "INVALID_OUTPUT"
        record = _candidate_record(candidate_id, work_order, timestamp, draft, solver, verification, review, models_used, visual_path, api_model_failures, generator_output, deterministic_adjustments)
        record["failure"] = {"stage": stage, "error_type": type(error).__name__}
        return record, False, reason


class CandidateRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def run_visual_smoke_test(
    call_structured: StructuredCaller,
    max_total_attempts: int = MAX_TOTAL_ATTEMPTS,
) -> dict[str, Any]:
    ensure_pipeline_directories()
    run_id = f"RUN-{uuid4().hex.upper()}"
    run_started = _timestamp()
    approved_stems: list[str] = []
    visual_stats = {
        work_order.visual_type.value: {"rendered": 0, "approved": 0, "rejected": 0, "render_failures": 0}
        for work_order in VISUAL_SMOKE_TEST
    }
    api_model_failures: list[dict] = []
    generated_by_domain: dict[str, int] = {}
    generated_by_difficulty: dict[str, int] = {}
    deterministic_counts: dict[str, int] = {}
    total_attempts = 0
    work_order_index = 0

    print("=" * 60)
    print("TESTFORGE")
    print("Branch 001 - Visual Smoke Test")
    print(f"Target: {len(VISUAL_SMOKE_TEST)} approved questions")
    print("=" * 60)

    while total_attempts < max_total_attempts and work_order_index < len(VISUAL_SMOKE_TEST):
        work_order = VISUAL_SMOKE_TEST[work_order_index]
        total_attempts += 1
        candidate_id = f"TF-{work_order.id}-{run_id[-8:]}-{total_attempts:03d}"
        visual_type = work_order.visual_type.value
        print("\n" + "-" * 60)
        print(f"WORK ORDER {work_order_index + 1}/{len(VISUAL_SMOKE_TEST)}")
        print(f"{DOMAIN_LABELS[work_order.domain]} | {work_order.skill} | {work_order.concept}")
        print(f"{work_order.difficulty.value.title()} | {work_order.response_type.value} | visual: {visual_type}")
        print(f"Attempt {total_attempts}: {candidate_id}")

        record, solver_matches, outcome = _run_candidate(call_structured, work_order, candidate_id, api_model_failures)
        draft_data = record.get("question")
        if draft_data:
            generated_by_domain[work_order.domain] = generated_by_domain.get(work_order.domain, 0) + 1
            generated_by_difficulty[work_order.difficulty.value] = generated_by_difficulty.get(work_order.difficulty.value, 0) + 1
        visual_path_value = record.get("visual_image_path")
        verification_data = record.get("deterministic_verifier_output")
        if verification_data and verification_data.get("verification_method") != "llm_only":
            method = verification_data["verification_method"]
            deterministic_counts[method] = deterministic_counts.get(method, 0) + 1
        if visual_path_value:
            visual_stats[visual_type]["rendered"] += 1
        elif outcome == "VISUAL_RENDER_FAILURE":
            visual_stats[visual_type]["render_failures"] += 1

        if outcome == "REVIEWED":
            draft = QuestionDraft.model_validate(draft_data)
            solver = StudentSolverResult.model_validate(record["solver_output"])
            verification = VerificationResult.model_validate(record["deterministic_verifier_output"])
            review = VisualReviewResult.model_validate(record["reviewer_output"])
            similarity = _similarity(draft.stem, approved_stems)
            reasons = _rejection_reasons(draft, solver, verification, review, similarity)
            if not reasons:
                record["similarity_to_approved"] = similarity
                approved_path = APPROVED_DIR / f"{candidate_id}.json"
                save_json_record(approved_path, record)
                approved_stems.append(draft.stem)
                visual_stats[visual_type]["approved"] += 1
                work_order_index += 1
                print("APPROVED")
                print(candidate_id)
                print(f"Verification: {verification.verification_method}")
                print(f"Progress: {work_order_index}/{len(VISUAL_SMOKE_TEST)} approved")
                continue
            reason = ", ".join(reasons)
            record["rejection_reasons"] = reasons
            record["similarity_to_approved"] = similarity
        else:
            reason = outcome
            record["rejection_reasons"] = [reason]
            if reason == "VISUAL_RENDER_FAILURE":
                visual_stats[visual_type]["render_failures"] += 1

        record["status"] = "rejected"
        record["rejection_reason"] = record["rejection_reasons"][0]
        rejected_path = REJECTED_DIR / f"{candidate_id}.json"
        save_json_record(rejected_path, record)
        visual_stats[visual_type]["rejected"] += 1
        print("REJECTED")
        print(f"Reason: {reason}")
        print("Generating replacement for the same work order...")

    completed_at = _timestamp()
    approved_count = work_order_index
    rejected_count = total_attempts - approved_count
    run_summary = {
        "run_id": run_id,
        "started_at": run_started,
        "completed_at": completed_at,
        "target_approved": len(VISUAL_SMOKE_TEST),
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "total_attempts": total_attempts,
        "approval_rate": round(approved_count / total_attempts * 100, 1) if total_attempts else 0.0,
        "api_model_failures": api_model_failures,
        "questions_generated_by_domain": generated_by_domain,
        "questions_generated_by_difficulty": generated_by_difficulty,
        "visual_type_results": visual_stats,
        "deterministic_verification_counts": deterministic_counts,
        "unfilled_work_orders": [item.id for item in VISUAL_SMOKE_TEST[work_order_index:]],
    }
    save_json_record(RUNS_DIR / f"{run_id}.json", run_summary)

    print("\n" + "=" * 60)
    print("VISUAL SMOKE TEST COMPLETE")
    print(f"Approved: {approved_count}/{len(VISUAL_SMOKE_TEST)}")
    print(f"Rejected: {rejected_count}")
    print(f"Total attempts: {total_attempts}/{max_total_attempts}")
    print("Visual types rendered/approved/rejected:")
    for kind, counts in visual_stats.items():
        print(f"  {kind}: {counts['rendered']} rendered, {counts['approved']} approved, {counts['rejected']} rejected, {counts['render_failures']} render failures")
    print("Deterministic verifier methods:")
    for method, count in deterministic_counts.items():
        print(f"  {method}: {count}")
    print("Saved: data/approved/, data/rejected/, data/visuals/, data/runs/")
    print("=" * 60)
    return run_summary


def run_resumable_smoke_test(
    call_structured: StructuredCaller,
    valid_existing_approvals: dict[str, dict[str, Any]],
    api_state: dict[str, Any],
    fresh: bool = False,
    max_candidate_attempts: int = MAX_TOTAL_ATTEMPTS,
) -> dict[str, Any]:
    ensure_pipeline_directories()
    run_id = f"RUN-{uuid4().hex.upper()}"
    started_at = _timestamp()
    completed_records = {} if fresh else dict(valid_existing_approvals)
    approved_stems = [
        QuestionDraft.model_validate(record["question"]).stem
        for record in valid_existing_approvals.values()
    ]
    work_order_positions = {work.id: index + 1 for index, work in enumerate(VISUAL_SMOKE_TEST)}
    visual_stats = {
        work.visual_type.value: {"rendered": 0, "approved": 0, "rejected": 0, "render_failures": 0}
        for work in VISUAL_SMOKE_TEST
    }
    api_model_failures: list[dict] = []
    infrastructure_failures: list[dict] = []
    generated_by_domain: dict[str, int] = {}
    generated_by_difficulty: dict[str, int] = {}
    deterministic_counts: dict[str, int] = {}
    candidate_attempts = 0
    candidate_rejections = 0
    paused_reason: str | None = None

    def make_summary(status: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "started_at": started_at,
            "completed_at": _timestamp(),
            "status": status,
            "target_approved": len(VISUAL_SMOKE_TEST),
            "valid_existing_approvals_reused": 0 if fresh else len(valid_existing_approvals),
            "approved_count": len(completed_records),
            "candidate_attempts": candidate_attempts,
            "candidate_rejections": candidate_rejections,
            "rejected_count": candidate_rejections,
            "total_attempts": candidate_attempts,
            "approval_rate": round(
                (len(completed_records) - (0 if fresh else len(valid_existing_approvals)))
                / candidate_attempts * 100,
                1,
            ) if candidate_attempts else 0.0,
            "api_requests": api_state["api_requests"],
            "api_failures": api_state["api_failures"],
            "rate_limit_events": api_state["rate_limit_events"],
            "quota_events": api_state["quota_events"],
            "unknown_429_events": api_state["unknown_429_events"],
            "invalid_outputs": api_state["invalid_outputs"],
            "api_model_failures": api_model_failures,
            "infrastructure_failures": infrastructure_failures,
            "questions_generated_by_domain": generated_by_domain,
            "questions_generated_by_difficulty": generated_by_difficulty,
            "visual_type_results": visual_stats,
            "deterministic_verification_counts": deterministic_counts,
            "completed_work_order_ids": sorted(completed_records),
            "remaining_work_order_ids": [
                work.id for work in VISUAL_SMOKE_TEST if work.id not in completed_records
            ],
            "stop_reason": paused_reason,
        }

    def checkpoint(status: str = "RUNNING") -> dict[str, Any]:
        summary = make_summary(status)
        save_json_record(RUNS_DIR / f"{run_id}.json", summary)
        return summary

    checkpoint()
    try:
        while len(completed_records) < len(VISUAL_SMOKE_TEST):
            if candidate_attempts >= max_candidate_attempts:
                paused_reason = "MAX_CANDIDATE_ATTEMPTS"
                break

            work_order = next(
                work for work in VISUAL_SMOKE_TEST if work.id not in completed_records
            )
            candidate_id = f"TF-{work_order.id}-{run_id[-8:]}-{uuid4().hex[:8].upper()}"
            print("\n" + "-" * 55)
            print(f"WORK ORDER {work_order_positions[work_order.id]}/{len(VISUAL_SMOKE_TEST)}")
            print(f"{work_order.skill} | {work_order.difficulty.value} | {work_order.visual_type.value} | {work_order.response_type.value}")
            print(f"Question candidates generated this run: {candidate_attempts}/{max_candidate_attempts}")
            try:
                record, _, outcome = _run_candidate(
                    call_structured,
                    work_order,
                    candidate_id,
                    api_model_failures,
                )
            except APIInfrastructureFailure as error:
                partial = error.candidate_record or {}
                question_data = partial.get("question")
                if question_data:
                    candidate_attempts += 1
                    generated_question = QuestionDraft.model_validate(question_data)
                    generated_by_domain[generated_question.domain] = generated_by_domain.get(generated_question.domain, 0) + 1
                    generated_by_difficulty[generated_question.difficulty.value] = generated_by_difficulty.get(generated_question.difficulty.value, 0) + 1
                    image_path = partial.get("visual_image_path")
                    if image_path:
                        visual_stats[work_order.visual_type.value]["rendered"] += 1
                failure_event = {
                    "timestamp": _timestamp(),
                    "reason": error.reason,
                    "stage": error.stage,
                    "model": error.model,
                    "status_code": error.status_code,
                    "classification": error.classification,
                    "candidate_id": partial.get("candidate_id", candidate_id),
                    "candidate_generated": bool(question_data),
                    "partial_candidate": partial if question_data else None,
                    "message": str(error)[:1200],
                }
                infrastructure_failures.append(failure_event)
                paused_reason = error.reason
                checkpoint("PAUSED")
                break

            question_data = record.get("question")
            if question_data:
                candidate_attempts += 1
                generated_question = QuestionDraft.model_validate(question_data)
                generated_by_domain[generated_question.domain] = generated_by_domain.get(generated_question.domain, 0) + 1
                generated_by_difficulty[generated_question.difficulty.value] = generated_by_difficulty.get(generated_question.difficulty.value, 0) + 1
            if record.get("visual_image_path"):
                visual_stats[work_order.visual_type.value]["rendered"] += 1
            elif outcome == "VISUAL_RENDER_FAILURE":
                visual_stats[work_order.visual_type.value]["render_failures"] += 1
            verifier_data = record.get("deterministic_verifier_output")
            if verifier_data and verifier_data.get("verification_method") != "llm_only":
                method = verifier_data["verification_method"]
                deterministic_counts[method] = deterministic_counts.get(method, 0) + 1

            if outcome == "REVIEWED":
                draft = QuestionDraft.model_validate(question_data)
                solver = StudentSolverResult.model_validate(record["solver_output"])
                verification = VerificationResult.model_validate(verifier_data)
                review = VisualReviewResult.model_validate(record["reviewer_output"])
                similarity = _similarity(draft.stem, approved_stems)
                reasons = _rejection_reasons(draft, solver, verification, review, similarity)
                if not reasons:
                    record["similarity_to_approved"] = similarity
                    record["status"] = "approved"
                    record["human_approved"] = False
                    record["training_rights_verified"] = False
                    save_json_record(APPROVED_DIR / f"{candidate_id}.json", record)
                    completed_records[work_order.id] = record
                    approved_stems.append(draft.stem)
                    visual_stats[work_order.visual_type.value]["approved"] += 1
                    print("APPROVED", work_order.id, candidate_id)
                    checkpoint()
                    continue

                candidate_rejections += 1
                record["rejection_reasons"] = reasons
                record["rejection_reason"] = reasons[0]
                record["status"] = "rejected"
                record["similarity_to_approved"] = similarity
                save_json_record(REJECTED_DIR / f"{candidate_id}.json", record)
                visual_stats[work_order.visual_type.value]["rejected"] += 1
                print("QUESTION REJECTED:", ", ".join(reasons))
                checkpoint()
                continue

            if outcome in {
                "ANSWER_DISAGREEMENT",
                "DETERMINISTIC_VERIFICATION_FAILED",
                "DETERMINISTIC_VERIFICATION_UNAVAILABLE",
            }:
                candidate_rejections += 1
                record["rejection_reasons"] = [outcome]
                record["rejection_reason"] = outcome
                record["status"] = "rejected"
                save_json_record(REJECTED_DIR / f"{candidate_id}.json", record)
                visual_stats[work_order.visual_type.value]["rejected"] += 1
                print("QUESTION REJECTED:", outcome)
            else:
                infrastructure_failures.append({
                    "timestamp": _timestamp(),
                    "reason": outcome,
                    "stage": record.get("failure", {}).get("stage"),
                    "candidate_id": candidate_id,
                    "candidate_generated": bool(question_data),
                    "partial_candidate": record if question_data else None,
                })
                print("INFRASTRUCTURE FAILURE:", outcome)
            checkpoint()

    except KeyboardInterrupt:
        paused_reason = "USER_INTERRUPT"

    complete = len(completed_records) == len(VISUAL_SMOKE_TEST)
    if complete:
        status = "COMPLETE"
    else:
        status = "PAUSED"
        if paused_reason is None:
            paused_reason = "MAX_CANDIDATE_ATTEMPTS"
    summary = checkpoint(status)

    if complete:
        print("\n" + "=" * 55)
        print("VISUAL SMOKE TEST COMPLETE")
        print("10/10 work orders approved")
        for work in VISUAL_SMOKE_TEST:
            print(f"{work.id}: {work.visual_type.value} | {work.response_type.value}")
        print("=" * 55)
    elif paused_reason == "USER_INTERRUPT":
        print("Production paused by user.")
        print("Progress saved.")
        print("Run python main.py again to resume.")
    else:
        print("\n" + "=" * 55)
        print("PRODUCTION PAUSED")
        print(f"Reason: {paused_reason}")
        print(f"Question candidates generated this run: {candidate_attempts}")
        print(f"API requests: {api_state['api_requests']}")
        print(f"API failures: {api_state['api_failures']}")
        print(f"Candidate rejections: {candidate_rejections}")
        print(f"Infrastructure failures: {len(infrastructure_failures)}")
        print(f"Progress preserved: {len(completed_records)}/{len(VISUAL_SMOKE_TEST)}")
        print(f"Remaining: {len(VISUAL_SMOKE_TEST) - len(completed_records)}")
        print("Run again later to resume.")
        print("=" * 55)
    return summary
