import time
import argparse
import re
import os
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar
from uuid import uuid4

from dotenv import load_dotenv
from google import genai
from google.genai import types
from blueprint import VISUAL_SMOKE_TEST
from pydantic import BaseModel, Field, ValidationError, model_validator
from model_router import BudgetReviewRequired, production_task_from_stage, route_task
from models import APIInfrastructureFailure
from storage import (
    APPROVED_PATH,
    REJECTED_PATH,
    RUNS_DIR,
    RUNS_PATH,
    append_record,
    ensure_data_files,
    ensure_pipeline_directories,
    save_json_record,
)


load_dotenv()


class AnswerChoices(BaseModel):
    A: str
    B: str
    C: str
    D: str


class QuestionWorkOrder(BaseModel):
    id: str
    domain: Literal[
        "algebra",
        "advanced_math",
        "problem_solving_and_data_analysis",
        "geometry_and_trigonometry",
    ]
    skill: str
    difficulty: Literal["easy", "medium", "hard"]
    question_type: Literal["multiple_choice"]


class PracticeQuestion(BaseModel):
    section: Literal["math"]

    domain: Literal[
        "algebra",
        "advanced_math",
        "problem_solving_and_data_analysis",
        "geometry_and_trigonometry"
    ]

    skill: str

    difficulty: Literal[
        "easy",
        "medium",
        "hard"
    ]

    question: str

    type: Literal[
        "multiple_choice"
    ]

    choices: AnswerChoices

    correct_answer: Literal[
        "A",
        "B",
        "C",
        "D"
    ]

    explanation: str

    generator_confidence: float = Field(
        ge=0,
        le=1
    )

    @model_validator(mode="after")
    def validate_choices(self):
        choice_values = [value.strip() for value in self.choices.model_dump().values()]
        if any(not value for value in choice_values):
            raise ValueError("Answer choices must not be empty.")
        if len(set(choice_values)) != 4:
            raise ValueError("Answer choices must be distinct.")
        return self


class SolverResult(BaseModel):
    answer: Literal["A", "B", "C", "D"]
    reasoning: str
    confidence: float = Field(ge=0, le=1)


class ReviewResult(BaseModel):
    pass_review: bool
    score: int = Field(ge=0, le=100)
    mathematical_validity: int = Field(ge=0, le=100)
    clarity: int = Field(ge=0, le=100)
    distractor_quality: int = Field(ge=0, le=100)
    difficulty_accuracy: int = Field(ge=0, le=100)
    difficulty_matches: bool
    originality_risk: Literal["low", "medium", "high"]
    issues: list[str]
    reviewer_notes: str


def create_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            "Check your .env file."
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
)
TARGET_APPROVED = 10
MAX_TOTAL_ATTEMPTS = 60
MAX_RETRIES = 2
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
RETRYABLE_MESSAGES = ("service_unavailable", "high demand", "timeout", "timed out")
RATE_LIMIT_RETRY_DELAYS = (5, 15)
MODEL_CIRCUIT_COOLDOWN_SECONDS = 120
MAX_API_REQUESTS_PER_RUN = 360


def new_api_state() -> dict[str, Any]:
    return {
        "api_requests": 0,
        "api_failures": 0,
        "rate_limit_events": 0,
        "quota_events": 0,
        "unknown_429_events": 0,
        "invalid_outputs": 0,
        "consecutive_rate_limit_errors": {},
        "model_circuit_open_until": {},
        "infrastructure_failures": [],
        "stop_reason": None,
    }


def sanitized_error_message(error: Exception) -> str:
    message = str(getattr(error, "message", "") or error)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return message[:1200]


def error_status_code(error: Exception) -> int | None:
    status_code = getattr(error, "code", None)
    if status_code is None:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def classify_429(message: str) -> str:
    normalized = message.lower()
    quota_phrases = (
        "quota", "daily limit", "per day", "per-day", "billing", "resource exhausted",
        "resource_exhausted", "limit: 0", "exceeded your current",
    )
    rate_phrases = (
        "rate limit", "rate-limit", "too many requests", "requests per minute",
        "per minute", "per-minute", "requests per second", "rate exceeded",
    )
    if any(phrase in normalized for phrase in quota_phrases):
        return "QUOTA_EXHAUSTED"
    if any(phrase in normalized for phrase in rate_phrases):
        return "RATE_LIMIT"
    return "UNKNOWN_429"


def is_temporary_api_error(error: Exception) -> bool:
    status_code = error_status_code(error)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    error_text = f"{sanitized_error_message(error)} {type(error).__name__}".lower()
    return (
        any(str(code) in error_text for code in RETRYABLE_STATUS_CODES)
        or any(message in error_text for message in RETRYABLE_MESSAGES)
    )


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def generate_structured_response(
    prompt: str | list[types.Part],
    response_model: type[ResponseModel],
    stage: str,
    candidate_id: str,
    api_model_failures: list[dict],
    api_state: dict[str, Any] | None = None,
) -> tuple[ResponseModel, str]:
    api_state = api_state if api_state is not None else new_api_state()
    if api_state["api_requests"] >= MAX_API_REQUESTS_PER_RUN:
        api_state["stop_reason"] = "API_REQUEST_BUDGET"
        raise APIInfrastructureFailure("API_REQUEST_BUDGET", "Run API request budget reached.", stage)

    task_type = production_task_from_stage(stage)
    try:
        result, model = route_task(
            task_type,
            prompt,
            response_model,
            "production",
            stage,
            candidate_id,
        )
    except BudgetReviewRequired as error:
        api_state["stop_reason"] = "BUDGET_REVIEW_REQUIRED"
        failure = {
            "candidate_id": candidate_id,
            "stage": stage,
            "model": None,
            "failure_type": "BUDGET_REVIEW_REQUIRED",
            "error_type": type(error).__name__,
            "message": "Daily AI budget would be exceeded.",
        }
        api_model_failures.append(failure)
        api_state["infrastructure_failures"].append(failure)
        raise APIInfrastructureFailure("BUDGET_REVIEW_REQUIRED", "Daily AI budget would be exceeded.", stage) from error
    except (ValidationError, ValueError) as error:
        api_state["invalid_outputs"] += 1
        message = sanitized_error_message(error)
        failure = {
            "candidate_id": candidate_id,
            "stage": stage,
            "model": None,
            "failure_type": "INVALID_OUTPUT",
            "error_type": type(error).__name__,
            "message": message,
        }
        api_model_failures.append(failure)
        raise
    except APIInfrastructureFailure as error:
        api_state["api_failures"] += 1
        api_state["stop_reason"] = error.reason
        failure = {
            "candidate_id": candidate_id,
            "stage": stage,
            "model": error.model,
            "failure_type": "API_FAILURE",
            "error_type": type(error).__name__,
            "status_code": error.status_code,
            "classification": error.classification,
            "message": sanitized_error_message(error),
        }
        api_model_failures.append(failure)
        api_state["infrastructure_failures"].append(failure)
        raise

    api_state["api_requests"] += 1
    print(f"Success using {model}.")
    return result, model


def probe_gemini(api_state: dict[str, Any]) -> str:
    """Make exactly one small request before resuming production."""
    try:
        client = create_client()
        api_state["api_requests"] += 1
        response = client.models.generate_content(
            model=MODELS[0],
            contents="Reply exactly OK.",
        )
        if (response.text or "").strip() != "OK":
            raise ValueError("Gemini probe did not return exactly OK.")
        return MODELS[0]
    except Exception as error:
        status_code = error_status_code(error)
        message = sanitized_error_message(error)
        classification = classify_429(message) if status_code == 429 else None
        failure = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "stage": "probe",
            "model": MODELS[0],
            "status_code": status_code,
            "classification": classification,
            "error_type": type(error).__name__,
            "message": message,
        }
        api_state["api_failures"] += int(status_code is not None)
        if classification == "QUOTA_EXHAUSTED":
            api_state["quota_events"] += 1
            reason = "GEMINI_QUOTA_EXHAUSTED"
        elif classification == "RATE_LIMIT":
            api_state["rate_limit_events"] += 1
            reason = "GEMINI_RATE_LIMIT"
        elif classification == "UNKNOWN_429":
            api_state["unknown_429_events"] += 1
            reason = "GEMINI_UNKNOWN_429"
        else:
            reason = "GEMINI_PROBE_FAILED"
        api_state["stop_reason"] = reason
        api_state["infrastructure_failures"].append(failure)
        raise APIInfrastructureFailure(
            reason,
            message,
            "probe",
            MODELS[0],
            status_code,
            classification,
        ) from error


def save_probe_pause(
    api_state: dict[str, Any],
    valid_existing_approvals: dict[str, dict],
    fresh: bool,
    reason: str,
) -> None:
    ensure_pipeline_directories()
    completed = set() if fresh else set(valid_existing_approvals)
    run_id = f"RUN-{uuid4().hex.upper()}"
    summary = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": "PAUSED",
        "stop_reason": reason,
        "target_approved": len(VISUAL_SMOKE_TEST),
        "valid_existing_approvals_reused": len(completed),
        "approved_count": len(completed),
        "candidate_attempts": 0,
        "candidate_rejections": 0,
        "rejected_count": 0,
        "total_attempts": 0,
        "api_requests": api_state["api_requests"],
        "api_failures": api_state["api_failures"],
        "rate_limit_events": api_state["rate_limit_events"],
        "quota_events": api_state["quota_events"],
        "unknown_429_events": api_state["unknown_429_events"],
        "api_model_failures": [],
        "infrastructure_failures": api_state["infrastructure_failures"],
        "completed_work_order_ids": sorted(completed),
        "remaining_work_order_ids": [work.id for work in VISUAL_SMOKE_TEST if work.id not in completed],
    }
    save_json_record(RUNS_DIR / f"{run_id}.json", summary)
    print("\n" + "=" * 55)
    print("PRODUCTION PAUSED")
    print(f"Reason: {reason}")
    print("Question candidates generated this run: 0")
    print(f"API requests: {api_state['api_requests']}")
    print(f"API failures: {api_state['api_failures']}")
    print(f"Progress preserved: {len(completed)}/{len(VISUAL_SMOKE_TEST)}")
    print(f"Remaining: {len(VISUAL_SMOKE_TEST) - len(completed)}")
    print("Run python main.py again to resume.")
    print("=" * 55)


class PipelineStageFailure(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def create_work_orders() -> list[QuestionWorkOrder]:
    work_order_specs = [
        ("algebra", "Solving linear equations in one variable", "easy"),
        ("algebra", "Solving linear equations in one variable", "medium"),
        ("algebra", "Solving systems of linear equations", "medium"),
        ("advanced_math", "Evaluating and interpreting functions", "medium"),
        ("problem_solving_and_data_analysis", "Solving percentage problems", "easy"),
        ("problem_solving_and_data_analysis", "Solving ratios and proportions", "medium"),
        ("advanced_math", "Solving quadratic equations", "medium"),
        ("advanced_math", "Solving quadratic equations", "hard"),
        ("problem_solving_and_data_analysis", "Interpreting data in tables and graphs", "medium"),
        ("geometry_and_trigonometry", "Solving geometry problems involving area and perimeter", "medium"),
    ]
    return [
        QuestionWorkOrder(
            id=f"WO-{index:03d}",
            domain=domain,
            skill=skill,
            difficulty=difficulty,
            question_type="multiple_choice",
        )
        for index, (domain, skill, difficulty) in enumerate(work_order_specs, start=1)
    ]


def generate_question(
    work_order: QuestionWorkOrder,
    candidate_id: str,
    api_model_failures: list[dict],
) -> tuple[PracticeQuestion, str]:
    prompt = f"""Create exactly ONE completely original SAT-style math practice question.
Do not reproduce or closely imitate official College Board or SAT questions.
Invent the question and context from scratch.

Work order:
Domain: {work_order.domain}
Skill: {work_order.skill}
Difficulty: {work_order.difficulty}
Question type: {work_order.question_type}

Match the work-order domain, skill, difficulty, and question type exactly.
Make the problem self-contained and solvable with standard high-school math.
Include exactly four distinct answer choices and exactly one mathematically
correct answer. Use realistic distractors based on common student mistakes.
Give a concise explanation that verifies the answer. Return only data matching
the provided schema.
"""
    question, model = generate_structured_response(
        prompt,
        PracticeQuestion,
        "generator",
        candidate_id,
        api_model_failures,
    )
    if (
        question.domain != work_order.domain
        or question.skill != work_order.skill
        or question.difficulty != work_order.difficulty
        or question.type != work_order.question_type
    ):
        api_model_failures.append({
            "candidate_id": candidate_id,
            "stage": "generator",
            "model": model,
            "attempt": 1,
            "failure_type": "INVALID_OUTPUT",
            "error_type": "WorkOrderMismatch",
            "status_code": None,
        })
        raise PipelineStageFailure("INVALID_OUTPUT", "Generated item did not match its work order.")
    return question, model


def solve_question(
    question: PracticeQuestion,
    candidate_id: str,
    api_model_failures: list[dict],
) -> tuple[SolverResult, str]:
    prompt = f"""You are an independent mathematics verification agent.

Solve this problem from scratch.

Do not assume the question writer's answer is correct.

Check your arithmetic carefully.

Choose exactly one answer: A, B, C, or D.

Provide concise reasoning and a confidence score between 0 and 1.

Question:
{question.question}

Answer choices:
A. {question.choices.A}
B. {question.choices.B}
C. {question.choices.C}
D. {question.choices.D}
"""
    return generate_structured_response(
        prompt, SolverResult, "solver", candidate_id, api_model_failures
    )


def review_question(
    question: PracticeQuestion,
    solver_result: SolverResult,
    candidate_id: str,
    api_model_failures: list[dict],
) -> tuple[ReviewResult, str]:
    prompt = f"""You are the final QA reviewer for an educational practice-question company.

Evaluate the question critically.

Check:
1. Mathematical validity.
2. Exactly one answer is correct.
3. The explanation is mathematically correct.
4. Wording is unambiguous.
5. The problem is self-contained.
6. Difficulty label is reasonable.
7. Distractors represent realistic student mistakes.
8. No answer choice accidentally duplicates another answer.
9. The problem reproduces and closely imitates official SAT/College Board questions.
10. The problem is educationally useful.

Difficulty guide:
EASY: direct one-step or straightforward multi-step application, little
interpretation, an obvious setup, and minimal conceptual traps.
MEDIUM: translating information into equations, multiple conceptual steps, a
less obvious setup, and meaningful distractors.
HARD: combining concepts, a non-obvious setup, careful reasoning, and multiple
plausible approaches or traps.

Be strict. If the claimed difficulty is substantially wrong, set
difficulty_accuracy below 80 and difficulty_matches to false. A question should
only pass if its score is at least 85, mathematical validity is at least 95,
difficulty_matches is true, it has no major ambiguity, exactly one answer is
correct, and originality risk is not high.

Question:
{question.question}

Domain: {question.domain}
Skill: {question.skill}
Difficulty: {question.difficulty}

Answer choices:
A. {question.choices.A}
B. {question.choices.B}
C. {question.choices.C}
D. {question.choices.D}

Generator's claimed answer: {question.correct_answer}
Generator's explanation: {question.explanation}
Solver's answer: {solver_result.answer}
Solver's reasoning: {solver_result.reasoning}

Return only data matching the provided schema. Set pass_review according to the
criteria above and include any concerns in issues and reviewer_notes.
"""
    return generate_structured_response(
        prompt, ReviewResult, "reviewer", candidate_id, api_model_failures
    )


def timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalized_question_text(question: str) -> str:
    without_punctuation = re.sub(r"[^\w\s]", "", question.lower())
    return " ".join(without_punctuation.split())


def highest_similarity(question: PracticeQuestion, approved: list[PracticeQuestion]) -> float:
    candidate_text = normalized_question_text(question.question)
    return max(
        (
            SequenceMatcher(None, candidate_text, normalized_question_text(item.question)).ratio()
            for item in approved
        ),
        default=0.0,
    )


def make_candidate_record(
    candidate_id: str,
    work_order: QuestionWorkOrder,
    timestamp: str,
    question: PracticeQuestion | None,
    solver_result: SolverResult | None,
    review: ReviewResult | None,
    models_used: dict[str, str],
) -> dict:
    return {
        "candidate_id": candidate_id,
        "work_order_id": work_order.id,
        "work_order": work_order.model_dump(mode="json"),
        "timestamp": timestamp,
        "question": question.model_dump(mode="json") if question else None,
        "solver_result": solver_result.model_dump(mode="json") if solver_result else None,
        "reviewer_result": review.model_dump(mode="json") if review else None,
        "models_used": models_used,
    }


def save_rejection(
    rejected_this_run: list[dict],
    candidate_id: str,
    work_order: QuestionWorkOrder,
    timestamp: str,
    question: PracticeQuestion | None,
    solver_result: SolverResult | None,
    review: ReviewResult | None,
    models_used: dict[str, str],
    reasons: list[str],
    stage: str,
) -> None:
    record = make_candidate_record(
        candidate_id,
        work_order,
        timestamp,
        question,
        solver_result,
        review,
        models_used,
    )
    record["rejection_reason"] = reasons[0]
    record["rejection_reasons"] = reasons
    record["failed_stage"] = stage
    rejected_this_run.append(record)
    append_record(REJECTED_PATH, record)


def review_rejection_reasons(review: ReviewResult) -> list[str]:
    reasons = []
    if review.score < 85:
        reasons.append("REVIEW_SCORE_TOO_LOW")
    if review.mathematical_validity < 95:
        reasons.append("MATHEMATICAL_VALIDITY_TOO_LOW")
    if not review.difficulty_matches or review.difficulty_accuracy < 80:
        reasons.append("DIFFICULTY_MISMATCH")
    if review.originality_risk == "high":
        reasons.append("HIGH_ORIGINALITY_RISK")
    if not review.pass_review:
        reasons.append("REVIEWER_REJECTED")
    return reasons


def run_production() -> dict:
    ensure_data_files()
    work_orders = create_work_orders()
    run_id = f"RUN-{uuid4().hex.upper()}"
    started_at = timestamp_now()
    approved_this_run: list[PracticeQuestion] = []
    rejected_this_run: list[dict] = []
    api_model_failures: list[dict] = []
    generated_by_domain = {domain: 0 for domain in PracticeQuestion.model_fields["domain"].annotation.__args__}
    generated_by_difficulty = {difficulty: 0 for difficulty in ("easy", "medium", "hard")}
    total_attempts = 0
    work_order_index = 0

    print("=" * 55)
    print("TESTFORGE")
    print("Branch 001 - Autonomous Production Run")
    print(f"Target: {TARGET_APPROVED} approved questions")
    print("=" * 55)

    while (
        len(approved_this_run) < TARGET_APPROVED
        and total_attempts < MAX_TOTAL_ATTEMPTS
        and work_order_index < len(work_orders)
    ):
        work_order = work_orders[work_order_index]
        total_attempts += 1
        candidate_id = f"TF-{uuid4().hex.upper()}"
        candidate_timestamp = timestamp_now()
        question = None
        solver_result = None
        review = None
        models_used: dict[str, str] = {}
        current_stage = "generator"

        print("\n" + "-" * 55)
        print(f"WORK ORDER {work_order_index + 1}/{len(work_orders)}")
        print(f"Skill: {work_order.skill}")
        print(f"Difficulty: {work_order.difficulty.title()}")
        print(f"Attempt {total_attempts}  |  Candidate {candidate_id}")

        try:
            print("\n[1/3] GENERATOR........")
            question, models_used["generator"] = generate_question(
                work_order,
                candidate_id,
                api_model_failures,
            )
            generated_by_domain[question.domain] += 1
            generated_by_difficulty[question.difficulty] += 1
            print(f"Candidate generated by {models_used['generator']}.")
            print(f"Question: {question.question}")
            for label, choice in question.choices.model_dump().items():
                print(f"{label}. {choice}")
            print(f"Generator answer: {question.correct_answer}")

            current_stage = "solver"
            print("\n[2/3] SOLVER...........")
            solver_result, models_used["solver"] = solve_question(
                question,
                candidate_id,
                api_model_failures,
            )
            answers_match = solver_result.answer == question.correct_answer
            print(f"Generator: {question.correct_answer}")
            print(f"Solver: {solver_result.answer}")
            print(f"ANSWER CHECK.....{'PASS' if answers_match else 'FAIL'}")

            if not answers_match:
                reasons = ["ANSWER_DISAGREEMENT"]
                save_rejection(
                    rejected_this_run,
                    candidate_id,
                    work_order,
                    candidate_timestamp,
                    question,
                    solver_result,
                    None,
                    models_used,
                    reasons,
                    current_stage,
                )
                print("REJECTED")
                print(f"Reason: {reasons[0]}")
                print("Generating replacement automatically...")
                continue

            current_stage = "reviewer"
            print("\n[3/3] QA REVIEWER")
            review, models_used["reviewer"] = review_question(
                question,
                solver_result,
                candidate_id,
                api_model_failures,
            )

            similarity = highest_similarity(question, approved_this_run)
            similarity_passes = similarity < 0.82
            print(f"REVIEW............{review.score}/100")
            print(f"MATHEMATICAL VALIDITY...{review.mathematical_validity}/100")
            print(f"DIFFICULTY........{'MATCH' if review.difficulty_matches else 'MISMATCH'}")
            print(f"SIMILARITY........{'PASS' if similarity_passes else f'FAIL ({similarity:.3f})'}")

            reasons = review_rejection_reasons(review)
            if not similarity_passes:
                reasons.append("TOO_SIMILAR_TO_EXISTING_QUESTION")
            final_pass = (
                answers_match
                and review.score >= 85
                and review.mathematical_validity >= 95
                and review.difficulty_matches is True
                and review.difficulty_accuracy >= 80
                and review.originality_risk != "high"
                and review.pass_review is True
                and similarity_passes
            )

            if final_pass:
                record = make_candidate_record(
                    candidate_id,
                    work_order,
                    candidate_timestamp,
                    question,
                    solver_result,
                    review,
                    models_used,
                )
                append_record(APPROVED_PATH, record)
                approved_this_run.append(question)
                work_order_index += 1
                print("APPROVED")
                print(candidate_id)
                print(f"Progress: {len(approved_this_run)}/{TARGET_APPROVED} approved")
            else:
                if not reasons:
                    reasons.append("REVIEWER_REJECTED")
                save_rejection(
                    rejected_this_run,
                    candidate_id,
                    work_order,
                    candidate_timestamp,
                    question,
                    solver_result,
                    review,
                    models_used,
                    reasons,
                    current_stage,
                )
                print("REJECTED")
                print(f"Reason: {', '.join(reasons)}")
                print("Generating replacement automatically...")

        except PipelineStageFailure as error:
            reasons = [error.reason]
            save_rejection(
                rejected_this_run,
                candidate_id,
                work_order,
                candidate_timestamp,
                question,
                solver_result,
                review,
                models_used,
                reasons,
                current_stage,
            )
            print("REJECTED")
            print(f"Reason: {error.reason} during {current_stage}.")
            print("Generating replacement automatically...")
        except Exception as error:
            reason = "INVALID_OUTPUT" if isinstance(error, (ValidationError, ValueError)) else "API_FAILURE"
            api_model_failures.append({
                "candidate_id": candidate_id,
                "stage": current_stage,
                "model": None,
                "attempt": 1,
                "failure_type": reason,
                "error_type": type(error).__name__,
                "status_code": None,
            })
            save_rejection(
                rejected_this_run,
                candidate_id,
                work_order,
                candidate_timestamp,
                question,
                solver_result,
                review,
                models_used,
                [reason],
                current_stage,
            )
            print("REJECTED")
            print(f"Reason: {reason} during {current_stage}.")
            print("Generating replacement automatically...")

    completed_at = timestamp_now()
    approved_count = len(approved_this_run)
    rejected_count = len(rejected_this_run)
    approval_rate = round(approved_count / total_attempts * 100, 1) if total_attempts else 0.0
    run_summary = {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "target_approved": TARGET_APPROVED,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "total_attempts": total_attempts,
        "approval_rate": approval_rate,
        "api_model_failures": api_model_failures,
        "questions_generated_by_domain": generated_by_domain,
        "questions_generated_by_difficulty": generated_by_difficulty,
    }
    append_record(RUNS_PATH, run_summary)

    approved_by_difficulty = {difficulty: 0 for difficulty in ("easy", "medium", "hard")}
    for question in approved_this_run:
        approved_by_difficulty[question.difficulty] += 1

    print("\n" + "=" * 55)
    print("PRODUCTION RUN COMPLETE")
    print(f"Target approved:     {TARGET_APPROVED}")
    print(f"Approved:            {approved_count}")
    print(f"Rejected attempts:   {rejected_count}")
    print(f"Total attempts:      {total_attempts}")
    print(f"Approval rate:       {approval_rate:.1f}%")
    print(f"Easy:                {approved_by_difficulty['easy']}")
    print(f"Medium:              {approved_by_difficulty['medium']}")
    print(f"Hard:                {approved_by_difficulty['hard']}")
    print(f"API/model failures:  {len(api_model_failures)}")
    print("\nSaved:")
    print("data/approved_questions.json")
    print("data/rejected_questions.json")
    print("data/runs.json")
    print("=" * 55)
    return run_summary




def main():
    from visual_pipeline import run_visual_smoke_test

    run_visual_smoke_test(generate_structured_response, MAX_TOTAL_ATTEMPTS)


if __name__ == "__main__":
    main()
