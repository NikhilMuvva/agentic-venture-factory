from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from deterministic import estimated_cost, utc_timestamp
from models import APIInfrastructureFailure


load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
USAGE_LOG_PATH = DATA_DIR / "model_usage.jsonl"
PRICE_PATH = ROOT / "config" / "model_prices.json"


class TaskType(str, Enum):
    SITE_CLASSIFICATION = "SITE_CLASSIFICATION"
    FIELD_EXTRACTION = "FIELD_EXTRACTION"
    LEAD_QUALIFICATION = "LEAD_QUALIFICATION"
    OUTREACH_PERSONALIZATION = "OUTREACH_PERSONALIZATION"
    OUTREACH_DRAFT = "OUTREACH_DRAFT"
    REPLY_CLASSIFICATION = "REPLY_CLASSIFICATION"
    REPLY_EXTRACTION = "REPLY_EXTRACTION"
    REPLY_DRAFT = "REPLY_DRAFT"
    BRANCH_ANALYSIS = "BRANCH_ANALYSIS"
    QUESTION_GENERATION = "QUESTION_GENERATION"
    QUESTION_SOLVING = "QUESTION_SOLVING"
    QUESTION_REVIEW = "QUESTION_REVIEW"


class ModelTier(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    ULTRA_CHEAP = "ULTRA_CHEAP"
    CHEAP = "CHEAP"
    QUALITY = "QUALITY"
    ESCALATION = "ESCALATION"


class ProviderName(str, Enum):
    QWEN = "qwen"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    LOCAL = "local"


class BudgetReviewRequired(RuntimeError):
    reason = "BUDGET_REVIEW_REQUIRED"


class SiteClassification(BaseModel):
    is_test_prep_business: bool
    exams: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class LeadQualification(BaseModel):
    fit_score: int = Field(ge=0, le=100)
    fit: str = Field(pattern="^(high|medium|low)$")
    reasons: list[str]
    recommended_offer: str
    confidence: float = Field(ge=0, le=1)


class ReplyClassification(BaseModel):
    category: str = Field(pattern="^(INTERESTED|NOT_INTERESTED|QUESTION|UNSUBSCRIBE|OUT_OF_OFFICE|OTHER)$")
    confidence: float = Field(ge=0, le=1)


@dataclass(frozen=True)
class ModelChoice:
    provider: ProviderName
    model: str


@dataclass(frozen=True)
class TaskPolicy:
    tier: ModelTier
    primary: ModelChoice
    fallback: ModelChoice | None = None
    escalation: ModelChoice | None = None
    confidence_threshold: float = 0.8
    max_output_tokens: int = 512
    batch_eligible: bool = False
    cheap_first_pass: ModelChoice | None = None


@dataclass
class ModelCallResult:
    parsed: BaseModel
    provider: ProviderName
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    retry_count: int = 0
    escalated: bool = False


class Provider(Protocol):
    name: ProviderName

    def available(self) -> bool:
        ...

    def generate_structured(
        self,
        prompt: Any,
        response_model: type[BaseModel],
        model: str,
        max_output_tokens: int,
    ) -> tuple[BaseModel, int, int]:
        ...

    def probe(self, model: str) -> str:
        ...


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


QWEN_FLASH = _env("QWEN_FLASH_MODEL", "qwen-flash")
QWEN_FLASH_PLUS = _env("QWEN_FALLBACK_MODEL", "qwen3.7-flash")
GEMINI_FLASH_LITE = _env("GEMINI_FLASH_LITE_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACKS = tuple(
    item.strip()
    for item in _env("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-3.8-flash,gemini-3.6-flash").split(",")
    if item.strip()
)
DEEPSEEK_FLASH = _env("DEEPSEEK_MODEL", "deepseek-flash")
LOCAL_MODEL = _env("LOCAL_LLM_MODEL", "local-qwen")

POLICIES: dict[TaskType, TaskPolicy] = {
    TaskType.SITE_CLASSIFICATION: TaskPolicy(ModelTier.ULTRA_CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH), ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), max_output_tokens=160, batch_eligible=True),
    TaskType.FIELD_EXTRACTION: TaskPolicy(ModelTier.ULTRA_CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH), ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), max_output_tokens=256, batch_eligible=True),
    TaskType.REPLY_CLASSIFICATION: TaskPolicy(ModelTier.ULTRA_CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH), ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), max_output_tokens=128),
    TaskType.REPLY_EXTRACTION: TaskPolicy(ModelTier.ULTRA_CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH), ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), max_output_tokens=256),
    TaskType.LEAD_QUALIFICATION: TaskPolicy(ModelTier.CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), batch_eligible=True),
    TaskType.OUTREACH_PERSONALIZATION: TaskPolicy(ModelTier.CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE)),
    TaskType.OUTREACH_DRAFT: TaskPolicy(ModelTier.CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), batch_eligible=False),
    TaskType.REPLY_DRAFT: TaskPolicy(ModelTier.QUALITY, ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), escalation=ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), cheap_first_pass=ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS)),
    TaskType.BRANCH_ANALYSIS: TaskPolicy(ModelTier.QUALITY, ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE)),
    TaskType.QUESTION_GENERATION: TaskPolicy(ModelTier.QUALITY, ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), ModelChoice(ProviderName.GEMINI, GEMINI_FALLBACKS[0] if GEMINI_FALLBACKS else GEMINI_FLASH_LITE), max_output_tokens=4096),
    TaskType.QUESTION_SOLVING: TaskPolicy(ModelTier.CHEAP, ModelChoice(ProviderName.QWEN, QWEN_FLASH_PLUS), ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE)),
    TaskType.QUESTION_REVIEW: TaskPolicy(ModelTier.QUALITY, ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE), ModelChoice(ProviderName.GEMINI, GEMINI_FALLBACKS[0] if GEMINI_FALLBACKS else GEMINI_FLASH_LITE), max_output_tokens=2048),
}


def task_policy(task_type: TaskType) -> TaskPolicy:
    return POLICIES[task_type]


def _has_binary_parts(prompt: Any) -> bool:
    return isinstance(prompt, list) and any(hasattr(item, "inline_data") or hasattr(item, "data") for item in prompt)


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        chunks: list[str] = []
        for item in prompt:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)
    return str(prompt)


def _token_estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _safe_json_text(response_text: str) -> str:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


class OpenAICompatibleProvider:
    def __init__(self, name: ProviderName, api_key_env: str, base_url_env: str, default_base_url: str | None = None) -> None:
        self.name = name
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.default_base_url = default_base_url

    def available(self) -> bool:
        if self.name == ProviderName.LOCAL:
            return bool(os.getenv(self.base_url_env))
        return bool(os.getenv(self.api_key_env) and (os.getenv(self.base_url_env) or self.default_base_url))

    def _client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("The official openai package is required for OpenAI-compatible providers.") from error
        api_key = os.getenv(self.api_key_env)
        base_url = os.getenv(self.base_url_env) or self.default_base_url
        if self.name == ProviderName.LOCAL and base_url and not api_key:
            api_key = "local"
        if not api_key or not base_url:
            raise RuntimeError(f"{self.name.value} provider is not configured.")
        return OpenAI(api_key=api_key.strip(), base_url=base_url.strip())

    def generate_structured(self, prompt: Any, response_model: type[BaseModel], model: str, max_output_tokens: int) -> tuple[BaseModel, int, int]:
        text = _prompt_text(prompt)
        schema = response_model.model_json_schema()
        response = self._client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the supplied JSON schema."},
                {"role": "user", "content": f"JSON schema:\n{json.dumps(schema)}\n\nTask:\n{text}"},
            ],
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        content = response.choices[0].message.content or ""
        parsed = response_model.model_validate_json(_safe_json_text(content))
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or _token_estimate(text))
        output_tokens = int(getattr(usage, "completion_tokens", 0) or _token_estimate(content))
        return parsed, input_tokens, output_tokens

    def probe(self, model: str) -> str:
        response = self._client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply exactly OK"}],
            temperature=0,
            max_tokens=8,
        )
        return (response.choices[0].message.content or "").strip()


class GeminiProvider:
    name = ProviderName.GEMINI

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def _client(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY was not found.")
        return genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def generate_structured(self, prompt: Any, response_model: type[BaseModel], model: str, max_output_tokens: int) -> tuple[BaseModel, int, int]:
        response = self._client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_model,
                max_output_tokens=max_output_tokens,
            ),
        )
        if response.parsed is not None:
            parsed = response_model.model_validate(response.parsed) if not isinstance(response.parsed, str) else response_model.model_validate_json(response.parsed)
        elif response.text:
            parsed = response_model.model_validate_json(response.text)
        else:
            raise ValueError("Gemini returned an empty response.")
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or _token_estimate(_prompt_text(prompt)))
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or _token_estimate(response.text or parsed.model_dump_json()))
        return parsed, input_tokens, output_tokens

    def probe(self, model: str) -> str:
        response = self._client().models.generate_content(model=model, contents="Reply exactly OK")
        return (response.text or "").strip()


def default_providers() -> dict[ProviderName, Provider]:
    return {
        ProviderName.QWEN: OpenAICompatibleProvider(ProviderName.QWEN, "DASHSCOPE_API_KEY", "QWEN_BASE_URL"),
        ProviderName.GEMINI: GeminiProvider(),
        ProviderName.DEEPSEEK: OpenAICompatibleProvider(ProviderName.DEEPSEEK, "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ProviderName.LOCAL: OpenAICompatibleProvider(ProviderName.LOCAL, "LOCAL_LLM_API_KEY", "LOCAL_LLM_BASE_URL"),
    }


def load_price_registry() -> dict[str, Any]:
    with PRICE_PATH.open("r", encoding="utf-8") as price_file:
        return json.load(price_file)


def model_price(model: str) -> tuple[float, float]:
    registry = load_price_registry()["models"]
    item = registry.get(model, {"input_per_million": 0.0, "output_per_million": 0.0})
    return float(item["input_per_million"]), float(item["output_per_million"])


def current_daily_spend(day: str | None = None, usage_log_path: Path = USAGE_LOG_PATH) -> float:
    if day is None:
        day = utc_timestamp()[:10]
    if not usage_log_path.exists():
        return 0.0
    total = 0.0
    with usage_log_path.open("r", encoding="utf-8") as usage_file:
        for line in usage_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("timestamp", "")).startswith(day):
                total += float(record.get("estimated_cost_usd", 0.0))
    return total


def daily_budget() -> float:
    return float(os.getenv("DAILY_AI_BUDGET_USD", "0.25"))


def append_usage(record: dict[str, Any], usage_log_path: Path = USAGE_LOG_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in record.items() if "key" not in key.lower() and "credential" not in key.lower()}
    with usage_log_path.open("a", encoding="utf-8") as usage_file:
        usage_file.write(json.dumps(clean, ensure_ascii=False) + "\n")


def _confidence(parsed: BaseModel) -> float | None:
    value = getattr(parsed, "confidence", None)
    return float(value) if value is not None else None


def _choice_sequence(policy: TaskPolicy, prompt: Any) -> list[ModelChoice]:
    choices = [policy.primary]
    if policy.fallback and policy.fallback not in choices:
        choices.append(policy.fallback)
    if _has_binary_parts(prompt):
        choices = [choice for choice in choices if choice.provider == ProviderName.GEMINI]
        if not choices:
            choices = [ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE)]
    return choices


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


@dataclass
class ModelRouter:
    providers: dict[ProviderName, Provider] = field(default_factory=default_providers)
    usage_log_path: Path = USAGE_LOG_PATH

    def choose_model(self, task_type: TaskType, prompt: Any = "") -> ModelChoice:
        policy = task_policy(task_type)
        choices = _choice_sequence(policy, prompt)
        return choices[0]

    def route(
        self,
        task_type: TaskType,
        prompt: Any,
        response_model: type[ResponseModel],
        department: str,
        agent: str,
        object_id: str | None = None,
    ) -> ModelCallResult:
        policy = task_policy(task_type)
        retry_count = 0
        escalated = False
        last_error: Exception | None = None
        choices = _choice_sequence(policy, prompt)

        for index, choice in enumerate(choices):
            provider = self.providers[choice.provider]
            if not provider.available():
                last_error = RuntimeError(f"{choice.provider.value} provider is not configured.")
                continue
            for attempt in range(2):
                start = time.monotonic()
                input_estimate = _token_estimate(_prompt_text(prompt))
                output_estimate = policy.max_output_tokens
                in_price, out_price = model_price(choice.model)
                potential_cost = estimated_cost(input_estimate, output_estimate, in_price, out_price)
                if current_daily_spend(usage_log_path=self.usage_log_path) + potential_cost > daily_budget():
                    raise BudgetReviewRequired("BUDGET_REVIEW_REQUIRED")
                try:
                    parsed, input_tokens, output_tokens = provider.generate_structured(prompt, response_model, choice.model, policy.max_output_tokens)
                    cost = estimated_cost(input_tokens, output_tokens, in_price, out_price)
                    latency_ms = int((time.monotonic() - start) * 1000)
                    confidence = _confidence(parsed)
                    should_escalate = (
                        confidence is not None
                        and confidence < policy.confidence_threshold
                        and policy.escalation is not None
                        and choice != policy.escalation
                    )
                    self._log(department, agent, task_type, choice, input_tokens, output_tokens, cost, latency_ms, True, retry_count, escalated, object_id)
                    if should_escalate:
                        choices.append(policy.escalation)
                        escalated = True
                        break
                    return ModelCallResult(parsed, choice.provider, choice.model, input_tokens, output_tokens, cost, latency_ms, retry_count, escalated)
                except (ValidationError, ValueError, json.JSONDecodeError) as error:
                    last_error = error
                    retry_count += 1
                    self._log(department, agent, task_type, choice, input_estimate, 0, 0.0, int((time.monotonic() - start) * 1000), False, retry_count, escalated, object_id)
                    if attempt == 0:
                        continue
                    break
                except BudgetReviewRequired:
                    raise
                except Exception as error:
                    last_error = error
                    self._log(department, agent, task_type, choice, input_estimate, 0, 0.0, int((time.monotonic() - start) * 1000), False, retry_count, escalated, object_id)
                    break
            if index == len(choices) - 1 and policy.escalation and choice != policy.escalation:
                choices.append(policy.escalation)
                escalated = True

        message = str(last_error or "All configured providers failed.")
        raise APIInfrastructureFailure("MODEL_ROUTER_UNAVAILABLE", message, agent, choices[-1].model if choices else None) from last_error

    def _log(
        self,
        department: str,
        agent: str,
        task_type: TaskType,
        choice: ModelChoice,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: int,
        success: bool,
        retry_count: int,
        escalated: bool,
        object_id: str | None,
    ) -> None:
        append_usage({
            "timestamp": utc_timestamp(),
            "department": department,
            "agent": agent,
            "task_type": task_type.value,
            "provider": choice.provider.value,
            "model": choice.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "latency_ms": latency_ms,
            "success": success,
            "retry_count": retry_count,
            "escalated": escalated,
            "object_id": object_id,
        }, self.usage_log_path)

    def probe(self, provider_name: ProviderName) -> tuple[bool, str]:
        choice = {
            ProviderName.QWEN: ModelChoice(ProviderName.QWEN, QWEN_FLASH),
            ProviderName.GEMINI: ModelChoice(ProviderName.GEMINI, GEMINI_FLASH_LITE),
            ProviderName.DEEPSEEK: ModelChoice(ProviderName.DEEPSEEK, DEEPSEEK_FLASH),
            ProviderName.LOCAL: ModelChoice(ProviderName.LOCAL, LOCAL_MODEL),
        }[provider_name]
        provider = self.providers[provider_name]
        if not provider.available():
            return False, "not configured"
        try:
            text = provider.probe(choice.model)
            return text == "OK", text
        except Exception as error:
            return False, str(error)[:400]


router = ModelRouter()


def route_task(
    task_type: TaskType,
    prompt: Any,
    response_model: type[ResponseModel],
    department: str,
    agent: str,
    object_id: str | None = None,
) -> tuple[ResponseModel, str]:
    result = router.route(task_type, prompt, response_model, department, agent, object_id)
    return response_model.model_validate(result.parsed), result.model


def production_task_from_stage(stage: str) -> TaskType:
    normalized = stage.lower()
    if "solver" in normalized:
        return TaskType.QUESTION_SOLVING
    if "review" in normalized:
        return TaskType.QUESTION_REVIEW
    return TaskType.QUESTION_GENERATION
