from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from deterministic import is_valid_email, normalize_url
from model_router import (
    BudgetReviewRequired,
    ModelChoice,
    ModelRouter,
    ProviderName,
    TaskType,
    task_policy,
)


class DummyResult(BaseModel):
    answer: str = "ok"
    confidence: float = Field(ge=0, le=1)


class FakeProvider:
    def __init__(self, name: ProviderName, responses: list[Any], configured: bool = True) -> None:
        self.name = name
        self.responses = responses
        self.configured = configured
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return self.configured

    def generate_structured(self, prompt: Any, response_model: type[BaseModel], model: str, max_output_tokens: int):
        self.calls.append((model, str(prompt)))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response_model.model_validate(response), 100, 20

    def probe(self, model: str) -> str:
        return "OK"


class ModelRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.usage_log = Path(self.tmpdir.name) / "usage.jsonl"
        self.old_budget = os.environ.get("DAILY_AI_BUDGET_USD")
        os.environ["DAILY_AI_BUDGET_USD"] = "100"

    def tearDown(self) -> None:
        if self.old_budget is None:
            os.environ.pop("DAILY_AI_BUDGET_USD", None)
        else:
            os.environ["DAILY_AI_BUDGET_USD"] = self.old_budget
        self.tmpdir.cleanup()

    def test_deterministic_utilities_make_no_model_calls(self) -> None:
        qwen = FakeProvider(ProviderName.QWEN, [{"confidence": 0.9}])
        router = ModelRouter({ProviderName.QWEN: qwen}, self.usage_log)
        self.assertEqual(normalize_url("WWW.Example.com/path/"), "https://example.com/path")
        self.assertTrue(is_valid_email("team@example.com"))
        self.assertEqual(qwen.calls, [])
        self.assertFalse(self.usage_log.exists())
        self.assertIsInstance(router, ModelRouter)

    def test_qwen_selected_for_cheap_task(self) -> None:
        qwen = FakeProvider(ProviderName.QWEN, [{"answer": "ok", "confidence": 0.95}])
        router = ModelRouter({ProviderName.QWEN: qwen}, self.usage_log)
        result = router.route(TaskType.LEAD_QUALIFICATION, "lead", DummyResult, "sales", "qualifier", "L1")
        self.assertEqual(result.provider, ProviderName.QWEN)
        self.assertEqual(qwen.calls[0][0], task_policy(TaskType.LEAD_QUALIFICATION).primary.model)

    def test_gemini_selected_for_quality_task(self) -> None:
        gemini = FakeProvider(ProviderName.GEMINI, [{"answer": "ok", "confidence": 0.95}])
        router = ModelRouter({ProviderName.GEMINI: gemini}, self.usage_log)
        result = router.route(TaskType.QUESTION_GENERATION, "question", DummyResult, "production", "generator", "C1")
        self.assertEqual(result.provider, ProviderName.GEMINI)

    def test_low_confidence_escalates(self) -> None:
        qwen = FakeProvider(ProviderName.QWEN, [{"answer": "weak", "confidence": 0.4}])
        gemini = FakeProvider(ProviderName.GEMINI, [{"answer": "strong", "confidence": 0.95}])
        router = ModelRouter({ProviderName.QWEN: qwen, ProviderName.GEMINI: gemini}, self.usage_log)
        result = router.route(TaskType.LEAD_QUALIFICATION, "lead", DummyResult, "sales", "qualifier", "L2")
        self.assertEqual(result.provider, ProviderName.GEMINI)
        self.assertTrue(result.escalated)
        self.assertEqual(len(qwen.calls), 1)
        self.assertEqual(len(gemini.calls), 1)

    def test_malformed_json_retries_once_then_escalates(self) -> None:
        qwen = FakeProvider(ProviderName.QWEN, [ValueError("malformed"), ValueError("still malformed")])
        gemini = FakeProvider(ProviderName.GEMINI, [{"answer": "fixed", "confidence": 0.95}])
        router = ModelRouter({ProviderName.QWEN: qwen, ProviderName.GEMINI: gemini}, self.usage_log)
        result = router.route(TaskType.LEAD_QUALIFICATION, "lead", DummyResult, "sales", "qualifier", "L3")
        self.assertEqual(result.provider, ProviderName.GEMINI)
        self.assertEqual(len(qwen.calls), 2)

    def test_cost_logging_and_budget_guard(self) -> None:
        qwen = FakeProvider(ProviderName.QWEN, [{"answer": "ok", "confidence": 0.95}])
        router = ModelRouter({ProviderName.QWEN: qwen}, self.usage_log)
        router.route(TaskType.LEAD_QUALIFICATION, "lead", DummyResult, "sales", "qualifier", "L4")
        record = json.loads(self.usage_log.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["provider"], "qwen")
        self.assertGreater(record["estimated_cost_usd"], 0)

        os.environ["DAILY_AI_BUDGET_USD"] = "0.00000001"
        with self.assertRaises(BudgetReviewRequired):
            router.route(TaskType.LEAD_QUALIFICATION, "lead", DummyResult, "sales", "qualifier", "L5")

    def test_api_keys_never_appear_in_logs(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "SECRET_SHOULD_NOT_LOG"
        qwen = FakeProvider(ProviderName.QWEN, [{"answer": "ok", "confidence": 0.95}])
        router = ModelRouter({ProviderName.QWEN: qwen}, self.usage_log)
        router.route(TaskType.LEAD_QUALIFICATION, "lead SECRET_SHOULD_NOT_LOG", DummyResult, "sales", "qualifier", "L6")
        self.assertNotIn("SECRET_SHOULD_NOT_LOG", self.usage_log.read_text(encoding="utf-8"))

    def test_task_policy_batch_flags(self) -> None:
        self.assertTrue(task_policy(TaskType.SITE_CLASSIFICATION).batch_eligible)
        self.assertTrue(task_policy(TaskType.LEAD_QUALIFICATION).batch_eligible)
        self.assertFalse(task_policy(TaskType.OUTREACH_DRAFT).batch_eligible)


if __name__ == "__main__":
    unittest.main()
