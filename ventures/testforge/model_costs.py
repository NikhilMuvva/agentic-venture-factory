from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from deterministic import utc_timestamp
from model_router import USAGE_LOG_PATH


def main() -> None:
    today = utc_timestamp()[:10]
    calls = defaultdict(int)
    tokens_in = 0
    tokens_out = 0
    spend_by_provider = defaultdict(float)
    spend_by_department = defaultdict(float)
    spend_by_task = defaultdict(float)

    path = Path(USAGE_LOG_PATH)
    if path.exists():
        with path.open("r", encoding="utf-8") as usage_file:
            for line in usage_file:
                record = json.loads(line)
                if not str(record.get("timestamp", "")).startswith(today):
                    continue
                provider = record.get("provider", "unknown")
                calls[provider] += 1
                tokens_in += int(record.get("input_tokens", 0))
                tokens_out += int(record.get("output_tokens", 0))
                cost = float(record.get("estimated_cost_usd", 0.0))
                spend_by_provider[provider] += cost
                spend_by_department[record.get("department", "unknown")] += cost
                spend_by_task[record.get("task_type", "unknown")] += cost

    print("TESTFORGE MODEL COST REPORT")
    print()
    print("Today:")
    print(f"Qwen calls: {calls['qwen']}")
    print(f"Gemini calls: {calls['gemini']}")
    print(f"DeepSeek calls: {calls['deepseek']}")
    print()
    print(f"Input tokens: {tokens_in}")
    print(f"Output tokens: {tokens_out}")
    print()
    print("Estimated spend:")
    print(f"Qwen: ${spend_by_provider['qwen']:.6f}")
    print(f"Gemini: ${spend_by_provider['gemini']:.6f}")
    print(f"DeepSeek: ${spend_by_provider['deepseek']:.6f}")
    print(f"TOTAL: ${sum(spend_by_provider.values()):.6f}")
    print()
    print("Cost by department:")
    for department, cost in sorted(spend_by_department.items()):
        print(f"{department.title()}: ${cost:.6f}")
    print()
    print("Cost by task:")
    for task, cost in sorted(spend_by_task.items()):
        print(f"{task.replace('_', ' ').title()}: ${cost:.6f}")


if __name__ == "__main__":
    main()
