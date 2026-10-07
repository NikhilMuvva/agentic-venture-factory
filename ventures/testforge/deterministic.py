from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal
from urllib.parse import urlparse, urlunparse


EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.IGNORECASE)


def normalize_url(url: str) -> str:
    value = url.strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), host, path, "", "", ""))


def exact_duplicate_key(*parts: str | None) -> str:
    return "|".join((part or "").strip().casefold() for part in parts)


def is_duplicate(key: str, seen_keys: set[str]) -> bool:
    return key in seen_keys


def has_already_contacted(lead_id: str, contacted_ids: set[str]) -> bool:
    return lead_id in contacted_ids


def is_valid_email(email: str | None) -> bool:
    return bool(email and EMAIL_RE.match(email.strip()))


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def estimated_cost(input_tokens: int, output_tokens: int, input_per_million: float, output_per_million: float) -> float:
    cost = (Decimal(input_tokens) * Decimal(str(input_per_million)) / Decimal(1_000_000))
    cost += Decimal(output_tokens) * Decimal(str(output_per_million)) / Decimal(1_000_000)
    return float(cost)
