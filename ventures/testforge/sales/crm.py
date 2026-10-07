from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from sales.models import Lead, OutreachDraft, QualifiedLead


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "sales.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id TEXT PRIMARY KEY,
    organization_name TEXT NOT NULL,
    website TEXT NOT NULL,
    contact_name TEXT,
    contact_email TEXT,
    organization_type TEXT NOT NULL,
    exams TEXT NOT NULL,
    location TEXT,
    source TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS qualifications (
    lead_id TEXT PRIMARY KEY,
    fit_score INTEGER NOT NULL,
    fit TEXT NOT NULL,
    reasons TEXT NOT NULL,
    recommended_offer TEXT NOT NULL,
    qualification_model TEXT NOT NULL,
    qualification_cost REAL NOT NULL,
    FOREIGN KEY (lead_id) REFERENCES leads(id)
);

CREATE TABLE IF NOT EXISTS outreach_drafts (
    lead_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    personalization_basis TEXT NOT NULL,
    model_used TEXT NOT NULL,
    human_approved INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (lead_id) REFERENCES leads(id)
);

CREATE TABLE IF NOT EXISTS contact_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    notes TEXT,
    FOREIGN KEY (lead_id) REFERENCES leads(id)
);

CREATE TABLE IF NOT EXISTS suppression_list (
    value TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def save_leads(leads: Iterable[Lead], path: Path = DB_PATH) -> None:
    with connect(path) as connection:
        connection.executemany(
            """
            INSERT INTO leads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                organization_name=excluded.organization_name,
                website=excluded.website,
                contact_name=excluded.contact_name,
                contact_email=excluded.contact_email,
                organization_type=excluded.organization_type,
                exams=excluded.exams,
                location=excluded.location,
                source=excluded.source,
                discovered_at=excluded.discovered_at,
                status=excluded.status
            """,
            [
                (
                    lead.id,
                    lead.organization_name,
                    lead.website,
                    lead.contact_name,
                    lead.contact_email,
                    lead.organization_type,
                    json.dumps(lead.exams),
                    lead.location,
                    lead.source,
                    lead.discovered_at,
                    lead.status,
                )
                for lead in leads
            ],
        )


def save_qualification(qualified: QualifiedLead, path: Path = DB_PATH) -> None:
    with connect(path) as connection:
        connection.execute(
            "UPDATE leads SET status = ? WHERE id = ?",
            ("qualified", qualified.lead.id),
        )
        connection.execute(
            """
            INSERT INTO qualifications VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(lead_id) DO UPDATE SET
                fit_score=excluded.fit_score,
                fit=excluded.fit,
                reasons=excluded.reasons,
                recommended_offer=excluded.recommended_offer,
                qualification_model=excluded.qualification_model,
                qualification_cost=excluded.qualification_cost
            """,
            (
                qualified.lead.id,
                qualified.fit_score,
                qualified.fit,
                json.dumps(qualified.reasons),
                qualified.recommended_offer,
                qualified.qualification_model,
                qualified.qualification_cost,
            ),
        )


def save_draft(draft: OutreachDraft, path: Path = DB_PATH) -> None:
    with connect(path) as connection:
        connection.execute("UPDATE leads SET status = ? WHERE id = ?", ("drafted", draft.lead_id))
        connection.execute(
            """
            INSERT INTO outreach_drafts VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(lead_id) DO UPDATE SET
                subject=excluded.subject,
                body=excluded.body,
                personalization_basis=excluded.personalization_basis,
                model_used=excluded.model_used,
                human_approved=excluded.human_approved
            """,
            (
                draft.lead_id,
                draft.subject,
                draft.body,
                draft.personalization_basis,
                draft.model_used,
                int(draft.human_approved),
            ),
        )


def suppression_values(path: Path = DB_PATH) -> set[str]:
    with connect(path) as connection:
        rows = connection.execute("SELECT value FROM suppression_list").fetchall()
    return {str(row["value"]).casefold() for row in rows}
