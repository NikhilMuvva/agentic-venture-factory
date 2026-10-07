from __future__ import annotations

from deterministic import utc_timestamp
from sales.models import Lead


def mock_leads() -> list[Lead]:
    discovered_at = utc_timestamp()
    return [
        Lead(id="MOCK-001", organization_name="Northstar Test Prep", website="https://northstarprep.example", contact_email="hello@northstarprep.example", organization_type="test_prep", exams=["SAT", "ACT"], location="Austin, TX", source="mock", discovered_at=discovered_at, status="new"),
        Lead(id="MOCK-002", organization_name="Summit STEM Tutoring", website="https://summitstem.example", contact_email="info@summitstem.example", organization_type="tutoring", exams=["SAT"], location="Denver, CO", source="mock", discovered_at=discovered_at, status="new"),
        Lead(id="MOCK-003", organization_name="Metro College Counseling", website="https://metrocollege.example", contact_email="team@metrocollege.example", organization_type="college_counseling", exams=["SAT", "AP"], location="Chicago, IL", source="mock", discovered_at=discovered_at, status="new"),
        Lead(id="MOCK-004", organization_name="BrightPath Learning Center", website="https://brightpath.example", contact_email="admin@brightpath.example", organization_type="learning_center", exams=["ACT"], location="Raleigh, NC", source="mock", discovered_at=discovered_at, status="new"),
        Lead(id="MOCK-005", organization_name="Evergreen Language School", website="https://evergreenlanguage.example", contact_email="contact@evergreenlanguage.example", organization_type="language_school", exams=["TOEFL"], location="Portland, OR", source="mock", discovered_at=discovered_at, status="new"),
    ]
