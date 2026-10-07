from __future__ import annotations

from deterministic import exact_duplicate_key, is_duplicate, is_valid_email, normalize_url
from sales import crm
from sales.lead_finder import mock_leads
from sales.models import Lead, OutreachDraft, QualifiedLead
from sales.outreach import draft_outreach
from sales.qualifier import classify_site, qualify_lead


def dedupe_leads(leads: list[Lead]) -> list[Lead]:
    seen: set[str] = set()
    deduped: list[Lead] = []
    for lead in leads:
        lead.website = normalize_url(lead.website)
        key = exact_duplicate_key(lead.organization_name, lead.website)
        if is_duplicate(key, seen):
            continue
        seen.add(key)
        deduped.append(lead)
    return deduped


def run_demo() -> dict[str, int | str]:
    leads = dedupe_leads(mock_leads())
    suppressed = crm.suppression_values()
    kept: list[Lead] = []
    for lead in leads:
        if lead.contact_email and lead.contact_email.casefold() in suppressed:
            lead.status = "suppressed"
            continue
        if lead.contact_email and not is_valid_email(lead.contact_email):
            lead.status = "rejected"
            continue
        kept.append(lead)
    crm.save_leads(kept)

    qualified: list[QualifiedLead] = []
    drafts: list[OutreachDraft] = []
    for lead in kept:
        classification = classify_site(lead)
        if not classification.is_test_prep_business:
            continue
        lead.status = "classified"
        qualification = qualify_lead(lead)
        crm.save_qualification(qualification)
        if qualification.fit_score < 70:
            continue
        qualified.append(qualification)
        draft = draft_outreach(qualification)
        crm.save_draft(draft)
        drafts.append(draft)

    return {
        "mock_leads": len(leads),
        "deduped_leads": len(kept),
        "leads_qualified": len(qualified),
        "drafts_created": len(drafts),
        "status": "HUMAN_APPROVAL_REQUIRED",
    }
