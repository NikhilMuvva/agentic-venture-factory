from __future__ import annotations

from model_router import TaskType, route_task
from sales.models import Lead, OutreachDraft, OutreachDraftResponse, QualifiedLead


def draft_outreach(qualified: QualifiedLead) -> OutreachDraft:
    lead: Lead = qualified.lead
    prompt = f"""Draft a concise first-touch sales email for TestForge.
Do not claim prior contact. Do not create fake facts. Do not include unsubscribe language yet because this is a human-review draft only.
Lead: {lead.organization_name}
Website: {lead.website}
Organization type: {lead.organization_type}
Exams: {', '.join(lead.exams)}
Fit score: {qualified.fit_score}
Reasons: {'; '.join(qualified.reasons)}
Recommended offer: {qualified.recommended_offer}"""
    response, model = route_task(TaskType.OUTREACH_DRAFT, prompt, OutreachDraftResponse, "sales", "outreach_drafter", lead.id)
    return OutreachDraft(
        lead_id=lead.id,
        subject=response.subject,
        body=response.body,
        personalization_basis=response.personalization_basis,
        model_used=model,
        human_approved=False,
    )
