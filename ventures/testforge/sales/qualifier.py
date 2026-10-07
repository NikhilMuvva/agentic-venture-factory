from __future__ import annotations

from model_router import LeadQualification, SiteClassification, TaskType, route_task
from sales.models import Lead, QualifiedLead


def classify_site(lead: Lead) -> SiteClassification:
    prompt = f"""Classify whether this organization is a test-prep or tutoring business.
Organization: {lead.organization_name}
Website: {lead.website}
Type hint: {lead.organization_type}
Known exams: {', '.join(lead.exams)}
Return exams only when supported by the evidence."""
    result, _ = route_task(TaskType.SITE_CLASSIFICATION, prompt, SiteClassification, "sales", "site_classifier", lead.id)
    return result


def qualify_lead(lead: Lead) -> QualifiedLead:
    prompt = f"""Qualify this organization as a potential customer for TestForge, an autonomous SAT-style question generation and QA system.
Organization: {lead.organization_name}
Website: {lead.website}
Type: {lead.organization_type}
Exams: {', '.join(lead.exams)}
Location: {lead.location or 'unknown'}
Give a conservative fit score and specific reasons."""
    result, model = route_task(TaskType.LEAD_QUALIFICATION, prompt, LeadQualification, "sales", "lead_qualifier", lead.id)
    return QualifiedLead(
        lead=lead,
        fit_score=result.fit_score,
        fit=result.fit,  # type: ignore[arg-type]
        reasons=result.reasons,
        recommended_offer=result.recommended_offer,
        qualification_model=model,
        qualification_cost=0.0,
    )
