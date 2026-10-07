from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LeadStatus = Literal["new", "classified", "qualified", "drafted", "suppressed", "rejected"]


class Lead(BaseModel):
    id: str
    organization_name: str
    website: str
    contact_name: str | None = None
    contact_email: str | None = None
    organization_type: str
    exams: list[str] = Field(default_factory=list)
    location: str | None = None
    source: str
    discovered_at: str
    status: LeadStatus = "new"


class QualifiedLead(BaseModel):
    lead: Lead
    fit_score: int = Field(ge=0, le=100)
    fit: Literal["high", "medium", "low"]
    reasons: list[str]
    recommended_offer: str
    qualification_model: str
    qualification_cost: float = 0.0


class OutreachDraft(BaseModel):
    lead_id: str
    subject: str
    body: str
    personalization_basis: str
    model_used: str
    human_approved: bool = False


class OutreachDraftResponse(BaseModel):
    subject: str
    body: str
    personalization_basis: str
    confidence: float = Field(ge=0, le=1)
