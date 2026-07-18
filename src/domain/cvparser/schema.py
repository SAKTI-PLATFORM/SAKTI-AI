"""Pydantic schema for LLM CV extraction output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EducationLevel = Literal["SMA", "D3", "S1", "S2", "S3"]


class ParsedPersonalInfo(BaseModel):
    full_name: str | None = None
    professional_headline: str | None = None
    email: str | None = None
    phone_number: str | None = None
    domicile: str | None = None
    linkedin_url: str | None = None
    profile_summary: str | None = None


class ParsedEducation(BaseModel):
    education_level: EducationLevel | None = None
    institution: str | None = None
    major: str | None = None
    degree: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    gpa: float | None = None
    is_current: bool = False


class ParsedExperience(BaseModel):
    title: str
    organization: str
    experience_type: str = "WORK"
    start_date: str | None = None
    end_date: str | None = None
    is_current: bool = False
    duration_months: int | None = None
    description: str | None = None


class ParsedProject(BaseModel):
    project_name: str
    description: str | None = None
    tools_used: str | None = None
    start_date: str | None = None
    end_date: str | None = None


class ParsedCertification(BaseModel):
    certification_name: str
    issuer: str
    issued_year: int | None = None
    certificate_url: str | None = None


class ParsedSkill(BaseModel):
    detected_text: str
    inferred_level: str | None = None
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    evidence_source: str | None = None
    evidence_strength: str | None = None


class ParsedCVResult(BaseModel):
    confidence_score: float = Field(ge=0, le=1)
    personal_info: ParsedPersonalInfo = Field(default_factory=ParsedPersonalInfo)
    educations: list[ParsedEducation] = Field(default_factory=list)
    experiences: list[ParsedExperience] = Field(default_factory=list)
    projects: list[ParsedProject] = Field(default_factory=list)
    certifications: list[ParsedCertification] = Field(default_factory=list)
    skills: list[ParsedSkill] = Field(default_factory=list)

    def to_response_dict(self) -> dict:
        return self.model_dump()
