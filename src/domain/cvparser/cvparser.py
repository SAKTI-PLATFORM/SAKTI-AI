"""Deterministic CV parser used by first-stage job seeker onboarding.

The parser intentionally starts with grounded text heuristics. It gives the
backend a stable contract while leaving room for a later LLM/OCR upgrade behind
the same class methods.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any

from src.infrastructure.llm.openrouter import (
    extract_cv_with_openrouter,
    is_openrouter_configured,
)


SECTION_HEADINGS: dict[str, tuple[str, ...]] = {
    "education": (
        "education",
        "educations",
        "pendidikan",
        "academic background",
        "riwayat pendidikan",
    ),
    "experience": (
        "experience",
        "experiences",
        "work experience",
        "pengalaman",
        "pengalaman kerja",
        "employment",
        "organization",
        "organisasi",
    ),
    "projects": ("project", "projects", "proyek", "portfolio", "portofolio"),
    "certifications": (
        "certification",
        "certifications",
        "certificate",
        "certificates",
        "sertifikasi",
        "lisensi",
    ),
    "skills": ("skills", "skill", "keahlian", "technical skills", "tools"),
}

KNOWN_SKILLS = {
    "python",
    "typescript",
    "javascript",
    "java",
    "golang",
    "react",
    "next.js",
    "nestjs",
    "node.js",
    "fastapi",
    "laravel",
    "mysql",
    "postgresql",
    "redis",
    "docker",
    "aws",
    "figma",
    "excel",
    "power bi",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVSection:
    name: str
    lines: list[str]


class CVParser:
    async def __call__(self, text: str) -> dict[str, Any]:
        return await self.parse_cv(text)

    async def parse_cv(self, text: str) -> dict[str, Any]:
        if is_openrouter_configured():
            try:
                parsed = await extract_cv_with_openrouter(text)
                return parsed.to_response_dict()
            except Exception as exc:
                # Keep onboarding usable during local demos or provider outages.
                logger.warning("OpenRouter CV parsing failed; using fallback parser: %s", exc)

        return self.parse_cv_with_rules(text)

    def parse_cv_with_rules(self, text: str) -> dict[str, Any]:
        normalized_text = self._normalize_text(text)
        lines = self._to_lines(normalized_text)

        educations = self.extract_educations(lines)
        experiences = self.extract_experience(lines)
        projects = self.extract_projects(lines)
        certifications = self.extract_certifications(lines)
        skills = self.extract_skills(lines)

        populated_sections = sum(
            1
            for section in (educations, experiences, projects, certifications, skills)
            if section
        )
        confidence_score = round(0.35 + min(populated_sections, 5) * 0.12, 2)

        return {
            "confidence_score": min(confidence_score, 0.95),
            "educations": educations,
            "experiences": experiences,
            "projects": projects,
            "certifications": certifications,
            "skills": skills,
        }

    def extract_educations(self, lines: list[str]) -> list[dict[str, Any]]:
        section_lines = self._section_lines(lines, "education")
        candidates = section_lines or [
            line for line in lines if self._education_level(line) is not None
        ]

        result: list[dict[str, Any]] = []
        for line in candidates:
            level = self._education_level(line)
            years = self._years(line)
            if level is None and not years:
                continue

            result.append(
                {
                    "education_level": level,
                    "institution": self._clean_record_text(line),
                    "major": self._major(line),
                    "degree": level or self._first_sentence(line),
                    "start_year": years[0] if len(years) > 1 else None,
                    "end_year": years[-1] if years else None,
                    "gpa": self._gpa(line),
                    "is_current": self._is_current(line),
                }
            )
        return result

    def extract_experience(self, lines: list[str]) -> list[dict[str, Any]]:
        section_lines = self._section_lines(lines, "experience")
        candidates = section_lines or [
            line
            for line in lines
            if re.search(r"\b(intern|engineer|developer|manager|staff|analyst)\b", line, re.I)
        ]

        result: list[dict[str, Any]] = []
        for line in candidates:
            title, organization = self._title_and_organization(line)
            if not title and not organization:
                continue

            years = self._years(line)
            result.append(
                {
                    "title": title or "Experience",
                    "organization": organization or "Unknown",
                    "experience_type": self._experience_type(line),
                    "start_date": str(years[0]) if len(years) > 1 else None,
                    "end_date": None if self._is_current(line) else (str(years[-1]) if years else None),
                    "is_current": self._is_current(line),
                    "duration_months": None,
                    "description": self._clean_record_text(line),
                }
            )
        return result

    def extract_projects(self, lines: list[str]) -> list[dict[str, Any]]:
        section_lines = self._section_lines(lines, "projects")
        result: list[dict[str, Any]] = []
        for line in section_lines:
            cleaned = self._clean_record_text(line)
            if len(cleaned) < 3:
                continue
            years = self._years(line)
            result.append(
                {
                    "project_name": self._project_name(cleaned),
                    "description": cleaned,
                    "tools_used": self._tools_used(line),
                    "start_date": str(years[0]) if len(years) > 1 else None,
                    "end_date": str(years[-1]) if years else None,
                }
            )
        return result

    def extract_certifications(self, lines: list[str]) -> list[dict[str, Any]]:
        section_lines = self._section_lines(lines, "certifications")
        result: list[dict[str, Any]] = []
        for line in section_lines:
            cleaned = self._clean_record_text(line)
            if len(cleaned) < 3:
                continue
            years = self._years(line)
            name, issuer = self._certification_name_and_issuer(cleaned)
            result.append(
                {
                    "certification_name": name,
                    "issuer": issuer or "Unknown",
                    "issued_year": years[-1] if years else None,
                    "certificate_url": None,
                }
            )
        return result

    def extract_skills(self, lines: list[str]) -> list[dict[str, Any]]:
        section_lines = self._section_lines(lines, "skills")
        raw = " ".join(section_lines) if section_lines else " ".join(lines)
        detected: set[str] = set()

        for skill in KNOWN_SKILLS:
            if re.search(rf"(?<!\w){re.escape(skill)}(?!\w)", raw, re.I):
                detected.add(self._skill_normalizers(skill))

        for item in re.split(r"[,|;/]", raw):
            normalized = self._skill_normalizers(item)
            if normalized and normalized.lower() in KNOWN_SKILLS:
                detected.add(normalized)

        return [
            {
                "detected_text": skill,
                "inferred_level": None,
                "confidence_score": 0.6,
                "evidence_source": "cv_text",
                "evidence_strength": "medium",
            }
            for skill in sorted(detected)
        ]

    def _skill_normalizers(self, value: str) -> str:
        normalized = re.sub(r"\s+", " ", value.strip().strip("-*•")).strip()
        aliases = {
            "js": "JavaScript",
            "ts": "TypeScript",
            "node": "Node.js",
            "nodejs": "Node.js",
            "nextjs": "Next.js",
            "postgres": "PostgreSQL",
        }
        lowered = normalized.lower()
        if lowered in aliases:
            return aliases[lowered]
        if lowered in {"aws", "sql"}:
            return lowered.upper()
        return normalized.title() if normalized else ""

    def _normalize_text(self, text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _to_lines(self, text: str) -> list[str]:
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _section_lines(self, lines: list[str], section: str) -> list[str]:
        current_section: str | None = None
        collected: list[str] = []
        heading_lookup = {
            heading: key for key, headings in SECTION_HEADINGS.items() for heading in headings
        }

        for line in lines:
            normalized = re.sub(r"[^a-zA-Z ]", "", line).strip().lower()
            matched_section = heading_lookup.get(normalized)
            if matched_section:
                current_section = matched_section
                continue
            if current_section == section:
                collected.append(line)
        return collected

    def _years(self, line: str) -> list[int]:
        return [int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", line)]

    def _education_level(self, line: str) -> str | None:
        patterns = {
            "SMA": r"\b(SMA|SMK|high school)\b",
            "D3": r"\b(D3|diploma)\b",
            "S1": r"\b(S1|bachelor|sarjana)\b",
            "S2": r"\b(S2|master|magister)\b",
            "S3": r"\b(S3|phd|doctor|doktor)\b",
        }
        for level, pattern in patterns.items():
            if re.search(pattern, line, re.I):
                return level
        return None

    def _major(self, line: str) -> str | None:
        match = re.search(r"(?:major|jurusan|program studi)\s*:?\s*([A-Za-z0-9 &.-]+)", line, re.I)
        return match.group(1).strip() if match else None

    def _gpa(self, line: str) -> float | None:
        match = re.search(r"\b(?:gpa|ipk)\s*:?\s*(\d(?:[.,]\d{1,2})?)", line, re.I)
        if not match:
            return None
        return float(match.group(1).replace(",", "."))

    def _is_current(self, line: str) -> bool:
        return bool(re.search(r"\b(present|current|now|sekarang|saat ini)\b", line, re.I))

    def _title_and_organization(self, line: str) -> tuple[str | None, str | None]:
        cleaned = self._clean_record_text(line)
        if " at " in cleaned.lower():
            title, organization = re.split(r"\s+at\s+", cleaned, maxsplit=1, flags=re.I)
            return title.strip(" -"), organization.strip(" -")
        if " - " in cleaned:
            first, second = cleaned.split(" - ", 1)
            return first.strip(), second.strip()
        return cleaned, None

    def _experience_type(self, line: str) -> str:
        if re.search(r"\b(intern|magang)\b", line, re.I):
            return "INTERNSHIP"
        if re.search(r"\b(organization|organisasi|volunteer)\b", line, re.I):
            return "ORGANIZATION"
        return "WORK"

    def _project_name(self, line: str) -> str:
        return re.split(r"\s[-:]\s", line, maxsplit=1)[0][:255]

    def _tools_used(self, line: str) -> str | None:
        match = re.search(r"(?:tools?|tech stack|stack)\s*:?\s*([A-Za-z0-9, .+/&-]+)", line, re.I)
        return match.group(1).strip() if match else None

    def _certification_name_and_issuer(self, line: str) -> tuple[str, str | None]:
        if " - " in line:
            name, issuer = line.split(" - ", 1)
            return name.strip(), issuer.strip()
        match = re.search(r"(.+?)\s+(?:by|oleh|from)\s+(.+)", line, re.I)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return line, None

    def _clean_record_text(self, line: str) -> str:
        return re.sub(r"^\s*[-*•]\s*", "", line).strip()

    def _first_sentence(self, line: str) -> str:
        return re.split(r"[.;]", self._clean_record_text(line), maxsplit=1)[0][:255]
