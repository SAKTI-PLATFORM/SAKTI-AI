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

from src.infrastructure.llm.deepseek import (
    extract_cv_with_deepseek,
    is_deepseek_configured,
)


SECTION_HEADINGS: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary",
        "profile",
        "professional summary",
        "profile summary",
        "about",
        "about me",
        "ringkasan",
        "profil",
        "tentang saya",
    ),
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

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class CVSection:
    name: str
    lines: list[str]


class CVParser:
    async def __call__(self, text: str) -> dict[str, Any]:
        return await self.parse_cv(text)

    async def parse_cv(self, text: str) -> dict[str, Any]:
        deepseek_configured = is_deepseek_configured()
        logger.info(
            "[CVParser] Starting parse: text_length=%d deepseek_configured=%s",
            len(text),
            deepseek_configured,
        )

        if deepseek_configured:
            try:
                parsed = await extract_cv_with_deepseek(text)
                logger.info("[CVParser] Parse completed using DeepSeek")
                return parsed.to_response_dict()
            except Exception as exc:
                # Keep onboarding usable during local demos or provider outages.
                logger.warning(
                    "[CVParser] DeepSeek parsing failed; using fallback parser: %s",
                    exc,
                    exc_info=True,
                )
        else:
            logger.info(
                "[CVParser] DEEPSEEK_API_KEY is not configured; using fallback parser"
            )

        result = self.parse_cv_with_rules(text)
        logger.info(
            (
                "[CVParser] Fallback parse completed: confidence=%.2f "
                "educations=%d experiences=%d projects=%d certifications=%d skills=%d"
            ),
            result["confidence_score"],
            len(result["educations"]),
            len(result["experiences"]),
            len(result["projects"]),
            len(result["certifications"]),
            len(result["skills"]),
        )
        return result

    def parse_cv_with_rules(self, text: str) -> dict[str, Any]:
        normalized_text = self._normalize_text(text)
        lines = self._to_lines(normalized_text)

        personal_info = self.extract_personal_info(lines)
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
            "personal_info": personal_info,
            "educations": educations,
            "experiences": experiences,
            "projects": projects,
            "certifications": certifications,
            "skills": skills,
        }

    def extract_personal_info(self, lines: list[str]) -> dict[str, str | None]:
        raw = "\n".join(lines)
        email_match = re.search(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            raw,
            re.I,
        )
        phone_match = re.search(r"(?<!\d)(?:\+?62|0)[\d\s().-]{8,}\d", raw)
        linkedin_match = re.search(
            r"(?:https?://)?(?:www\.)?linkedin\.com/in/[^\s|,;]+",
            raw,
            re.I,
        )

        full_name = self._labeled_value(lines, ("name", "nama", "nama lengkap"))
        if not full_name:
            full_name = self._header_name(lines)

        professional_headline = self._labeled_value(
            lines,
            ("headline", "professional headline", "title", "jabatan"),
        )
        if not professional_headline and full_name:
            professional_headline = self._line_after(lines, full_name)

        domicile = self._labeled_value(
            lines,
            ("domicile", "domisili", "location", "lokasi", "address", "alamat"),
        )
        summary_lines = self._section_lines(lines, "summary")
        profile_summary = " ".join(summary_lines).strip()[:1000] or None

        return {
            "full_name": full_name,
            "professional_headline": professional_headline,
            "email": email_match.group(0) if email_match else None,
            "phone_number": phone_match.group(0).strip() if phone_match else None,
            "domicile": domicile,
            "linkedin_url": linkedin_match.group(0) if linkedin_match else None,
            "profile_summary": profile_summary,
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

    def _labeled_value(
        self,
        lines: list[str],
        labels: tuple[str, ...],
    ) -> str | None:
        pattern = "|".join(re.escape(label) for label in labels)
        for line in lines:
            match = re.match(rf"^(?:{pattern})\s*[:|-]\s*(.+)$", line, re.I)
            if match:
                return match.group(1).strip() or None
        return None

    def _header_name(self, lines: list[str]) -> str | None:
        headings = {
            heading.lower()
            for section_headings in SECTION_HEADINGS.values()
            for heading in section_headings
        }
        for line in lines[:8]:
            cleaned = line.strip()
            lowered = re.sub(r"[^a-zA-Z ]", "", cleaned).strip().lower()
            if lowered in headings or re.search(r"@|https?://|linkedin|\d", cleaned, re.I):
                continue
            words = cleaned.split()
            if 2 <= len(words) <= 6 and 3 <= len(cleaned) <= 80:
                return cleaned
        return None

    def _line_after(self, lines: list[str], value: str) -> str | None:
        try:
            start_index = lines.index(value)
        except ValueError:
            return None
        headings = {
            heading.lower()
            for section_headings in SECTION_HEADINGS.values()
            for heading in section_headings
        }
        for line in lines[start_index + 1 : start_index + 4]:
            lowered = re.sub(r"[^a-zA-Z ]", "", line).strip().lower()
            if lowered in headings or re.search(r"@|https?://|linkedin|(?:\+?62|0)\d", line, re.I):
                continue
            if len(line) <= 120:
                return line
        return None

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
