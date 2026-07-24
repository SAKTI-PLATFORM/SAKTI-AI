# Implementation Plan: JobMatcher & TalentForger Modules
## SAKTI Platform — LangGraph + Pydantic AI + Langfuse

> Dokumen ini adalah implementation plan siap pakai untuk AI coding agent (Claude Code / Antigravity).
> Konteks: modul `SkillParser` sudah selesai dan menghasilkan `parsed_result` (personalInfo, educations, experiences, projects, skills). Dokumen ini melanjutkan ke `JobMatcher` dan `TalentForger`.

---

## 1. Ringkasan Alur Sistem

```
SkillParser (DONE) 
    ↓ parsed_result (skills w/ confidence_score, evidence_strength)
JobMatcher
    ↓ career_match_results + skill_gap_results (match_id sebagai kontrak)
TalentForger
    ↓ learning_paths + resource_recommendations
```

Kedua modul dibangun sebagai **graph LangGraph terpisah**, dijalankan **asynchronous** melalui Redis queue + workers, dengan **progress title** dipublish real-time via Redis Pub/Sub, dan full traceability via **Langfuse**.

---

## 2. Struktur Graph LangGraph

### 2.1 Graph JobMatcher

```
START
 → get_preferences
     ├─ get_skills_preferences   (parallel)
     └─ get_career_preferences   (parallel)
 → search_job_market              (tool call, async)
 → calc_similarity                (embedding user_skills vs role_skill_requirements)
 → score_dimensions               (skill_match, experience_project, education, riasec_fit, ocean_workstyle, preference)
 → generate_skill_gap             (current_level vs required_level per skill)
 → generate_market_demand         (agregasi frekuensi skill di job_posting_skills)
 → _explains                      (LLM node: match_reason + gap.reason)
 → END → emit career_match_results + career_match_score_details + skill_gap_results
```

**Conditional edge:** jika `total_match_score >= MATCH_THRESHOLD` (default 0.70) → lanjut generate_skill_gap ringan (hanya skill non-mandatory); jika `< MATCH_THRESHOLD` → full skill_gap_results + trigger TalentForger di akhir.

### 2.2 Graph TalentForger

```
START
 → _get_job_references            (fetch role_templates + role_skill_requirements by match_id)
 → _get_course_references         (tool search: cari course utk skill_gap, cache di vector DB)
 → _get_cert_references           (tool search: sertifikasi relevan)
 → get_preferences
     ├─ get_skills_preferences    (parallel)
     └─ get_career_preferences    (parallel)
 → recommends
     ├─ recommends_jobs
     ├─ recommends_course
     └─ recommends_certs
 → _explains                      (LLM node: recommendation_reason per resource)
 → END → emit learning_paths + learning_path_steps + learning_resources + resource_recommendations
```

---

## 3. Pydantic Schemas — State & Output

### 3.1 Shared / Input Schemas

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

class SkillEvidence(BaseModel):
    detected_text: str
    confidence_score: float = Field(ge=0, le=1)
    learning_hours: Optional[int] = None
    working_hours: Optional[int] = None
    evidence_source: str
    evidence_strength: Literal["low", "medium", "high"]

class UserProfileInput(BaseModel):
    user_id: str
    cv_id: str
    onboarding_session_id: str
    skills: list[SkillEvidence]
    educations: list[dict]
    experiences: list[dict]
    projects: list[dict]

class UserPreferences(BaseModel):
    preferred_industries: list[str] = []
    preferred_locations: list[str] = []
    salary_expectation_min: Optional[int] = None
    work_mode_pref: Literal["Remote", "Hybrid", "On-site", "Any"] = "Any"
    career_goal: Optional[str] = None
```

### 3.2 JobMatcher State Graph

```python
class RoleReference(BaseModel):
    role_id: str
    role_name: str
    role_category: str
    role_level: str
    description: str

class SimilarityResult(BaseModel):
    role_id: str
    raw_similarity: float
    matched_skills: list[str]
    unmatched_skills: list[str]

class ScoreDetail(BaseModel):
    match_id: str
    skill_match_score: float
    experience_project_score: float
    education_score: float
    riasec_fit_score: float
    ocean_workstyle_score: float
    preference_score: float

class CareerMatchResult(BaseModel):
    match_id: str
    user_id: str
    role_id: str
    total_match_score: float
    match_reason: str
    created_at: datetime

class SkillGapResult(BaseModel):
    gap_id: str
    match_id: str
    skill_id: str
    current_level: Literal["None", "Beginner", "Intermediate", "Advanced"]
    required_level: Literal["Beginner", "Intermediate", "Advanced"]
    gap_level: Literal["Low", "Medium", "High"]
    priority: Literal["Low", "Medium", "High"]
    reason: str

class JobMatcherState(BaseModel):
    user_profile: UserProfileInput
    preferences: Optional[UserPreferences] = None
    candidate_roles: list[RoleReference] = []
    similarity_results: list[SimilarityResult] = []
    score_details: list[ScoreDetail] = []
    career_matches: list[CareerMatchResult] = []
    skill_gaps: list[SkillGapResult] = []
    market_demand: dict[str, float] = {}
    progress_step: int = 0
    trace_id: Optional[str] = None

class JobMatcherOutput(BaseModel):
    career_match_results: list[CareerMatchResult]
    career_match_score_details: list[ScoreDetail]
    skill_gap_results: list[SkillGapResult]
```

### 3.3 TalentForger State Graph

```python
class LearningPath(BaseModel):
    learning_path_id: str
    match_id: str
    target_role: str
    learning_path_type: str
    estimated_duration_weeks: int
    created_at: datetime

class LearningPathStep(BaseModel):
    step_id: str
    learning_path_id: str
    gap_id: str
    step_order: int
    week: int
    topic: str
    objective: str
    related_skill_id: str

class LearningResource(BaseModel):
    resource_id: str
    skill_id: str
    resource_title: str
    resource_type: Literal["Course", "Certification", "Article", "Video"]
    provider: str
    difficulty_level: Literal["Beginner", "Intermediate", "Advanced"]
    estimated_duration_hours: int
    url: str

class ResourceRecommendation(BaseModel):
    recommendation_id: str
    step_id: str
    resource_id: str
    recommendation_reason: str
    priority_order: int

class TalentForgerState(BaseModel):
    match_id: str
    user_profile: UserProfileInput
    preferences: Optional[UserPreferences] = None
    skill_gaps: list[SkillGapResult]
    role_references: list[RoleReference] = []
    course_candidates: list[LearningResource] = []
    cert_candidates: list[LearningResource] = []
    learning_paths: list[LearningPath] = []
    learning_path_steps: list[LearningPathStep] = []
    resource_recommendations: list[ResourceRecommendation] = []
    progress_step: int = 0
    trace_id: Optional[str] = None

class TalentForgerOutput(BaseModel):
    learning_paths: list[LearningPath]
    learning_path_steps: list[LearningPathStep]
    learning_resources: list[LearningResource]
    resource_recommendations: list[ResourceRecommendation]
```

---

## 4. Contoh Output per Node

**`calc_similarity`:**
```json
{"role_id": "ROLE001", "raw_similarity": 0.81, "matched_skills": ["SQL", "Python"], "unmatched_skills": ["Tableau"]}
```

**Output akhir JobMatcher:**
```json
{
  "career_match_results": [{"match_id": "MATCH001", "user_id": "USR001", "role_id": "ROLE001", "total_match_score": 81.5, "match_reason": "Strong SQL and dashboard evidence, high investigative fit, and preference alignment with hybrid data roles."}],
  "skill_gap_results": [{"gap_id": "GAP001", "match_id": "MATCH001", "skill_id": "SKL002", "current_level": "Beginner", "required_level": "Intermediate", "gap_level": "Medium", "priority": "High", "reason": "Python is mandatory for many Data Analyst postings and is currently below the required level."}]
}
```

**Output akhir TalentForger:**
```json
{
  "learning_paths": [{"learning_path_id": "LP001", "match_id": "MATCH001", "target_role": "Data Analyst", "estimated_duration_weeks": 4}],
  "resource_recommendations": [{"step_id": "STEP001", "resource_id": "RES001", "recommendation_reason": "Matches the highest-priority Python skill gap for the target Data Analyst role.", "priority_order": 1}]
}
```

---

## 5. Tools (Function Calls)

| Tool | Dipakai di | Signature | Fungsi |
|---|---|---|---|
| `search_job_postings` | JobMatcher | `(query: str, location: str, skills: list[str]) -> list[RoleReference]` | Web search/scrape fallback jika DB job_market kosong |
| `fetch_role_templates` | JobMatcher | `(role_category: str) -> list[RoleReference]` | Query DB role_templates + role_skill_requirements |
| `compute_skill_embedding` | JobMatcher | `(skill_list: list[str]) -> list[float]` | Generate embedding untuk cosine similarity |
| `calc_match_score` | JobMatcher | `(user_skills, role_requirements, weights: dict) -> ScoreDetail` | Weighted scoring |
| `fetch_market_demand` | JobMatcher | `(skill_id: str) -> float` | Agregasi frekuensi skill di job_posting_skills |
| `search_courses` | TalentForger | `(skill_gap: SkillGapResult, provider_pref: list[str]) -> list[LearningResource]` | Web search/API course provider |
| `search_certifications` | TalentForger | `(skill_gap: SkillGapResult) -> list[LearningResource]` | Web search sertifikasi relevan |
| `fetch_gap_details` | TalentForger | `(match_id: str) -> list[SkillGapResult]` | Ambil skill_gap_results dari JobMatcher output |
| `build_learning_sequence` | TalentForger | `(gaps, resources) -> list[LearningPathStep]` | Urutkan step berdasarkan prerequisite |
| `log_trace` | Semua node | `(node_name: str, input: dict, output: dict, latency_ms: float)` | Kirim span ke Langfuse |
| `emit_progress` | Semua node | `(session_id: str, turn_title: str, step: int, total_steps: int)` | Publish progress ke Redis pub/sub |

---

## 6. Backend & Async Workers

- **Queue:** `celery` atau `rq` dengan Redis broker, dua queue terpisah: `jobmatcher_queue` dan `talentforger_queue` agar scaling independen.
- **Progress channel:** Redis Pub/Sub `progress:{session_id}` — setiap node publish event:
  ```json
  {"turn_title": "Menghitung skor kecocokan skill...", "step": 3, "total_steps": 7, "module": "JobMatcher"}
  ```
- **Cache:** hasil `search_courses` / `search_job_postings` disimpan di Redis dengan TTL (`CACHE_TTL_SECONDS`) agar tidak re-fetch tool eksternal berulang.
- **Worker pattern:** `JobMatcherWorker` dan `TalentForgerWorker` masing-masing menjalankan `graph.ainvoke()` secara async; setelah JobMatcher selesai, worker otomatis enqueue task ke `talentforger_queue` dengan payload `match_id`.
- **Checkpointing:** gunakan LangGraph `RedisSaver` sebagai checkpointer sehingga state graph bisa di-resume jika worker crash di tengah proses.

---

## 7. Traceability (Langfuse)

- Satu `graph.ainvoke()` = satu **trace** Langfuse, dengan `session_id = onboarding_session_id`.
- Setiap node LangGraph = satu **span**, nama span sesuai nama node (`get_preferences`, `calc_similarity`, dst), mencatat input/output + `metadata.turn_title`.
- Tag trace: `module=JobMatcher` / `module=TalentForger`, `user_id`.
- Feedback (`feedback_logs.rating`) dikirim balik sebagai Langfuse **score** terhadap `trace_id` terkait untuk mengukur kualitas prompt/versi tertentu.

---

## 8. Prompt Engineering

- Prompt di-versioning via **Langfuse Prompt Management**, bukan hardcode. Node LLM (`_explains`, `recommends_course`) masing-masing punya template dengan variabel `{user_skills}`, `{role_requirements}`, `{gap_list}`.
- Output LLM di-constraint via Pydantic AI `result_type` — wajib isi field `match_reason` / `recommendation_reason` dengan referensi eksplisit ke `evidence_strength` dan `confidence_score` dari SkillParser agar explainable.
- Few-shot examples memakai sample JSON (career_match, learning_path) sebagai referensi format.

---

## 9. Vector Data & Memory Management

- **Vector store** (pgvector / Chroma): embedding skill taxonomy, role_skill_requirements, course catalog — dipakai di `calc_similarity` dan `search_courses` untuk semantic matching.
- **Short-term memory:** state graph (Pydantic model di atas) disimpan sebagai Redis checkpoint per run, hidup selama proses berjalan.
- **Long-term memory:** hasil akhir disimpan permanen di Postgres; `feedback_logs` dipakai untuk fine-tune bobot scoring / retrieval preference per `user_id` pada run berikutnya.

---

## 10. config.py — Parameter yang Perlu Diatur

```
JOBMATCHER_SCORING_WEIGHTS = {
    "skill": 0.4, "experience": 0.2, "education": 0.2,
    "riasec": 0.1, "ocean": 0.05, "preference": 0.05
}
MATCH_THRESHOLD = 0.70

REDIS_URL = "redis://localhost:6379/0"
REDIS_PROGRESS_CHANNEL_PREFIX = "progress:"
CACHE_TTL_SECONDS = 86400

LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_HOST = "..."

LLM_MODEL_PER_NODE = {
    "_explains": "gpt-4o-mini",
    "recommends_course": "gpt-4o-mini"
}

VECTOR_DB_URL = "postgresql://..."
EMBEDDING_MODEL = "text-embedding-3-small"

TOOL_SEARCH_PROVIDERS = {
    "job": "...", "course": "...", "cert": "..."
}

WORKER_CONCURRENCY = 4
QUEUE_NAMES = {"jobmatcher": "jobmatcher_queue", "talentforger": "talentforger_queue"}
MIN_RESOURCES_PER_GAP = 3
```

---

## 11. Testing Plan (End-to-End dengan Sample Nyata)

Gunakan input dari `result-full_AFTER_SKILL_PARSERS-2.json` (skills: Python, React.js, SQL, dsb, dengan confidence_score dan evidence_strength) sebagai test case.

- **Unit test `calc_similarity`:** assert cosine similarity Python (confidence 0.9) vs role requirement Python Intermediate menghasilkan skor > 0.75.
- **Unit test `generate_skill_gap`:** assert output schema sesuai `SkillGapResult` (gap_id, current_level, required_level, priority).
- **Integration test full graph JobMatcher:** input CV Anargya → expect minimal satu `CareerMatchResult` dengan `total_match_score` > 70 untuk role software engineering.
- **Integration test TalentForger:** gunakan `match_id` dari hasil JobMatcher di atas → expect `learning_paths` dan minimal `MIN_RESOURCES_PER_GAP` (3) `resource_recommendations` per gap.
- **Trace assertion:** verifikasi di Langfuse trace punya span lengkap sesuai node graph, tanpa span error, latency Matchmaker < 2 detik.
- **Regression test feedback loop:** simulasikan `feedback_logs` rating rendah (rating=1) lalu verifikasi sistem menurunkan `priority_order` rekomendasi terkait pada run berikutnya.

---

## 12. Urutan Implementasi yang Disarankan

1. Definisikan semua Pydantic schema (section 3) sebagai kontrak data.
2. Implementasikan tools (section 5) satu per satu dengan unit test.
3. Bangun graph JobMatcher node-by-node di LangGraph, integrasikan Langfuse callback.
4. Bangun graph TalentForger, hubungkan via `match_id`.
5. Setup Redis queue + worker + progress pub/sub.
6. Setup config.py sesuai section 10.
7. Jalankan testing plan (section 11) end-to-end dengan sample CV Anargya.
