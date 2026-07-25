import asyncio
import json
import logging
from pathlib import Path
from src.features.job_matcher.service import JobMatcherService

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def main():
    print("Memulai simulasi HANYA untuk JobMatcher...")
    data_file = Path("data/result-full_AFTER_SKILL_PARSERS.json").resolve()
    
    if not data_file.exists():
        print(f"File {data_file} tidak ditemukan!")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    job_matcher = JobMatcherService()
    
    print("\nMenjalankan JobMatcher Graph...")
    matcher_output = await job_matcher.run(raw_data)
    
    output_file = Path("data/result-jobmatcher.json")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(matcher_output.model_dump_json(indent=2))
        
    print(f"\n[OK] JobMatcher Selesai! Data disimpan di: {output_file.absolute()}")
    
    print(f"\nTotal Rekomendasi Karir: {len(matcher_output.career_match_results)}")
    print(f"Total Skill Gaps: {len(matcher_output.skill_gap_results)}")
    print(f"Total Lowongan Pekerjaan Riil (Active Job Postings): {len(matcher_output.active_job_postings)}")
    
    if matcher_output.career_match_results:
        best_match = max(matcher_output.career_match_results, key=lambda x: x.total_match_score)
        print(f"\n[Best Match] Role: {best_match.role_name} (Score: {best_match.total_match_score:.2f}%)")
        print(f"Alasan: {best_match.match_reason}")

    if matcher_output.active_job_postings:
        print("\n[Jobs] Lowongan Pekerjaan Tersedia:")
        for job in matcher_output.active_job_postings[:5]:
            print(f"  - {job.title} di {job.company} ({job.location})")
            print(f"    Link: {job.url}")

if __name__ == "__main__":
    asyncio.run(main())
