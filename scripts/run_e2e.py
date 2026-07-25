import asyncio
import json
import logging
from pathlib import Path
from pprint import pprint

from src.features.job_matcher.service import JobMatcherService
from src.features.talent_forger.service import TalentForgerService
from src.infrastructure.adapters.onboarding_adapter import parse_onboarding_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


async def run_pipeline(json_path: str):
    print("==================================================")
    print("Memulai Pipeline SAKTI: JobMatcher & TalentForger")
    print(f"Membaca file: {json_path}")
    print("==================================================")

    # 1. Parse Data
    with open(json_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    
    profile = parse_onboarding_data(raw_data)
    print(f"\nProfil User ID: {profile.user_id}")
    print(f"Total Skills Terdeteksi: {len(profile.skills)}")
    print(f"Total Edukasi: {len(profile.educations)}")
    print(f"Total Pengalaman: {len(profile.experiences)}")

    # 2. Inisialisasi Service
    job_matcher = JobMatcherService()
    talent_forger = TalentForgerService()

    try:
        # 3. Jalankan JobMatcher
        print("\n[1/2] Menjalankan JobMatcher...")
        matcher_output = await job_matcher.run(raw_data)
        # Simpan hasil JobMatcher
        jm_output_file = Path("data/result-jobmatcher.json")
        with open(jm_output_file, "w", encoding="utf-8") as f:
            f.write(matcher_output.model_dump_json(indent=2))
            
        print(f"\nJobMatcher Selesai! Data disimpan di: {jm_output_file.absolute()}")
        print(f"Total Rekomendasi Karir: {len(matcher_output.career_match_results)}")
        print(f"Total Skill Gaps: {len(matcher_output.skill_gap_results)}")
        
        if not matcher_output.career_match_results:
            print("Tidak ada role yang cocok. Berhenti.")
            return
            
        best_match = matcher_output.career_match_results[0]
        print(f"\nBest Match Role ID: {best_match.role_id} (Score: {best_match.total_match_score}%)")
        print(f"Alasan: {best_match.match_reason}")

        # 4. Jalankan TalentForger
        print(f"\n[2/2] Menjalankan TalentForger untuk Match ID: {best_match.match_id}...")
        forger_output = await talent_forger.run(
            match_id=best_match.match_id,
            skill_gaps=[gap.model_dump() for gap in matcher_output.skill_gap_results],
            raw_input=raw_data
        )

        print("\nTalentForger Selesai!")
        print(f"Total Learning Paths: {len(forger_output.learning_paths)}")
        print(f"Total Resources: {len(forger_output.learning_resources)} (Berbayar) | {len(forger_output.free_materials)} (Gratis)")
        
        if forger_output.learning_paths:
            lp = forger_output.learning_paths[0]
            print(f"\n* Learning Path Utama: {lp.target_role} ({lp.estimated_duration_weeks} minggu)")
            
            print("\n- Steps:")
            steps = sorted([s for s in forger_output.learning_path_steps if s.learning_path_id == lp.learning_path_id], key=lambda x: x.step_order)
            for step in steps:
                print(f"  Minggu {step.week}: {step.topic} - {step.objective}")
                
            # Simpan seluruh output ke JSON agar tidak hilang
            output_file = Path("data/result-talentforger.json")
            combined_output = {
                "job_matcher": matcher_output.model_dump(mode="json"),
                "talent_forger": forger_output.model_dump(mode="json"),
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(combined_output, f, indent=2)
            print(f"\nData lengkap berhasil disimpan ke: {output_file.absolute()}")

                
    except Exception as e:
        print("\nError saat menjalankan pipeline:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    data_file = Path("data/result-full_AFTER_SKILL_PARSERS.json").resolve()
    if not data_file.exists():
        print(f"File {data_file} tidak ditemukan!")
    else:
        asyncio.run(run_pipeline(str(data_file)))
