from unittest import TestCase

from src.domain.cvparser.cvparser import CVParser
from src.features.cvparser.parse_cv.parse_cv import ParseCVResult


class CVParserPersonalInfoTest(TestCase):
    def test_extracts_personal_info_with_fallback_parser(self) -> None:
        text = """Anargya Isadhi Maheswara
Software Engineer & AI Enthusiast
anargya@example.com | +62 812-3456-7890
linkedin.com/in/anargya
Location: Bogor, Indonesia

Summary
Computer Science student experienced in scalable web applications and AI products.

Skills
Python, TypeScript, Power BI
"""

        result = CVParser().parse_cv_with_rules(text)

        self.assertEqual(
            result["personal_info"],
            {
                "full_name": "Anargya Isadhi Maheswara",
                "professional_headline": "Software Engineer & AI Enthusiast",
                "email": "anargya@example.com",
                "phone_number": "+62 812-3456-7890",
                "domicile": "Bogor, Indonesia",
                "linkedin_url": "linkedin.com/in/anargya",
                "profile_summary": (
                    "Computer Science student experienced in scalable web "
                    "applications and AI products."
                ),
            },
        )
        self.assertTrue(result["skills"])
        for skill in result["skills"]:
            self.assertIn("learning_hours", skill)
            self.assertIn("working_hours", skill)
            self.assertIsNone(skill["learning_hours"])
            self.assertIsNone(skill["working_hours"])

        response = ParseCVResult(**result).model_dump()
        self.assertIn("personal_info", response)
        self.assertIn("learning_hours", response["skills"][0])
        self.assertIn("working_hours", response["skills"][0])
