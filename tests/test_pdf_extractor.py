from unittest import TestCase
from unittest.mock import MagicMock, patch

from src.domain.cvparser.pdf_extractor import extract_pdf_text


class PdfExtractorTest(TestCase):
    def test_rejects_content_without_pdf_signature(self) -> None:
        with self.assertRaisesRegex(ValueError, "bukan PDF yang valid"):
            extract_pdf_text(b"plain text")

    @patch("src.domain.cvparser.pdf_extractor.pdfplumber.open")
    def test_combines_text_from_all_pages(self, open_pdf: MagicMock) -> None:
        first_page = MagicMock()
        first_page.extract_text.return_value = "First page"
        second_page = MagicMock()
        second_page.extract_text.return_value = "Second page"
        open_pdf.return_value.__enter__.return_value.pages = [
            first_page,
            second_page,
        ]

        result = extract_pdf_text(b"%PDF-1.7\ncontent")

        self.assertEqual(result, "First page\n\nSecond page")

    @patch("src.domain.cvparser.pdf_extractor.pdfplumber.open")
    def test_rejects_pdf_without_embedded_text(self, open_pdf: MagicMock) -> None:
        page = MagicMock()
        page.extract_text.return_value = ""
        open_pdf.return_value.__enter__.return_value.pages = [page]

        with self.assertRaisesRegex(ValueError, "Teks CV tidak terdeteksi"):
            extract_pdf_text(b"%PDF-1.7\nimage-only")
