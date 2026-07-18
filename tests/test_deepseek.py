import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from src.infrastructure.llm import deepseek


class DeepSeekClientTest(IsolatedAsyncioTestCase):
    async def test_extracts_structured_cv_through_deepseek(self) -> None:
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "confidence_score": 0.9,
                                "educations": [],
                                "experiences": [],
                                "projects": [],
                                "certifications": [],
                                "skills": [],
                            }
                        )
                    }
                }
            ]
        }
        response = MagicMock()
        response.status_code = 200
        response.is_error = False
        response.json.return_value = response_payload

        client = MagicMock()
        client.post = AsyncMock(return_value=response)
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=None)
        configured_settings = SimpleNamespace(
            deepseek_api_key="test-key",
            deepseek_base_url="https://api.deepseek.com/",
            deepseek_model="deepseek-chat",
            deepseek_max_tokens=8000,
            deepseek_timeout_seconds=120,
        )

        with (
            patch.object(deepseek, "settings", configured_settings),
            patch.object(deepseek.httpx, "AsyncClient", return_value=client_context),
        ):
            result = await deepseek.extract_cv_with_deepseek("Sample CV")

        response.raise_for_status.assert_called_once_with()
        client.post.assert_awaited_once()
        url = client.post.await_args.args[0]
        request = client.post.await_args.kwargs
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(request["json"]["model"], "deepseek-chat")
        self.assertEqual(request["json"]["max_tokens"], 8000)
        self.assertEqual(request["json"]["response_format"], {"type": "json_object"})
        self.assertFalse(request["json"]["stream"])
        self.assertIn("personal_info", request["json"]["messages"][1]["content"])
        self.assertIn("professional_headline", deepseek.SYSTEM_PROMPT)
        self.assertEqual(result.confidence_score, 0.9)

    async def test_retries_with_compact_output_when_response_is_truncated(self) -> None:
        truncated_response = MagicMock(status_code=200, is_error=False)
        truncated_response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"confidence_score": 0.9'},
                }
            ]
        }
        complete_response = MagicMock(status_code=200, is_error=False)
        complete_response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "confidence_score": 0.9,
                                "educations": [],
                                "experiences": [],
                                "projects": [],
                                "certifications": [],
                                "skills": [],
                            }
                        )
                    },
                }
            ]
        }
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[truncated_response, complete_response]
        )
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=None)
        configured_settings = SimpleNamespace(
            deepseek_api_key="test-key",
            deepseek_base_url="https://api.deepseek.com",
            deepseek_model="deepseek-chat",
            deepseek_max_tokens=8000,
            deepseek_timeout_seconds=120,
        )

        with (
            patch.object(deepseek, "settings", configured_settings),
            patch.object(deepseek.httpx, "AsyncClient", return_value=client_context),
        ):
            result = await deepseek.extract_cv_with_deepseek("Sample CV")

        self.assertEqual(client.post.await_count, 2)
        retry_request = client.post.await_args_list[1].kwargs["json"]
        retry_prompt = retry_request["messages"][1]["content"]
        self.assertIn("Prioritaskan JSON selesai", retry_prompt)
        self.assertEqual(result.confidence_score, 0.9)

    async def test_requires_deepseek_api_key(self) -> None:
        unconfigured_settings = SimpleNamespace(deepseek_api_key=None)

        with patch.object(deepseek, "settings", unconfigured_settings):
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                await deepseek.extract_cv_with_deepseek("Sample CV")
