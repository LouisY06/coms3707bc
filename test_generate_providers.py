import importlib
import os
import unittest
from unittest.mock import patch


class FakeAnthropicMessage:
    content = [type("Block", (), {"text": "anthropic answer"})()]


class FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeAnthropicMessage()


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeAnthropicMessages()


class FakeOpenAIMessage:
    content = "openai answer"


class FakeOpenAIChoice:
    message = FakeOpenAIMessage()


class FakeOpenAICompletion:
    choices = [FakeOpenAIChoice()]


class FakeOpenAICompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOpenAICompletion()


class FakeOpenAIChat:
    def __init__(self):
        self.completions = FakeOpenAICompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeOpenAIChat()


class FakeGeminiResponse:
    text = "gemini answer"


class FakeGeminiModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGeminiResponse()


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeGeminiModels()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        self.generate = importlib.reload(importlib.import_module("generate"))

    def test_import_does_not_require_api_keys(self):
        self.assertTrue(hasattr(self.generate, "generate_responses"))

    def test_openai_models_use_shared_token_cap(self):
        client = FakeOpenAIClient()

        responses = self.generate.generate_responses(
            "gpt-4o-mini",
            "Q: 1 + 1?\nA:",
            n=2,
            temperature=0.8,
            top_p=1.0,
            openai_client=client,
        )

        self.assertEqual(responses, ["openai answer", "openai answer"])
        self.assertEqual(len(client.chat.completions.calls), 2)
        self.assertEqual(client.chat.completions.calls[0]["max_tokens"], 4096)

    def test_claude_models_use_anthropic_messages_api(self):
        client = FakeAnthropicClient()

        responses = self.generate.generate_responses(
            "claude-haiku-4-5-20251001",
            "Q: 1 + 1?\nA:",
            n=2,
            temperature=0.8,
            top_p=1.0,
            anthropic_client=client,
        )

        self.assertEqual(responses, ["anthropic answer", "anthropic answer"])
        self.assertEqual(len(client.messages.calls), 2)
        self.assertEqual(client.messages.calls[0]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(client.messages.calls[0]["messages"][0]["content"], "Q: 1 + 1?\nA:")
        self.assertEqual(client.messages.calls[0]["temperature"], 0.8)
        self.assertEqual(client.messages.calls[0]["max_tokens"], 4096)
        self.assertNotIn("top_p", client.messages.calls[0])

    def test_supported_models_include_cheap_anthropic_model(self):
        self.assertIn("claude-haiku-4-5-20251001", self.generate.SUPPORTED_MODELS)

    def test_gemini_models_use_google_genai_models_api(self):
        client = FakeGeminiClient()

        responses = self.generate.generate_responses(
            "gemini-2.0-flash",
            "Q: 1 + 1?\nA:",
            n=2,
            temperature=0.8,
            top_p=1.0,
            gemini_client=client,
        )

        self.assertEqual(responses, ["gemini answer", "gemini answer"])
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(client.models.calls[0]["model"], "gemini-2.0-flash")
        self.assertEqual(client.models.calls[0]["contents"], "Q: 1 + 1?\nA:")
        self.assertEqual(client.models.calls[0]["config"]["temperature"], 0.8)
        self.assertEqual(client.models.calls[0]["config"]["top_p"], 1.0)
        self.assertEqual(client.models.calls[0]["config"]["max_output_tokens"], 4096)
        self.assertEqual(client.models.calls[0]["config"]["thinking_config"]["thinking_budget"], 0)

    def test_supported_models_include_gemini_flash(self):
        self.assertIn("gemini-2.0-flash", self.generate.SUPPORTED_MODELS)
        self.assertIn("gemini-2.5-flash", self.generate.SUPPORTED_MODELS)

    def test_retryable_api_errors_are_retried(self):
        class TemporaryError(Exception):
            status_code = 503

        calls = {"count": 0}

        def flaky_call():
            calls["count"] += 1
            if calls["count"] == 1:
                raise TemporaryError("try again")
            return "ok"

        with patch("generate.time.sleep") as sleep:
            self.assertEqual(self.generate.run_with_retries(flaky_call, max_attempts=2, initial_delay=1), "ok")

        self.assertEqual(calls["count"], 2)
        sleep.assert_called_once_with(1)

    def test_google_server_error_without_status_code_is_retryable(self):
        class ServerError(Exception):
            pass

        self.assertTrue(self.generate.is_retryable_api_error(ServerError("high demand")))


if __name__ == "__main__":
    unittest.main()
