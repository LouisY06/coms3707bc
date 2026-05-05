import importlib
import os
import unittest


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


class ProviderTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.generate = importlib.reload(importlib.import_module("generate"))

    def test_import_does_not_require_api_keys(self):
        self.assertTrue(hasattr(self.generate, "generate_responses"))

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

    def test_supported_models_include_cheap_anthropic_model(self):
        self.assertIn("claude-haiku-4-5-20251001", self.generate.SUPPORTED_MODELS)


if __name__ == "__main__":
    unittest.main()
