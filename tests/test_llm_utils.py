import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class ProviderError(Exception):
    pass


def load_llm_utils():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = Mock()

    huggingface = types.ModuleType("huggingface_hub")
    huggingface.InferenceClient = Mock

    together = types.ModuleType("together")
    together.Together = Mock
    together_error = types.ModuleType("together.error")
    together_error.RateLimitError = ProviderError
    together_error.ServiceUnavailableError = ProviderError

    dependencies = {
        "dotenv": dotenv,
        "huggingface_hub": huggingface,
        "together": together,
        "together.error": together_error,
    }
    module_path = Path(__file__).resolve().parents[1] / "llm_utils.py"
    spec = importlib.util.spec_from_file_location("llm_utils_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies), patch.dict(os.environ, {}, clear=True):
        spec.loader.exec_module(module)
    return module


llm_utils = load_llm_utils()


class AskLLMTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        llm_utils.llm = self.client

    def test_plain_text_answer_is_reused_without_second_llm_call(self):
        self.client.chat.return_value = "A p-value summarizes evidence against a null hypothesis."

        tool, answer = llm_utils.ask_llm([], "What is a p-value?")

        self.assertIsNone(tool)
        self.assertEqual(
            answer, "A p-value summarizes evidence against a null hypothesis."
        )
        self.client.chat.assert_called_once()

    def test_valid_json_tool_call_is_returned_without_second_llm_call(self):
        response = '{"tool":"stats","action":"summary","args":{"col":"x"}}'
        self.client.chat.return_value = response

        tool, answer = llm_utils.ask_llm([], "Summarize x")

        self.assertEqual(
            tool, {"tool": "stats", "action": "summary", "args": {"col": "x"}}
        )
        self.assertEqual(answer, response)
        self.client.chat.assert_called_once()

    def test_empty_response_uses_fallback_llm_call(self):
        self.client.chat.side_effect = ["", "Fallback answer"]

        tool, answer = llm_utils.ask_llm([], "What is a p-value?")

        self.assertIsNone(tool)
        self.assertEqual(answer, "Fallback answer")
        self.assertEqual(self.client.chat.call_count, 2)


if __name__ == "__main__":
    unittest.main()
