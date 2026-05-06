import unittest

from evaluation_utils import answers_match, normalize_answer


class EvaluationUtilsTests(unittest.TestCase):
    def test_svamp_integer_float_answers_match_integer_strings(self):
        self.assertEqual(normalize_answer("6.0", "svamp"), "6")
        self.assertTrue(answers_match("6", "6.0", "svamp"))

    def test_non_numeric_answers_use_case_insensitive_string_match(self):
        self.assertTrue(answers_match("Yes", "yes", "strategyqa"))
        self.assertTrue(answers_match("A", "a", "aqua"))

    def test_null_prediction_is_incorrect(self):
        self.assertFalse(answers_match(None, "A", "aqua"))


if __name__ == "__main__":
    unittest.main()
