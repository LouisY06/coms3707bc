import unittest

from generate import extract_answer


class ExtractAnswerTests(unittest.TestCase):
    def test_aqua_accepts_letter_followed_by_parenthesis(self):
        self.assertEqual(extract_answer("Therefore, the answer is **E) $78.20**.", "aqua"), "E")
        self.assertEqual(extract_answer("**Answer: B) 0.26**", "aqua"), "B")

    def test_aqua_accepts_latex_text_choice(self):
        self.assertEqual(extract_answer(r"Thus, the answer is \( \text{A) 36} \).", "aqua"), "A")

    def test_aqua_still_accepts_parenthesized_choice(self):
        self.assertEqual(extract_answer("The answer is (C).", "aqua"), "C")

    def test_aqua_maps_boxed_numeric_answer_to_choice(self):
        question = "What was the original price? Answer Choices: A)$61 B)$65 C)$67.40 D)$70 E)$78.20"
        self.assertEqual(extract_answer(r"Therefore, the original price is \( \boxed{78.20} \).", "aqua", question), "E")

    def test_aqua_accepts_boxed_letter_without_text(self):
        self.assertEqual(extract_answer(r"The final answer is $\boxed{A}$", "aqua"), "A")

    def test_aqua_does_not_parse_probability_notation_as_choice(self):
        self.assertIsNone(extract_answer("Let A be the event. P(A) = 0.56.", "aqua"))

    def test_aqua_accepts_option_letter_phrase(self):
        self.assertEqual(extract_answer("Therefore, the bag currently holds 20 marbles, option A.", "aqua"), "A")

    def test_strategyqa_uncertain_remains_unparsed(self):
        self.assertIsNone(extract_answer("The answer is uncertain.", "strategyqa"))


if __name__ == "__main__":
    unittest.main()
