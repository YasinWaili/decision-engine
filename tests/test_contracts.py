import unittest

from decision_engine.predictors import MockPredictor
from decision_engine.schemas import DecisionRequest, DecisionResponse


class ContractTests(unittest.TestCase):
    def test_valid_mock_output_is_explicit_and_abstains(self):
        request = DecisionRequest("A cracked screen", "Physical damage?", ("Yes", "No"))
        result = MockPredictor().predict(request)
        self.assertEqual(result.probabilities, {"Yes": 0.5, "No": 0.5})
        self.assertTrue(result.abstained)
        self.assertFalse(result.calibrated)
        self.assertIn("mock", result.source)

    def test_duplicate_options_rejected(self):
        with self.assertRaises(ValueError):
            DecisionRequest("state", "question", ("Yes", " yes "))

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            DecisionRequest("state", " ", ("Yes", "No"))

    def test_single_option_rejected(self):
        with self.assertRaises(ValueError):
            DecisionRequest("state", "question", ("Yes",))

    def test_invalid_distributions_rejected(self):
        for probabilities in ({"A": 0.9, "B": 0.9}, {"A": float("nan"), "B": 0.5}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                DecisionResponse("A", probabilities, False, "test")

    def test_decision_must_be_top_option(self):
        with self.assertRaises(ValueError):
            DecisionResponse("A", {"A": 0.2, "B": 0.8}, False, "test")


if __name__ == "__main__":
    unittest.main()
