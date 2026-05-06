import json
import tempfile
import unittest
from pathlib import Path

from plot_results import load_results, rows_from_results, strategy_averages


class PlotResultsTests(unittest.TestCase):
    def test_rows_from_results_orders_methods_and_preserves_scores(self):
        results = {
            "model-a/aqua": {
                "model": "model-a",
                "dataset": "aqua",
                "greedy": 80.0,
                "majority_vote": 86.0,
                "cpw": 88.0,
                "scw": 84.0,
            }
        }

        rows = rows_from_results(results)

        self.assertEqual(
            rows,
            [
                {
                    "model": "model-a",
                    "dataset": "aqua",
                    "strategy": "Greedy",
                    "accuracy": 80.0,
                    "improvement_over_greedy": 0.0,
                },
                {
                    "model": "model-a",
                    "dataset": "aqua",
                    "strategy": "Majority Vote",
                    "accuracy": 86.0,
                    "improvement_over_greedy": 6.0,
                },
                {
                    "model": "model-a",
                    "dataset": "aqua",
                    "strategy": "CPW",
                    "accuracy": 88.0,
                    "improvement_over_greedy": 8.0,
                },
                {
                    "model": "model-a",
                    "dataset": "aqua",
                    "strategy": "SCW",
                    "accuracy": 84.0,
                    "improvement_over_greedy": 4.0,
                },
            ],
        )

    def test_strategy_averages_groups_by_strategy(self):
        rows = [
            {"strategy": "Greedy", "accuracy": 70.0},
            {"strategy": "Greedy", "accuracy": 90.0},
            {"strategy": "CPW", "accuracy": 80.0},
        ]

        self.assertEqual(strategy_averages(rows), {"Greedy": 80.0, "CPW": 80.0})

    def test_load_results_reads_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            path.write_text(json.dumps({"x/y": {"model": "x", "dataset": "y"}}))

            self.assertEqual(load_results(path), {"x/y": {"model": "x", "dataset": "y"}})


if __name__ == "__main__":
    unittest.main()
