import importlib
import unittest


class RunExperimentImportTests(unittest.TestCase):
    def test_import_does_not_require_evaluation_dependencies(self):
        module = importlib.import_module("run_experiment")

        self.assertTrue(hasattr(module, "run_step_generate"))


if __name__ == "__main__":
    unittest.main()
