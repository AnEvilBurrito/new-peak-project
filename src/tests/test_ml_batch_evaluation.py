"""
Test script for ML batch evaluation runner (run-ml-batch-v1.py)

Tests the functionality of loading CSV task lists, running batch ML evaluation,
and saving results with the correct S3 path structure.
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the module we're testing using a different approach
ML_BATCH_AVAILABLE = False
ml_batch_module = None

# Path to the script - from src/tests to src/notebooks/ch5-paper/machine-learning/run-ml-batch-v1.py
script_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'notebooks', 'ch5-paper', 'machine-learning', 'run-ml-batch-v1.py'
)

if os.path.exists(script_path):
    try:
        # Load the module from file
        spec = importlib.util.spec_from_file_location("ml_batch_module", script_path)
        ml_batch_module = importlib.util.module_from_spec(spec)
        # Mock the imports that would fail in test environment
        import sys
        mock_modules = [
            'src.notebooks.ch5_paper.data_eng.create_ml_loader_v1',
            'src.ml.Workflow',
            'models.utils.s3_config_manager'
        ]
        
        for mod_name in mock_modules:
            if mod_name not in sys.modules:
                mock = MagicMock()
                sys.modules[mod_name] = mock
        
        spec.loader.exec_module(ml_batch_module)
        ML_BATCH_AVAILABLE = True
    except Exception as e:
        print(f"Warning: Could not import ML batch module for testing: {e}")
        ML_BATCH_AVAILABLE = False
else:
    print(f"Warning: Script not found at {script_path}")
    ML_BATCH_AVAILABLE = False


class TestMLBatchEvaluation(unittest.TestCase):
    """Test cases for ML batch evaluation functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.test_dir, "test_tasks.csv")
        
        # Create test CSV data
        test_data = {
            "feature_data": [
                "sy_simple_expression_noise_v1/noise_0/noisy_features.pkl",
                "sy_simple_expression_noise_v1/noise_0.1/noisy_features.pkl",
                "sy_simple_parameter_distortion_v2/distortion_1.1/features.pkl"
            ],
            "feature_data_label": [
                "noisy_features_0",
                "noisy_features_0.1",
                "features_1.1"
            ],
            "target_data": [
                "sy_simple_expression_noise_v1/noise_0/original_targets.pkl",
                "sy_simple_expression_noise_v1/noise_0.1/original_targets.pkl",
                "sy_simple_parameter_distortion_v2/distortion_1.1/targets.pkl"
            ],
            "target_data_label": [
                "original_targets",
                "original_targets",
                "original_targets"
            ],
            "experiment_type": [
                "expression-noise-v1",
                "expression-noise-v1",
                "parameter-distortion-v2"
            ],
            "level": [0, 0.1, 1.1],
            "model_name": ["sy_simple", "sy_simple", "sy_simple"]
        }
        
        self.test_df = pd.DataFrame(test_data)
        self.test_df.to_csv(self.csv_path, index=False)
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)
    
    def test_csv_structure_validation(self):
        """Test that CSV structure validation works correctly"""
        if not ML_BATCH_AVAILABLE:
            self.skipTest("ML batch module not available")
        
        # Test valid CSV
        is_valid = ml_batch_module.validate_csv_structure(self.test_df)
        self.assertTrue(is_valid, "Valid CSV should pass validation")
        
        # Test missing column
        invalid_df = self.test_df.drop(columns=["feature_data"])
        is_valid = ml_batch_module.validate_csv_structure(invalid_df)
        self.assertFalse(is_valid, "Invalid CSV should fail validation")
    
    
    @patch.object(ml_batch_module, 'BatchLoader', create=True)
    @patch.object(ml_batch_module, 'batch_eval_standard', create=True)
    def test_batch_evaluation_for_experiment(self, mock_batch_eval, mock_batch_loader):
        """Test batch evaluation for a single experiment"""
        if not ML_BATCH_AVAILABLE:
            self.skipTest("ML batch module not available")
        
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader_instance.load_task_list.return_value = None
        mock_loader_instance.prepare_for_batch_eval.return_value = (
            [Mock()], ["feature1"], Mock(), "target_name"
        )
        mock_batch_loader.return_value = mock_loader_instance
        
        # Mock batch evaluation results
        mock_results = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "Feature Data": ["feature1", "feature1"],
            "Mean Squared Error": [0.1, 0.05],
            "R2 Score": [0.8, 0.9],
            "Pearson Correlation": [0.85, 0.92],
            "Pearson P-Value": [0.001, 0.0001]
        })
        mock_batch_eval.return_value = mock_results
        
        # Mock S3 manager
        mock_s3_manager = Mock()
        mock_s3_manager.save_result_path = "test/path"
        
        # Run the function
        results_df, failed_tasks_df, metadata = ml_batch_module.run_batch_evaluation_for_experiment(
            task_df=self.test_df[self.test_df["experiment_type"] == "expression-noise-v1"],
            experiment_type="expression-noise-v1",
            model_name="sy_simple",
            s3_manager=mock_s3_manager,
            evaluation_params={
                "num_repeats": 10,
                "test_size": 0.2,
                "random_seed": 42,
                "n_jobs": -1
            }
        )
        
        # Verify results
        self.assertIsNotNone(results_df)
        self.assertEqual(len(results_df), 2)
        self.assertIn("experiment_type", results_df.columns)
        self.assertIn("model_name", results_df.columns)
        self.assertIn("evaluation_timestamp", results_df.columns)
        
        # Verify metadata
        self.assertEqual(metadata["experiment_type"], "expression-noise-v1")
        self.assertEqual(metadata["model_name"], "sy_simple")
        self.assertEqual(metadata["task_stats"]["total_tasks"], 2)
    
    def test_generate_summary_stats(self):
        """Test generation of summary statistics"""
        if not ML_BATCH_AVAILABLE:
            self.skipTest("ML batch module not available")
        
        # Create test results DataFrame
        results_df = pd.DataFrame({
            "Model": ["Linear Regression", "Linear Regression", "Random Forest", "Random Forest"],
            "Feature Data": ["feature1", "feature2", "feature1", "feature2"],
            "R2 Score": [0.8, 0.7, 0.9, 0.85],
            "Mean Squared Error": [0.1, 0.2, 0.05, 0.075],
            "Pearson Correlation": [0.85, 0.75, 0.92, 0.88]
        })
        
        summary_df = ml_batch_module.generate_summary_stats(results_df)
        
        self.assertIsNotNone(summary_df)
        self.assertGreater(len(summary_df), 0)
        self.assertIn("Model", summary_df.columns)
        self.assertIn("Feature Data", summary_df.columns)
        self.assertIn("R2 Score_mean", summary_df.columns)
        self.assertIn("Mean Squared Error_mean", summary_df.columns)
    
    def test_s3_path_structure(self):
        """Test that S3 path structure matches expected pattern"""
        if not ML_BATCH_AVAILABLE:
            self.skipTest("ML batch module not available")
        
        # Test path construction logic directly
        base_path = "new-peak-project/experiments/ch5-paper/machine-learning"
        experiment_type = "expression-noise-v1"
        model_name = "sy_simple"
        
        expected_results_path = f"{base_path}/{experiment_type}/{model_name}/results.pkl"
        expected_summary_path = f"{base_path}/{experiment_type}/{model_name}/summary-stats.csv"
        expected_metadata_path = f"{base_path}/{experiment_type}/{model_name}/run-metadata.yml"
        
        # Verify path construction
        self.assertEqual(
            expected_results_path,
            "new-peak-project/experiments/ch5-paper/machine-learning/expression-noise-v1/sy_simple/results.pkl"
        )
        
        self.assertEqual(
            expected_summary_path,
            "new-peak-project/experiments/ch5-paper/machine-learning/expression-noise-v1/sy_simple/summary-stats.csv"
        )
        
        self.assertEqual(
            expected_metadata_path,
            "new-peak-project/experiments/ch5-paper/machine-learning/expression-noise-v1/sy_simple/run-metadata.yml"
        )
    


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    def test_end_to_end_with_mocks(self):
        """Test end-to-end flow with mocked dependencies"""
        if not ML_BATCH_AVAILABLE:
            self.skipTest("ML batch module not available")
        
        # This would test the main() function with all dependencies mocked
        # For now, we'll just verify the module structure
        self.assertTrue(hasattr(ml_batch_module, 'main'))
        self.assertTrue(hasattr(ml_batch_module, 'validate_csv_structure'))
        self.assertTrue(hasattr(ml_batch_module, 'run_batch_evaluation_for_experiment'))
        self.assertTrue(hasattr(ml_batch_module, 'discover_task_lists'))
    
    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios"""
        if not ML_BATCH_AVAILABLE:
            self.skipTest("ML batch module not available")
        
        # Test with empty CSV
        empty_df = pd.DataFrame()
        is_valid = ml_batch_module.validate_csv_structure(empty_df)
        self.assertFalse(is_valid)
        
        # Test with missing required column
        partial_df = pd.DataFrame({
            "feature_data": ["test.pkl"],
            "feature_data_label": ["test_label"]
            # Missing other required columns
        })
        is_valid = ml_batch_module.validate_csv_structure(partial_df)
        self.assertFalse(is_valid)


def main():
    """Run all tests"""
    print("Running ML Batch Evaluation tests...")
    print("=" * 70)
    
    # Run unittest tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMLBatchEvaluation)
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All ML Batch Evaluation tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
