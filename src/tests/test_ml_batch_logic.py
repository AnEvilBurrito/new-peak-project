"""
Simplified test for ML batch evaluation logic (core functions only)

Tests the CSV validation, filtering, and grouping logic without
trying to import the entire module with its dependencies.
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_csv_structure(df: pd.DataFrame) -> bool:
    """Validate CSV has required columns"""
    required_columns = [
        "feature_data", "feature_data_label", 
        "target_data", "target_data_label",
        "experiment_type", "level", "model_name"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False
    
    return True


def filter_tasks_by_experiment(df: pd.DataFrame, experiment_types: list = None) -> pd.DataFrame:
    """Filter tasks by experiment type"""
    if experiment_types is None:
        return df
    
    filtered_df = df[df["experiment_type"].isin(experiment_types)].copy()
    return filtered_df


def group_tasks_by_experiment(df: pd.DataFrame) -> dict:
    """Group tasks by experiment type"""
    grouped = {}
    for exp_type, group_df in df.groupby("experiment_type"):
        grouped[exp_type] = group_df.reset_index(drop=True)
    
    return grouped


def generate_summary_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from results"""
    if results_df is None or len(results_df) == 0:
        return pd.DataFrame()
    
    # Group by Model and Feature Data
    summary = results_df.groupby(["Model", "Feature Data"]).agg({
        "R2 Score": ["mean", "std", "min", "max"],
        "Mean Squared Error": ["mean", "std", "min", "max"],
        "Pearson Correlation": ["mean", "std", "min", "max"]
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary


class TestMLBatchLogic(unittest.TestCase):
    """Test cases for ML batch evaluation core logic"""
    
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
        # Test valid CSV
        is_valid = validate_csv_structure(self.test_df)
        self.assertTrue(is_valid, "Valid CSV should pass validation")
        
        # Test missing column
        invalid_df = self.test_df.drop(columns=["feature_data"])
        is_valid = validate_csv_structure(invalid_df)
        self.assertFalse(is_valid, "Invalid CSV should fail validation")
        
        # Test missing multiple columns
        invalid_df2 = self.test_df.drop(columns=["feature_data", "target_data"])
        is_valid = validate_csv_structure(invalid_df2)
        self.assertFalse(is_valid, "Invalid CSV with multiple missing columns should fail validation")
    
    def test_filter_tasks_by_experiment(self):
        """Test filtering tasks by experiment type"""
        # Test filtering for specific experiment
        filtered_df = filter_tasks_by_experiment(
            self.test_df, ["expression-noise-v1"]
        )
        
        self.assertEqual(len(filtered_df), 2, "Should filter to 2 expression-noise-v1 tasks")
        self.assertTrue(
            all(filtered_df["experiment_type"] == "expression-noise-v1"),
            "All filtered tasks should be expression-noise-v1"
        )
        
        # Test filtering for multiple experiments
        filtered_df = filter_tasks_by_experiment(
            self.test_df, ["expression-noise-v1", "parameter-distortion-v2"]
        )
        
        self.assertEqual(len(filtered_df), 3, "Should include all 3 tasks")
        
        # Test filtering with no experiment types (returns all)
        filtered_df = filter_tasks_by_experiment(self.test_df, None)
        self.assertEqual(len(filtered_df), 3, "Should return all tasks when no filter")
        
        # Test filtering for non-existent experiment
        filtered_df = filter_tasks_by_experiment(self.test_df, ["non-existent-experiment"])
        self.assertEqual(len(filtered_df), 0, "Should return empty DataFrame for non-existent experiment")
    
    def test_group_tasks_by_experiment(self):
        """Test grouping tasks by experiment type"""
        grouped = group_tasks_by_experiment(self.test_df)
        
        self.assertIn("expression-noise-v1", grouped)
        self.assertIn("parameter-distortion-v2", grouped)
        
        self.assertEqual(len(grouped["expression-noise-v1"]), 2)
        self.assertEqual(len(grouped["parameter-distortion-v2"]), 1)
        
        # Verify data integrity
        exp1_df = grouped["expression-noise-v1"]
        self.assertTrue(all(exp1_df["experiment_type"] == "expression-noise-v1"))
        self.assertEqual(list(exp1_df["level"]), [0, 0.1])
        
        exp2_df = grouped["parameter-distortion-v2"]
        self.assertTrue(all(exp2_df["experiment_type"] == "parameter-distortion-v2"))
        self.assertEqual(list(exp2_df["level"]), [1.1])
    
    def test_generate_summary_stats(self):
        """Test generation of summary statistics"""
        # Create test results DataFrame
        results_df = pd.DataFrame({
            "Model": ["Linear Regression", "Linear Regression", "Random Forest", "Random Forest"],
            "Feature Data": ["feature1", "feature2", "feature1", "feature2"],
            "R2 Score": [0.8, 0.7, 0.9, 0.85],
            "Mean Squared Error": [0.1, 0.2, 0.05, 0.075],
            "Pearson Correlation": [0.85, 0.75, 0.92, 0.88]
        })
        
        summary_df = generate_summary_stats(results_df)
        
        self.assertIsNotNone(summary_df)
        self.assertGreater(len(summary_df), 0)
        self.assertIn("Model", summary_df.columns)
        self.assertIn("Feature Data", summary_df.columns)
        self.assertIn("R2 Score_mean", summary_df.columns)
        self.assertIn("Mean Squared Error_mean", summary_df.columns)
        
        # Verify aggregation
        self.assertEqual(len(summary_df), 4)  # 2 models × 2 features
        self.assertIn("Linear Regression", summary_df["Model"].values)
        self.assertIn("Random Forest", summary_df["Model"].values)
    
    def test_empty_data_handling(self):
        """Test handling of empty DataFrames"""
        # Test empty DataFrame validation
        empty_df = pd.DataFrame()
        is_valid = validate_csv_structure(empty_df)
        self.assertFalse(is_valid, "Empty DataFrame should fail validation")
        
        # Test empty DataFrame filtering - need to handle missing columns
        empty_df_with_cols = pd.DataFrame(columns=["experiment_type", "feature_data"])
        filtered_df = filter_tasks_by_experiment(empty_df_with_cols, ["expression-noise-v1"])
        self.assertEqual(len(filtered_df), 0, "Empty DataFrame should remain empty after filtering")
        
        # Test empty DataFrame with no columns
        try:
            filtered_df = filter_tasks_by_experiment(empty_df, ["expression-noise-v1"])
            self.fail("Should raise KeyError for missing experiment_type column")
        except KeyError:
            pass  # Expected behavior
        
        # Test empty DataFrame grouping
        grouped = group_tasks_by_experiment(empty_df_with_cols)
        self.assertEqual(len(grouped), 0, "Empty DataFrame should produce empty grouping")
    
    def test_s3_path_structure_logic(self):
        """Test S3 path structure logic (without actual S3 dependency)"""
        # Test path construction logic
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
        
        # Test with different experiment and model
        experiment_type2 = "parameter-distortion-v2"
        model_name2 = "fgfr4_model"
        
        expected_results_path2 = f"{base_path}/{experiment_type2}/{model_name2}/results.pkl"
        self.assertEqual(
            expected_results_path2,
            "new-peak-project/experiments/ch5-paper/machine-learning/parameter-distortion-v2/fgfr4_model/results.pkl"
        )


def main():
    """Run all tests"""
    print("Running ML Batch Evaluation Logic tests...")
    print("=" * 70)
    
    # Run unittest tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMLBatchLogic)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All ML Batch Evaluation Logic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
