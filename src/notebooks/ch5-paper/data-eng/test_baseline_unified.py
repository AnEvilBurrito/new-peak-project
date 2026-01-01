"""
Test script to verify the unified baseline architecture works correctly.

This script tests both parameter-distortion-v2.py and expression-noise-v1.py
with small configurations to ensure they work with the new baseline generator.
"""

import sys
import os
import logging

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_baseline_generator():
    """Test the baseline generator module."""
    logger.info("Testing baseline generator module...")
    
    try:
        from baseline_generator import generate_baseline_virtual_models
        from models.utils.s3_config_manager import S3ConfigManager
        from models.Solver.RoadrunnerSolver import RoadrunnerSolver
        
        # Load model objects
        s3_manager = S3ConfigManager()
        model_name = "sy_simple"
        gen_path = s3_manager.save_result_path
        
        model_spec = s3_manager.load_data_from_path(
            f"{gen_path}/models/{model_name}/model_spec.pkl", 
            data_format='pkl'
        )
        
        model_builder = s3_manager.load_data_from_path(
            f"{gen_path}/models/{model_name}/model_builder.pkl", 
            data_format='pkl'
        )
        
        # Setup solver
        solver = RoadrunnerSolver()
        solver.compile(model_builder.get_sbml_model())
        
        # Generate baseline virtual models with small sample size
        baseline_data = generate_baseline_virtual_models(
            model_spec=model_spec,
            model_builder=model_builder,
            solver=solver,
            n_samples=10,  # Small for testing
            seed=42,
            simulation_params={'start': 0, 'end': 10000, 'points': 101}
        )
        
        # Check results
        assert 'features' in baseline_data
        assert 'targets' in baseline_data
        assert 'parameters' in baseline_data
        assert 'timecourses' in baseline_data
        
        logger.info(f"✅ Baseline generator test passed:")
        logger.info(f"  Features shape: {baseline_data['features'].shape}")
        logger.info(f"  Targets shape: {baseline_data['targets'].shape}")
        logger.info(f"  Parameters shape: {baseline_data['parameters'].shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Baseline generator test failed: {e}")
        return False


def test_parameter_distortion():
    """Test parameter-distortion-v2.py with small configuration."""
    logger.info("Testing parameter distortion with baseline...")
    
    try:
        # Import the module using importlib for hyphenated filename
        import importlib.util
        import sys
        
        # Get the path to parameter-distortion-v2.py
        pd_path = os.path.join(current_dir, "parameter-distortion-v2.py")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("parameter_distortion_v2", pd_path)
        pd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pd_module)
        sys.modules["parameter_distortion_v2"] = pd_module
        
        # Temporarily modify configuration for testing
        original_samples = pd_module.N_SAMPLES
        original_upload = pd_module.UPLOAD_S3
        
        pd_module.N_SAMPLES = 5  # Very small for testing
        pd_module.UPLOAD_S3 = False
        pd_module.DISTORTION_FACTORS = [0, 0.1]  # Only test 2 levels
        
        # Test with mock S3 manager
        class MockS3Manager:
            def __init__(self):
                self.save_result_path = "mock_path"
                self.data = {}
            
            def load_data_from_path(self, path, data_format='pkl'):
                # Mock loading of model objects
                if "model_spec.pkl" in path:
                    from models.utils.s3_config_manager import S3ConfigManager
                    real_s3 = S3ConfigManager()
                    model_name = "sy_simple"
                    gen_path = real_s3.save_result_path
                    return real_s3.load_data_from_path(
                        f"{gen_path}/models/{model_name}/model_spec.pkl", 
                        data_format='pkl'
                    )
                elif "model_builder.pkl" in path:
                    from models.utils.s3_config_manager import S3ConfigManager
                    real_s3 = S3ConfigManager()
                    model_name = "sy_simple"
                    gen_path = real_s3.save_result_path
                    return real_s3.load_data_from_path(
                        f"{gen_path}/models/{model_name}/model_builder.pkl", 
                        data_format='pkl'
                    )
                elif "model_tuner.pkl" in path:
                    from models.utils.s3_config_manager import S3ConfigManager
                    real_s3 = S3ConfigManager()
                    model_name = "sy_simple"
                    gen_path = real_s3.save_result_path
                    return real_s3.load_data_from_path(
                        f"{gen_path}/models/{model_name}/model_tuner.pkl", 
                        data_format='pkl'
                    )
                return None
            
            def save_data_from_path(self, path, data, data_format="pkl"):
                # Mock save - just store in memory
                self.data[path] = data
        
        mock_s3 = MockS3Manager()
        
        # Run the process_single_model function
        success = pd_module.process_single_model("sy_simple", mock_s3)
        
        # Restore original configuration
        pd_module.N_SAMPLES = original_samples
        pd_module.UPLOAD_S3 = original_upload
        
        if success:
            logger.info("✅ Parameter distortion test passed")
            return True
        else:
            logger.error("❌ Parameter distortion test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Parameter distortion test failed: {e}")
        return False


def test_expression_noise():
    """Test expression-noise-v1.py with small configuration."""
    logger.info("Testing expression noise with baseline...")
    
    try:
        # Import the module using importlib for hyphenated filename
        import importlib.util
        import sys
        
        # Get the path to expression-noise-v1.py
        en_path = os.path.join(current_dir, "expression-noise-v1.py")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("expression_noise_v1", en_path)
        en_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(en_module)
        sys.modules["expression_noise_v1"] = en_module
        
        # Temporarily modify configuration for testing
        original_samples = en_module.N_SAMPLES
        original_upload = en_module.UPLOAD_S3
        
        en_module.N_SAMPLES = 5  # Very small for testing
        en_module.UPLOAD_S3 = False
        en_module.NOISE_LEVELS = [0, 0.1]  # Only test 2 levels
        
        # Test with mock S3 manager
        class MockS3Manager:
            def __init__(self):
                self.save_result_path = "mock_path"
                self.data = {}
            
            def load_data_from_path(self, path, data_format='pkl'):
                # Mock loading of model objects
                if "model_spec.pkl" in path:
                    from models.utils.s3_config_manager import S3ConfigManager
                    real_s3 = S3ConfigManager()
                    model_name = "sy_simple"
                    gen_path = real_s3.save_result_path
                    return real_s3.load_data_from_path(
                        f"{gen_path}/models/{model_name}/model_spec.pkl", 
                        data_format='pkl'
                    )
                elif "model_builder.pkl" in path:
                    from models.utils.s3_config_manager import S3ConfigManager
                    real_s3 = S3ConfigManager()
                    model_name = "sy_simple"
                    gen_path = real_s3.save_result_path
                    return real_s3.load_data_from_path(
                        f"{gen_path}/models/{model_name}/model_builder.pkl", 
                        data_format='pkl'
                    )
                elif "model_tuner.pkl" in path:
                    from models.utils.s3_config_manager import S3ConfigManager
                    real_s3 = S3ConfigManager()
                    model_name = "sy_simple"
                    gen_path = real_s3.save_result_path
                    return real_s3.load_data_from_path(
                        f"{gen_path}/models/{model_name}/model_tuner.pkl", 
                        data_format='pkl'
                    )
                return None
            
            def save_data_from_path(self, path, data, data_format="pkl"):
                # Mock save - just store in memory
                self.data[path] = data
        
        mock_s3 = MockS3Manager()
        
        # Run the process_single_model function
        success = en_module.process_single_model("sy_simple", mock_s3)
        
        # Restore original configuration
        en_module.N_SAMPLES = original_samples
        en_module.UPLOAD_S3 = original_upload
        
        if success:
            logger.info("✅ Expression noise test passed")
            return True
        else:
            logger.error("❌ Expression noise test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Expression noise test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("UNIFIED BASELINE ARCHITECTURE TEST SUITE")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Baseline generator
    test_results.append(("Baseline Generator", test_baseline_generator()))
    
    # Test 2: Parameter distortion
    test_results.append(("Parameter Distortion", test_parameter_distortion()))
    
    # Test 3: Expression noise
    test_results.append(("Expression Noise", test_expression_noise()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Unified baseline architecture is working.")
    else:
        logger.error("⚠️ SOME TESTS FAILED! Check the logs above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
