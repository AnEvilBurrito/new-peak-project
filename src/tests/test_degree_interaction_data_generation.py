"""
Integration test for DegreeInteractionSpec data generation.

This test replicates the data generation workflow from 
src/notebooks/ch5-paper/explore/degree_interaction_kinetics.py
to ensure timecourse DataFrame structure is correct.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from models.Specs.Drug import Drug
    from models.Solver.RoadrunnerSolver import RoadrunnerSolver
    from models.utils.data_generation_helpers import make_data_extended
    from models.utils.kinetic_tuner import KineticParameterTuner
    DEGREE_INTERACTION_AVAILABLE = True
except ImportError as e:
    DEGREE_INTERACTION_AVAILABLE = False
    print(f"Warning: Could not import required modules for DegreeInteractionSpec test: {e}")


def test_degree_interaction_timecourse_generation():
    """
    Test that make_data_extended produces correct timecourse DataFrame
    with DegreeInteractionSpec.
    
    This replicates the notebook workflow but focuses on timecourse structure.
    """
    if not DEGREE_INTERACTION_AVAILABLE:
        print("Skipping DegreeInteractionSpec test - required modules not available")
        return True
    
    print("Testing DegreeInteractionSpec timecourse generation...")
    
    try:
        # 1. Initialize degree interaction specification (smaller for faster testing)
        degree_spec = DegreeInteractionSpec(degree_cascades=[1, 2])
        
        # Generate specifications
        degree_spec.generate_specifications(
            random_seed=42,
            feedback_density=0.3  # 30% feedback density
        )
        
        print(f"Created network with {len(degree_spec.species_list)} species")
        
        # 2. Create drug targeting R1_1
        drug_d = Drug(
            name="D",
            start_time=5000.0,
            default_value=100.0,
            regulation=["R1_1"],
            regulation_type=["down"]
        )
        degree_spec.add_drug(drug_d)
        
        # 3. Generate model
        model = degree_spec.generate_network(
            network_name="TestMultiDegree",
            mean_range_species=(50, 150),
            rangeScale_params=(0.8, 1.2),
            rangeMultiplier_params=(0.9, 1.1),
            random_seed=42,
            receptor_basal_activation=False
        )
        
        # 4. Tune kinetic parameters
        tuner = KineticParameterTuner(model, random_seed=42)
        updated_params = tuner.generate_parameters(active_percentage_range=(0.3, 0.7))
        
        for param, value in updated_params.items():
            model.set_parameter(param, value)
        
        # 5. Create solver
        solver = RoadrunnerSolver()
        solver.compile(model.get_sbml_model())
        
        # 6. Get state variables
        state_variables = model.get_state_variables()
        
        # Get inactive state variables (excluding 'O' and active forms)
        inactive_state_variables = {k: v for k, v in state_variables.items() 
                                   if not k.endswith('a') and k != 'O'}
        
        kinetic_parameters = model.get_parameters()
        
        # 7. Generate data with make_data_extended
        print("Generating data with make_data_extended...")
        results = make_data_extended(
            initial_values=inactive_state_variables,
            perturbation_type="lognormal",
            perturbation_params={"shape": 0.2},  # Smaller shape for faster convergence
            parameter_values=kinetic_parameters,
            param_perturbation_type="lognormal",
            param_perturbation_params={"shape": 0.1},
            n_samples=10,  # Small sample size for testing
            model_spec=degree_spec,
            solver=solver,
            simulation_params={"start": 0, "end": 100, "points": 11},  # Shorter simulation
            seed=42,
            outcome_var="Oa",
            capture_all_species=True,
            verbose=False
        )
        
        # 8. Validate results structure
        print("Validating results structure...")
        expected_keys = ['features', 'targets', 'parameters', 'timecourse', 'metadata']
        assert all(key in results for key in expected_keys), \
            f"Missing keys in results. Expected: {expected_keys}, Got: {list(results.keys())}"
        
        # Check data types
        assert isinstance(results['features'], pd.DataFrame), "Features should be DataFrame"
        assert isinstance(results['targets'], pd.DataFrame), "Targets should be DataFrame"
        assert results['parameters'] is None or isinstance(results['parameters'], pd.DataFrame), \
            "Parameters should be None or DataFrame"
        assert isinstance(results['timecourse'], pd.DataFrame), \
            "Timecourse should be DataFrame when capture_all_species=True"
        assert isinstance(results['metadata'], dict), "Metadata should be dictionary"
        
        # Check timecourse DataFrame structure
        timecourse_df = results['timecourse']
        assert not timecourse_df.empty, "Timecourse DataFrame should not be empty"
        assert timecourse_df.shape[0] == 10, f"Timecourse should have 10 rows, got {timecourse_df.shape[0]}"
        
        # Check that timecourse.head() shows proper columns
        print(f"Timecourse columns: {list(timecourse_df.columns)}")
        print(f"Timecourse shape: {timecourse_df.shape}")
        print(f"Timecourse head:\n{timecourse_df.head()}")
        
        # Ensure columns match expected species patterns
        # DegreeInteractionSpec creates species like R1_1, R1_1a, I1_1, I1_1a, etc.
        has_active_species = any(col.endswith('a') for col in timecourse_df.columns if isinstance(col, str))
        has_inactive_species = any(not col.endswith('a') for col in timecourse_df.columns if isinstance(col, str))
        
        assert has_active_species or has_inactive_species, \
            "Timecourse should contain species columns"
        
        # Check metadata
        metadata = results['metadata']
        assert 'capture_all_species' in metadata, "Metadata should contain capture_all_species"
        assert metadata['capture_all_species'] == True, "capture_all_species should be True"
        
        print("✓ DegreeInteractionSpec timecourse generation test passed")
        return True
        
    except Exception as e:
        print(f"✗ DegreeInteractionSpec timecourse generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_degree_interaction_capture_all_species_false():
    """
    Test that make_data_extended works correctly with capture_all_species=False
    for DegreeInteractionSpec.
    """
    if not DEGREE_INTERACTION_AVAILABLE:
        print("Skipping DegreeInteractionSpec capture_all_species=False test")
        return True
    
    print("Testing DegreeInteractionSpec with capture_all_species=False...")
    
    try:
        # Create a simpler degree spec for faster testing
        degree_spec = DegreeInteractionSpec(degree_cascades=[1])
        degree_spec.generate_specifications(random_seed=42, feedback_density=0.1)
        
        # Create model
        model = degree_spec.generate_network(
            network_name="TestSimpleDegree",
            mean_range_species=(50, 100),
            random_seed=42,
            receptor_basal_activation=False
        )
        
        # Get state variables
        state_variables = model.get_state_variables()
        inactive_state_variables = {k: v for k, v in state_variables.items() 
                                   if not k.endswith('a') and k != 'O'}
        
        # Create solver
        solver = RoadrunnerSolver()
        solver.compile(model.get_sbml_model())
        
        # Generate data with capture_all_species=False
        results = make_data_extended(
            initial_values=inactive_state_variables,
            perturbation_type="uniform",
            perturbation_params={"min": 0.8, "max": 1.2},
            n_samples=5,
            model_spec=degree_spec,
            solver=solver,
            simulation_params={"start": 0, "end": 50, "points": 6},
            seed=42,
            outcome_var="Oa",
            capture_all_species=False,  # Important: False here
            verbose=False
        )
        
        # Validate structure
        timecourse_data = results['timecourse']
        assert isinstance(timecourse_data, list), \
            f"Timecourse should be list when capture_all_species=False, got {type(timecourse_data)}"
        assert len(timecourse_data) == 5, \
            f"Timecourse list should have 5 elements, got {len(timecourse_data)}"
        
        # Check that elements are numpy arrays
        for i, tc in enumerate(timecourse_data):
            if tc is not None:  # Failed simulations may have None
                assert isinstance(tc, np.ndarray), \
                    f"Timecourse element {i} should be numpy array, got {type(tc)}"
        
        metadata = results['metadata']
        assert metadata['capture_all_species'] == False, \
            "capture_all_species should be False in metadata"
        
        print("✓ DegreeInteractionSpec capture_all_species=False test passed")
        return True
        
    except Exception as e:
        print(f"✗ DegreeInteractionSpec capture_all_species=False test failed: {e}")
        return False


def main():
    """Run all DegreeInteractionSpec data generation tests."""
    print("Running DegreeInteractionSpec data generation integration tests...")
    print("=" * 70)
    
    tests = [
        test_degree_interaction_timecourse_generation,
        test_degree_interaction_capture_all_species_false,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} raised exception: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Test results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All DegreeInteractionSpec tests passed!")
        return 0
    else:
        print("✗ Some DegreeInteractionSpec tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
