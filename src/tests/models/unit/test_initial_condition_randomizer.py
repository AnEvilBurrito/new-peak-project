"""
Unit tests for initial_condition_randomizer module.
"""
import pytest
import re
from models.utils.initial_condition_randomizer import InitialConditionRandomizer


class TestInitialConditionRandomizer:
    """Test initial condition randomization utilities."""
    
    def test_initialization(self, multi_reaction_model):
        """Test InitialConditionRandomizer initialization."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Should have model and default ranges
        assert randomizer.model is multi_reaction_model
        assert len(randomizer.state_ranges) == 0  # No specific state ranges yet
        assert len(randomizer.pattern_ranges) > 0  # Should have default patterns
        
        # Check default patterns exist
        patterns = [info['pattern'] for info in randomizer.pattern_ranges]
        assert "*" in patterns  # Default pattern
    
    def test_initialization_uncompiled_model(self):
        """Test that uncompiled models raise error."""
        uncompiled_model = pytest.importorskip('models.ModelBuilder').ModelBuilder("uncompiled")
        
        with pytest.raises(ValueError, match="must be pre-compiled"):
            InitialConditionRandomizer(uncompiled_model)
    
    def test_set_range_for_state(self, multi_reaction_model):
        """Test setting range for specific state."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Set range for R1
        randomizer.set_range_for_state('R1', 100.0, 500.0)
        assert randomizer.state_ranges['R1'] == (100.0, 500.0)
        
        # Set range for Oa
        randomizer.set_range_for_state('Oa', 0.0, 50.0)
        assert randomizer.state_ranges['Oa'] == (0.0, 50.0)
    
    def test_set_range_for_state_invalid(self, multi_reaction_model):
        """Test invalid state range settings."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # min_val >= max_val
        with pytest.raises(ValueError, match="must be less than"):
            randomizer.set_range_for_state('R1', 500.0, 100.0)
        
        # Non-existent state
        with pytest.raises(ValueError, match="not found in model"):
            randomizer.set_range_for_state('Nonexistent', 0.0, 100.0)
    
    def test_set_range_for_pattern(self, multi_reaction_model):
        """Test setting range for pattern."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Clear existing patterns for testing
        randomizer.pattern_ranges = []
        
        # Set pattern for receptors
        randomizer.set_range_for_pattern('R*', 100.0, 500.0)
        assert len(randomizer.pattern_ranges) == 1
        pattern_info = randomizer.pattern_ranges[0]
        assert pattern_info['pattern'] == 'R*'
        assert pattern_info['min_val'] == 100.0
        assert pattern_info['max_val'] == 500.0
        
        # Set pattern for activated forms
        randomizer.set_range_for_pattern('*a', 0.0, 50.0)
        assert len(randomizer.pattern_ranges) == 2
    
    def test_set_range_for_pattern_invalid(self, multi_reaction_model):
        """Test invalid pattern range settings."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        with pytest.raises(ValueError, match="must be less than"):
            randomizer.set_range_for_pattern('R*', 500.0, 100.0)
    
    def test_get_range_for_state(self, multi_reaction_model):
        """Test getting range for state with priority."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Clear existing patterns for testing
        randomizer.pattern_ranges = []
        
        # Add pattern range first
        randomizer.set_range_for_pattern('R*', 100.0, 500.0)
        randomizer.set_range_for_pattern('*', 1.0, 100.0)  # Default
        
        # Test pattern matching
        assert randomizer._get_range_for_state('R1') == (100.0, 500.0)
        assert randomizer._get_range_for_state('R2') == (100.0, 500.0)
        assert randomizer._get_range_for_state('O') == (1.0, 100.0)  # Matches default
        
        # Add specific state range (should override pattern)
        randomizer.set_range_for_state('R1', 200.0, 400.0)
        assert randomizer._get_range_for_state('R1') == (200.0, 400.0)  # Specific overrides pattern
        assert randomizer._get_range_for_state('R2') == (100.0, 500.0)  # Still uses pattern
    
    def test_randomize_initial_conditions(self, multi_reaction_model, rng_seed):
        """Test randomizing all initial conditions."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Clear defaults and set specific ranges for testing
        randomizer.pattern_ranges = []
        randomizer.set_range_for_pattern('*', 100.0, 200.0)  # All states 100-200
        
        # Randomize with seed
        randomized_model = randomizer.randomize_initial_conditions(seed=rng_seed)
        
        # Should return new model
        assert randomized_model is not multi_reaction_model
        assert randomized_model.name == multi_reaction_model.name
        
        # Check states were randomized
        original_states = multi_reaction_model.states
        randomized_states = randomized_model.states
        
        assert len(original_states) == len(randomized_states)
        
        # States should be different and within range
        for state_name in original_states:
            orig_val = original_states[state_name]
            rand_val = randomized_states[state_name]
            
            # Values should be different
            assert orig_val != rand_val
            
            # Should be within specified range
            assert 100.0 <= rand_val <= 200.0
    
    def test_randomize_initial_conditions_reproducible(self, multi_reaction_model, rng_seed):
        """Test that randomization is reproducible with same seed."""
        randomizer1 = InitialConditionRandomizer(multi_reaction_model)
        randomizer2 = InitialConditionRandomizer(multi_reaction_model)
        
        # Clear defaults for consistent testing
        randomizer1.pattern_ranges = []
        randomizer2.pattern_ranges = []
        randomizer1.set_range_for_pattern('*', 100.0, 200.0)
        randomizer2.set_range_for_pattern('*', 100.0, 200.0)
        
        # Randomize with same seed
        model1 = randomizer1.randomize_initial_conditions(seed=rng_seed)
        model2 = randomizer2.randomize_initial_conditions(seed=rng_seed)
        
        # States should be identical
        states1 = model1.states
        states2 = model2.states
        
        for state_name in states1:
            assert states1[state_name] == pytest.approx(states2[state_name])
    
    def test_randomize_subset_initial_conditions(self, multi_reaction_model, rng_seed):
        """Test randomizing subset of initial conditions."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Clear defaults
        randomizer.pattern_ranges = []
        randomizer.set_range_for_pattern('*', 100.0, 200.0)
        
        # Randomize only non-activated receptors (R1, R2) using pattern that excludes 'a'
        randomized_model = randomizer.randomize_subset_initial_conditions('R[0-9]', seed=rng_seed)
        
        # Get original and randomized states
        original_states = multi_reaction_model.states
        randomized_states = randomized_model.states
        
        # Only R1 and R2 should change
        for state_name in original_states:
            orig_val = original_states[state_name]
            rand_val = randomized_states[state_name]
            
            if state_name in ['R1', 'R2']:
                assert orig_val != rand_val
                assert 100.0 <= rand_val <= 200.0
            else:
                # Other states (R1a, R2a, O, Oa) should remain unchanged
                assert orig_val == rand_val
    
    def test_randomize_subset_initial_conditions_invalid(self, multi_reaction_model):
        """Test randomizing subset with non-matching pattern."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        with pytest.raises(ValueError, match="No states found"):
            randomizer.randomize_subset_initial_conditions('Nonexistent*')
    
    def test_validate_initial_condition_ranges(self, multi_reaction_model):
        """Test initial condition range validation."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Set tight ranges that likely exclude current values
        randomizer.pattern_ranges = []
        randomizer.set_range_for_pattern('*', 200.0, 300.0)  # Current states: ~100
        
        validation_results = randomizer.validate_initial_condition_ranges()
        
        # All states should be invalid with these ranges
        for state_name, is_valid in validation_results.items():
            assert not is_valid, f"State {state_name} should be invalid"
    
    def test_get_initial_condition_statistics(self, multi_reaction_model):
        """Test initial condition statistics."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Set some ranges for testing
        randomizer.pattern_ranges = []
        randomizer.set_range_for_pattern('R*', 100.0, 500.0)
        randomizer.set_range_for_state('O', 50.0, 150.0)
        
        stats = randomizer.get_initial_condition_statistics()
        
        # Should have stats for patterns
        assert 'R*' in stats
        assert 'O' in stats  # Specific state range
        # Note: 'other' may not be present if all states match patterns
        
        # Check R* stats - pattern 'R*' matches R1, R2, R1a, R2a
        r_stats = stats.get('R*', {})
        if r_stats:
            # Pattern 'R*' should match 4 states: R1, R2, R1a, R2a
            assert r_stats['count'] == 4
            assert 'R1' in r_stats['states']
            assert 'R2' in r_stats['states']
            assert 'R1a' in r_stats['states']
            assert 'R2a' in r_stats['states']
    
    def test_get_state_categories(self, multi_reaction_model):
        """Test state categorization."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        categories = randomizer.get_state_categories()
        
        # Should have expected categories
        assert 'receptors' in categories
        assert 'activated' in categories
        assert 'outcomes' in categories
        
        # Check categorization
        if 'receptors' in categories:
            assert 'R1' in categories['receptors']
            assert 'R2' in categories['receptors']
        
        if 'activated' in categories:
            assert 'R1a' in categories['activated']
            assert 'R2a' in categories['activated']
        
        if 'outcomes' in categories:
            assert 'O' in categories['outcomes']
            assert 'Oa' in categories['outcomes']
    
    def test_set_category_ranges(self, multi_reaction_model):
        """Test setting ranges for entire category."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        # Set range for receptors
        randomizer.set_category_ranges('receptors', 100.0, 500.0)
        
        # Check that ranges were set for R1 and R2
        assert randomizer.state_ranges['R1'] == (100.0, 500.0)
        assert randomizer.state_ranges['R2'] == (100.0, 500.0)
        
        # R1a and R2a should not be affected (they're in 'activated' category)
        assert 'R1a' not in randomizer.state_ranges
        assert 'R2a' not in randomizer.state_ranges
    
    def test_set_category_ranges_invalid(self, multi_reaction_model):
        """Test setting ranges for invalid category."""
        randomizer = InitialConditionRandomizer(multi_reaction_model)
        
        with pytest.raises(ValueError, match="Invalid category"):
            randomizer.set_category_ranges('nonexistent', 0.0, 100.0)
    
    def test_model_spec4_integration(self, model_spec4_example, rng_seed):
        """Test initial condition randomization with real ModelSpec4 model."""
        randomizer = InitialConditionRandomizer(model_spec4_example)
        
        # Clear defaults
        randomizer.pattern_ranges = []
        
        # Set category ranges
        randomizer.set_category_ranges('receptors', 100.0, 500.0)
        randomizer.set_category_ranges('activated', 0.0, 100.0)
        randomizer.set_range_for_pattern('*', 50.0, 200.0)  # Default for others
        
        # Randomize all initial conditions
        randomized_model = randomizer.randomize_initial_conditions(seed=rng_seed)
        
        # Should return new model
        assert randomized_model is not model_spec4_example
        
        # Validate that states are within ranges
        for state_name, value in randomized_model.states.items():
            # Determine expected range based on category
            if re.match(r'^R\d+$', state_name):  # Receptors
                assert 100.0 <= value <= 500.0
            elif re.match(r'^R\d+a$', state_name):  # Activated receptors
                assert 0.0 <= value <= 100.0
            else:  # Others
                assert 50.0 <= value <= 200.0
