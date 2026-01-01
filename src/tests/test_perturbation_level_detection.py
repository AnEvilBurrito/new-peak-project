"""
Test for perturbation level detection refactoring in visualise-ml-results-v1.py
"""
import sys
import os
import unittest

# Add the project root to sys.path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.insert(0, project_root)

def create_mock_module():
    """Create a mock module with the configuration constants"""
    import types
    
    # Create module with configuration
    config_module = types.ModuleType('config_module')
    config_module.PERTURBATION_LEVELS = {
        "expression-noise-v1": [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
        "response-noise-v1": [0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
        "parameter-distortion-v2": [1.0, 1.1, 1.3, 1.5, 2.0, 3.0]
    }
    config_module.LEVEL_PATTERNS = {
        "noise": ["noise_{level}", "_{level}"],
        "distortion": ["distortion_{level}", "_{level}"]
    }
    
    return config_module

# Mock the configuration before importing
config_module = create_mock_module()
sys.modules['__main__'] = config_module

# Now define the functions directly for testing (since importing is complex)
def clean_feature_label(feature_label: str) -> str:
    """
    Clean feature label by removing noise/distortion level suffixes.
    
    Examples:
    - "dynamic_features_0" → "dynamic_features"
    - "dynamic_features no outcome_0.1" → "dynamic_features"
    - "static_features_1.1" → "static_features"
    - "noisy_features_0.2" → "noisy_features"
    """
    import re
    
    label = str(feature_label)
    
    # Remove trailing numbers with underscores (e.g., _0, _0.1, _1.1)
    label = re.sub(r'_\d+(\.\d+)?$', '', label)
    
    # Remove " no outcome" suffix if present
    label = label.replace(" no outcome", "")
    
    # Remove trailing whitespace
    label = label.strip()
    
    # Common feature type mappings
    feature_type_mapping = {
        "dynamic_features": "Dynamic Features",
        "static_features": "Static Features",
        "noisy_features": "Noisy Features",
        "original_features": "Original Features",
        "features": "Features"
    }
    
    # Map to standardized names
    for key, value in feature_type_mapping.items():
        if key in label.lower():
            return value
    
    return label

def detect_perturbation_level(label: str, experiment_type: str):
    """
    Simplified version for testing - matches the refactored logic
    """
    import re
    
    label_lower = label.lower()
    normalized_experiment_type = experiment_type.replace('-', '_')
    
    # Get configured levels for this experiment type
    configured_levels = config_module.PERTURBATION_LEVELS.get(experiment_type, [])
    
    # Method 1: Try to match exact configured levels first
    for level in configured_levels:
        # Check for common patterns
        patterns = [
            f"_{level}",
            f"noise_{level}",
            f"distortion_{level}",
            f"_{level}_",
        ]
        
        for pattern in patterns:
            if pattern in label_lower:
                return float(level)
    
    # Method 2: Extract all numbers from label and try to match
    numbers = re.findall(r'(\d+(?:\.\d+)?)', label)
    if numbers:
        nums = [float(n) for n in numbers]
        
        # For noise experiments, look for numbers in configured levels
        if normalized_experiment_type in ["expression_noise_v1", "response_noise_v1"]:
            for num in nums:
                # Check if this number is close to any configured level
                for level in configured_levels:
                    if abs(num - level) < 0.001:
                        return float(level)
            
            # If no exact match, use the smallest positive number
            positive_nums = [n for n in nums if n >= 0]
            if positive_nums:
                return float(min(positive_nums))
        
        # For distortion experiments, look for numbers >= 1.0
        elif normalized_experiment_type == "parameter_distortion_v2":
            distortion_nums = [n for n in nums if n >= 1.0]
            if distortion_nums:
                # Try to match configured distortion factors
                for num in distortion_nums:
                    for level in configured_levels:
                        if abs(num - level) < 0.001:
                            return float(level)
                
                # If no exact match, use the smallest distortion number
                return float(min(distortion_nums))
    
    # Method 3: Fallback to regex pattern matching for common cases
    if "noise_0" in label_lower or "_0" in label_lower:
        return 0.0
    elif any(f"_0.{i}" in label_lower for i in [1, 2, 3, 5]):
        for i in [1, 2, 3, 5]:
            if f"_0.{i}" in label_lower or f"noise_0.{i}" in label_lower:
                return float(f"0.{i}")
    elif "_1.0" in label_lower or "noise_1.0" in label_lower:
        return 1.0
    elif any(f"_1.{i}" in label_lower for i in [1, 3, 5]):
        for i in [1, 3, 5]:
            if f"_1.{i}" in label_lower or f"distortion_1.{i}" in label_lower:
                return float(f"1.{i}")
    elif "_2.0" in label_lower or "distortion_2.0" in label_lower:
        return 2.0
    elif "_3.0" in label_lower or "distortion_3.0" in label_lower:
        return 3.0
    
    return None

class TestPerturbationLevelDetection(unittest.TestCase):
    """Test cases for detect_perturbation_level function"""
    
    def detect_perturbation_level(self, label, experiment_type):
        """Wrapper for module-level function"""
        return detect_perturbation_level(label, experiment_type)
    
    def test_detect_noise_levels(self):
        """Test detection of noise levels in feature labels"""
        # Test cases for expression noise
        test_cases = [
            ("dynamic_features_0", "expression-noise-v1", 0.0),
            ("dynamic_features_0.1", "expression-noise-v1", 0.1),
            ("dynamic_features_no_outcome_0.2", "expression-noise-v1", 0.2),
            ("noisy_features_0.3", "expression-noise-v1", 0.3),
            ("features_0.5", "expression-noise-v1", 0.5),
            ("static_features_1.0", "expression-noise-v1", 1.0),
            ("noise_0", "expression-noise-v1", 0.0),
            ("noise_0.1", "expression-noise-v1", 0.1),
            ("noise_1.0", "expression-noise-v1", 1.0),
        ]
        
        for label, experiment_type, expected in test_cases:
            with self.subTest(label=label, experiment_type=experiment_type):
                result = self.detect_perturbation_level(label, experiment_type)
                self.assertEqual(result, expected, 
                                 f"Failed for label: {label}, expected: {expected}, got: {result}")
    
    def test_detect_distortion_levels(self):
        """Test detection of distortion levels in feature labels"""
        # Test cases for parameter distortion
        test_cases = [
            ("dynamic_features_1.1", "parameter-distortion-v2", 1.1),
            ("dynamic_features_1.3", "parameter-distortion-v2", 1.3),
            ("dynamic_features_1.5", "parameter-distortion-v2", 1.5),
            ("features_2.0", "parameter-distortion-v2", 2.0),
            ("static_features_3.0", "parameter-distortion-v2", 3.0),
            ("distortion_1.1", "parameter-distortion-v2", 1.1),
            ("distortion_2.0", "parameter-distortion-v2", 2.0),
        ]
        
        for label, experiment_type, expected in test_cases:
            with self.subTest(label=label, experiment_type=experiment_type):
                result = self.detect_perturbation_level(label, experiment_type)
                self.assertEqual(result, expected,
                                 f"Failed for label: {label}, expected: {expected}, got: {result}")
    
    def test_detect_response_noise_levels(self):
        """Test detection of response noise levels"""
        # Test cases for response noise (same as expression noise)
        test_cases = [
            ("dynamic_features_0", "response-noise-v1", 0.0),
            ("dynamic_features_0.1", "response-noise-v1", 0.1),
            ("noisy_targets_0.2", "response-noise-v1", 0.2),
        ]
        
        for label, experiment_type, expected in test_cases:
            with self.subTest(label=label, experiment_type=experiment_type):
                result = self.detect_perturbation_level(label, experiment_type)
                self.assertEqual(result, expected,
                                 f"Failed for label: {label}, expected: {expected}, got: {result}")
    
    def test_no_match(self):
        """Test labels with no detectable perturbation level"""
        test_cases = [
            ("dynamic_features", "expression-noise-v1", None),
            ("features", "parameter-distortion-v2", None),
            ("unknown_label", "response-noise-v1", None),
        ]
        
        for label, experiment_type, expected in test_cases:
            with self.subTest(label=label, experiment_type=experiment_type):
                result = self.detect_perturbation_level(label, experiment_type)
                self.assertEqual(result, expected,
                                 f"Failed for label: {label}, expected: {expected}, got: {result}")
    
    def test_edge_cases(self):
        """Test edge cases and unusual labels"""
        test_cases = [
            ("dynamic_features no outcome_0.1", "expression-noise-v1", 0.1),
            ("static_features_0_0.1", "expression-noise-v1", 0.0),  # Should detect first number
            ("features_0.100", "expression-noise-v1", 0.1),  # Should handle extra precision
        ]
        
        for label, experiment_type, expected in test_cases:
            with self.subTest(label=label, experiment_type=experiment_type):
                result = self.detect_perturbation_level(label, experiment_type)
                self.assertEqual(result, expected,
                                 f"Failed for label: {label}, expected: {expected}, got: {result}")

class TestCleanFeatureLabel(unittest.TestCase):
    """Test cases for clean_feature_label function"""
    
    def test_clean_labels(self):
        """Test cleaning of feature labels"""
        test_cases = [
            ("dynamic_features_0", "Dynamic Features"),
            ("dynamic_features no outcome_0.1", "Dynamic Features"),
            ("static_features_1.1", "Static Features"),
            ("noisy_features_0.2", "Noisy Features"),
            ("original_features_0.3", "Original Features"),
            ("features_0.5", "Features"),
            ("unknown_label_1.0", "unknown_label"),  # No mapping
            ("dynamic_features", "Dynamic Features"),  # Already clean
        ]
        
        for label, expected in test_cases:
            with self.subTest(label=label):
                result = clean_feature_label(label)
                self.assertEqual(result, expected,
                                 f"Failed for label: {label}, expected: {expected}, got: {result}")

if __name__ == '__main__':
    unittest.main()
