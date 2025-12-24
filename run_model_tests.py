#!/usr/bin/env python
"""
Test runner for ModelBuilder utility tests.
Runs only the model tests with proper Python path configuration.
"""
import sys
import os

# Add src to Python path so we can import models
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

import pytest

def run_model_tests():
    """Run only the model tests."""
    print("=" * 80)
    print("Running ModelBuilder Utility Tests")
    print("=" * 80)
    print(f"Python path: {sys.path[0]}")
    print(f"Test directory: src/models/tests/")
    print("=" * 80)
    
    # Run pytest on the model tests directory
    test_dir = "src/models/tests/"
    
    # Use these arguments:
    # -v: verbose output
    # --tb=short: shorter tracebacks
    # --disable-warnings: disable warning messages
    args = [
        test_dir,
        '-v',
        '--tb=short',
        '--disable-warnings',
    ]
    
    exit_code = pytest.main(args)
    
    print("=" * 80)
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    print("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(run_model_tests())
