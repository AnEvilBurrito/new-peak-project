import sys
import os

# Add src to path
sys.path.insert(0, 'src')

# Run pytest
import pytest

# Run specific test file
result = pytest.main(['src/tests/models/unit/test_parameter_optimizer.py', '-v'])
sys.exit(result)
