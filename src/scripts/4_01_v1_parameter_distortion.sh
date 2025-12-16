#!/bin/bash

# Batch Parameter Distortion Execution Script
# Section 4 / Experiment 01 / Version 1

echo "🚀 Starting Batch Parameter Distortion Execution via Shell Script"

# Set script start time for shell-level timing
SCRIPT_START_TIME=$(date +%s)

# Activate uv environment
echo "🔧 Activating uv environment..."
source ../.venv/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate uv environment"
    echo "💡 Trying alternative activation path..."
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate environment. Please check your uv setup."
        exit 1
    fi
fi

echo "✅ Environment activated successfully"

# Set environment variables based on .env configuration
echo "🔧 Setting environment variables..."
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Execute the Python batch script
echo "🚀 Executing Python batch script..."
python 4_01_v1_batch-parameter-distortion.py

# Capture Python script exit code
PYTHON_EXIT_CODE=$?

# Calculate total execution time
SCRIPT_END_TIME=$(date +%s)
SCRIPT_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

# Convert to minutes and seconds
MINUTES=$((SCRIPT_DURATION / 60))
SECONDS=$((SCRIPT_DURATION % 60))

echo "📊 Shell Script Execution Summary:"
echo "  - Start Time: $(date -d @$SCRIPT_START_TIME)"
echo "  - End Time: $(date -d @$SCRIPT_END_TIME)"
echo "  - Total Duration: ${MINUTES}m ${SECONDS}s"
echo "  - Python Exit Code: $PYTHON_EXIT_CODE"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ Batch execution completed successfully via shell script"
else
    echo "❌ Batch execution failed with exit code: $PYTHON_EXIT_CODE"
fi

# Deactivate environment
echo "🔧 Deactivating environment..."
deactivate

echo "✅ Shell script execution completed"
