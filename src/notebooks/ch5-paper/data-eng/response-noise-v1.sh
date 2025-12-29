#!/bin/bash

# Batch Parameter Distortion Execution Script
# Section 4 / Experiment 01 / Version 1

# Set script variables with absolute paths
VENV_PATH="/home/dawsonlan/new-peak-project/.venv/bin/activate"
PYTHON_SCRIPT="/home/dawsonlan/new-peak-project/src/notebooks/ch5-paper/data-eng/response-noise-v1.py"
PROJECT_DIR="/home/dawsonlan/new-peak-project"

echo "🚀 Starting Batch Parameter Distortion Execution via Shell Script"

# Set script start time for shell-level timing
SCRIPT_START_TIME=$(date +%s)

# Change to project directory
echo "🔧 Changing to project directory..."
cd "$PROJECT_DIR" || { echo "❌ Failed to change to project directory"; exit 1; }

# Check if virtual environment exists
if [ ! -f "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Source the virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_PATH"

# Set environment variables based on .env configuration
echo "🔧 Setting environment variables..."
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Execute the Python batch script
echo "🚀 Executing Python batch script..."
python "$PYTHON_SCRIPT"

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
