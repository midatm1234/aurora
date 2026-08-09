#!/bin/bash
# Run Aurora O3 v4 fine-tuning via papermill in mamba aurora env
# Usage: tmux new -s aurora-o3-v4 "./run_O3_v4_papermill.sh"

set -e  # Exit on error

cd "$(dirname "$0")"  # cd to finetune/ directory

CONFIG_NAME="backup/aurora_O3_finetune_US-WEST/aurora_O3_finetune_US-WEST_3day_lead_config_v4.yaml"
INPUT_NB="aurora_finetune_rollout.ipynb"
OUTPUT_NB="outputs/O3_US-WEST_3day_lead_v4/aurora_finetune_rollout_executed_$(date +%Y%m%d_%H%M%S).ipynb"
LOG_FILE="outputs/O3_US-WEST_3day_lead_v4/training_$(date +%Y%m%d_%H%M%S).log"

# Create output directory
mkdir -p outputs/O3_US-WEST_3day_lead_v4

echo "=========================================="
echo "Aurora O3 v4 Fine-Tuning (papermill)"
echo "Config:    $CONFIG_NAME"
echo "Input NB:  $INPUT_NB"
echo "Output NB: $OUTPUT_NB"
echo "Log:       $LOG_FILE"
echo "Date:      $(date)"
echo "=========================================="
echo

# Initialize mamba for this shell session
eval "$(/home/azureuser/miniforge3/bin/mamba shell hook --shell bash)"

# Activate aurora environment
echo "Activating mamba aurora environment..."
mamba activate aurora

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo

# Environment variables
export TQDM_DISABLE=0
export AURORA_DISABLE_PROGRESS_BARS=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

# Run notebook with papermill (with parameters to override config)
echo "Starting papermill execution..."
echo

papermill \
  "$INPUT_NB" \
  "$OUTPUT_NB" \
  -p CONFIG_PATH_NAME "$CONFIG_NAME" \
  --log-output \
  --progress-bar 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully: $(date)"
    echo "Executed notebook: $OUTPUT_NB"
    echo "Checkpoints: outputs/checkpoints/O3_US-WEST_3day_lead_v4/"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi
echo "=========================================="

exit $EXIT_CODE
