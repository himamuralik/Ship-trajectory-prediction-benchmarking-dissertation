#!/bin/bash
# run_evaluation.sh - Executes model evaluations using pre-tuned parameters

set -e  # Exit immediately if any command fails

# --------------------------
# Configuration
# --------------------------
DATASET="new_york"
TIMEGAP=30
HISTORY_LEN=3
EXPERIMENT_NAME="Final Evaluation"
TRACKING_DIR="mlruns/0"  # Default MLflow tracking URI
MODELS=("bi_lstm" "bi_lstm_attention" "long_term_fusion")

# --------------------------
# Validation
# --------------------------
echo "Validating tuning results..."
for model in "${MODELS[@]}"; do
  if [ ! -f "tuning_results/${model}.json" ]; then
    echo "Error: Tuning results missing for ${model}"
    echo "Run tune_models.sh first"
    exit 1
  fi
done

# --------------------------
# Evaluation Function
# --------------------------
run_evaluation() {
  local model_type=$1
  local params_file=$2
  local extra_params=$3
  
  echo -e "\nEvaluating ${model_type}..."
  
  # Load tuned parameters
  local model_params=$(jq -r '.best_params | to_entries[] | "-P \(.key)=\(.value)"' "${params_file}")
  
  # Base command
  local base_cmd="mlflow run . -e evaluate_model \
    -P dataset_name=${DATASET} \
    -P time_gap=${TIMEGAP} \
    -P length_of_history=${HISTORY_LEN} \
    ${model_params} \
    --experiment-name '${EXPERIMENT_NAME}'"
  
  # Add model-specific params
  if [ "${model_type}" == "long_term_fusion" ]; then
    base_cmd+=" -P hours_out=3"
  fi
  
  # Add any extra params
  if [ -n "${extra_params}" ]; then
    base_cmd+=" ${extra_params}"
  fi
  
  # Execute
  echo "Running: ${base_cmd}"
  eval "${base_cmd}"
}

# --------------------------
# Main Execution
# --------------------------
echo -e "\nStarting model evaluations at $(date)"

# Run evaluations sequentially
run_evaluation "bi_lstm" "tuning_results/bi_lstm.json"
run_evaluation "bi_lstm_attention" "tuning_results/bi_lstm_attention.json"
run_evaluation "long_term_fusion" "tuning_results/long_term_fusion.json"

# --------------------------
# Completion
# --------------------------
echo -e "\nAll evaluations completed successfully at $(date)"
echo "Results available in MLflow under experiment '${EXPERIMENT_NAME}'"
