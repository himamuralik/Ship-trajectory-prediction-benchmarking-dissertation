#!/bin/bash

# Configuration
DATASET="new_york"
EXPERIMENT_NAME="Full Pipeline"
HOURS_OUT=3  # For fusion model
MLFLOW_TRACKING_URI="file:./mlruns"  # Local MLflow tracking

# Initialize directories
mkdir -p tuning_results
export MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Hyperparameter Tuning Section
echo "=== Starting Hyperparameter Tuning ==="

# 1. Basic BiLSTM Tuning
echo "[1/3] Tuning Basic BiLSTM model..."
mlflow run . -e tune_model \
  -P model_type=bi_lstm \
  -P dataset_name=$DATASET \
  --experiment-name "$EXPERIMENT_NAME"

# 2. BiLSTM with Attention Tuning  
echo "[2/3] Tuning BiLSTM with Attention..."
mlflow run . -e tune_model \
  -P model_type=bi_lstm_attention \
  -P dataset_name=$DATASET \
  --experiment-name "$EXPERIMENT_NAME"

# 3. Fusion Model Tuning
echo "[3/3] Tuning Fusion Model ($HOURS_OUT-hour prediction)..."
mlflow run . -e tune_model \
  -P model_type=long_term_fusion \
  -P dataset_name=$DATASET \
  --experiment-name "$EXPERIMENT_NAME" \
  -P hours_out=$HOURS_OUT

# Wait for all tuning to complete
echo "Waiting for tuning jobs to complete..."
wait

# Parameter Extraction Functions
get_mlflow_run_id() {
    local model_type=$1
    mlflow runs search --experiment-names "$EXPERIMENT_NAME" \
      --filter "params.model_type = '$model_type'" \
      --order-by "metrics.val_loss ASC" \
      --max-results 1 | grep -oP "(?<=run_id: )[a-f0-9]+"
}

extract_best_params() {
    local model_type=$1
    local run_id=$(get_mlflow_run_id "$model_type")
    
    if [ -z "$run_id" ]; then
        echo "ERROR: No run found for model type $model_type" >&2
        exit 1
    fi

    echo "Extracting parameters from run $run_id"
    mlflow artifacts download --run-id $run_id --artifact-path tuning_results -d tuning_results/$model_type
    
    local params_file=$(find tuning_results/${model_type} -name "*.json" | head -1)
    
    if [ ! -f "$params_file" ]; then
        echo "ERROR: No parameters file found for $model_type" >&2
        exit 1
    fi

    # Model-specific parameter mapping
    case $model_type in
        "bi_lstm")
            jq -c '{
                learning_rate: .[0].hyperparameters.learning_rate,
                lstm_num_layers: .[0].hyperparameters.lstm_num_layers,
                lstm_hidden_dim: .[0].hyperparameters.lstm_hidden_dim,
                num_dense_layers: .[0].hyperparameters.num_dense_layers,
                dense_layer_size: .[0].hyperparameters.dense_layer_size,
                dropout: .[0].hyperparameters.dropout,
                reg_coefficient: .[0].hyperparameters.reg_coefficient
            }' "$params_file"
            ;;
        "bi_lstm_attention")
            jq -c '{
                learning_rate: .[0].hyperparameters.learning_rate,
                lstm_num_layers: .[0].hyperparameters.lstm_num_layers,
                lstm_hidden_dim: .[0].hyperparameters.lstm_hidden_dim,
                num_dense_layers: .[0].hyperparameters.num_dense_layers,
                dense_layer_size: .[0].hyperparameters.dense_layer_size,
                dropout: .[0].hyperparameters.dropout,
                reg_coefficient: .[0].hyperparameters.reg_coefficient,
                attention_weight: (.[0].hyperparameters.attention_weight // 0.5),
                use_attention_loss: (.[0].hyperparameters.use_attention_loss // false),
                use_mse_loss: (.[0].hyperparameters.use_mse_loss // false),
                mse_weight: (.[0].hyperparameters.mse_weight // 0.1)
            }' "$params_file"
            ;;
        "long_term_fusion")
            jq -c '{
                learning_rate: .[0].hyperparameters.learning_rate,
                lstm_num_layers: .[0].hyperparameters.lstm_num_layers,
                lstm_hidden_dim: .[0].hyperparameters.lstm_hidden_dim,
                num_dense_layers: .[0].hyperparameters.num_dense_layers,
                dense_layer_size: .[0].hyperparameters.dense_layer_size,
                dropout: .[0].hyperparameters.dropout,
                reg_coefficient: .[0].hyperparameters.reg_coefficient,
                number_of_fusion_weather_layers: 3,
                fusion_layer_structure: "standard"
            }' "$params_file"
            ;;
        *)
            jq -c '.[0].hyperparameters' "$params_file"
            ;;
    esac
}

# Extract best parameters for each model
echo "=== Extracting Best Parameters ==="
BI_LSTM_PARAMS=$(extract_best_params "bi_lstm") || exit 1
ATTN_PARAMS=$(extract_best_params "bi_lstm_attention") || exit 1
FUSION_PARAMS=$(extract_best_params "long_term_fusion") || exit 1

# Helper function to convert JSON params to CLI arguments
json_to_cli_args() {
    local json_params=$1
    echo $json_params | jq -r 'to_entries | map(
        if .value | type == "boolean" then
            "-P \(.key)=\(if .value then "True" else "False" end)"
        else
            "-P \(.key)=\(.value)"
        end
    ) | join(" ")'
}

# Data Creation Section
echo "=== Creating Optimized Datasets ==="

# 1. Basic BiLSTM Data
echo "[1/3] Creating Basic BiLSTM dataset..."
eval mlflow run . -e create_data \
  -P time_gap=30 \
  -P length_of_history=3 \
  -P loss=mse \
  -P time_of_day=hour_day \
  -P weather=ignore \
  -P dataset_name=$DATASET \
  --experiment-name "$EXPERIMENT_NAME" \
  -P model_type=bi_lstm \
  -P sog_cog=raw \
  -P batch_size=2048 \
  -P direction=bidirectional \
  -P distance_traveled=ignore \
  -P layer_type=lstm \
  $(json_to_cli_args "$BI_LSTM_PARAMS")

# 2. BiLSTM with Attention Data
echo "[2/3] Creating BiLSTM with Attention dataset..."
eval mlflow run . -e create_data \
  -P time_gap=30 \
  -P length_of_history=3 \
  -P loss=haversine \
  -P time_of_day=hour_day \
  -P weather=ignore \
  -P dataset_name=$DATASET \
  --experiment-name "$EXPERIMENT_NAME" \
  -P model_type=bi_lstm_attention \
  -P sog_cog=raw \
  -P batch_size=2048 \
  -P direction=bidirectional \
  -P distance_traveled=ignore \
  -P layer_type=lstm \
  $(json_to_cli_args "$ATTN_PARAMS")

# 3. Fusion Model Data
echo "[3/3] Creating Fusion Model dataset..."
eval mlflow run . -e create_data \
  -P time_gap=30 \
  -P length_of_history=3 \
  -P hours_out=$HOURS_OUT \
  -P loss=mse \
  -P time_of_day=hour_day \
  -P weather=ignore \
  -P dataset_name=$DATASET \
  --experiment-name "$EXPERIMENT_NAME" \
  -P model_type=long_term_fusion \
  -P sog_cog=raw \
  -P batch_size=2048 \
  -P direction=bidirectional \
  -P distance_traveled=ignore \
  -P layer_type=lstm \
  -P extended_recurrent_idxs=vt_dst_and_time \
  $(json_to_cli_args "$FUSION_PARAMS")

echo "=== Pipeline Completed Successfully ==="
echo "All datasets created with optimized parameters"
