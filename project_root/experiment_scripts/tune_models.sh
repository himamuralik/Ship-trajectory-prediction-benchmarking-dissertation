#!/bin/bash

# Hyperparameter Tuning Section
echo "Starting hyperparameter tuning..."

# 1. Basic BiLSTM Tuning
echo "Tuning Basic BiLSTM model..."
mlflow run . -e tune_model \
  -P model_type=bi_lstm \
  -P dataset_name=new_york \
  --experiment-name 'Hyperparameter Tuning'

# 2. BiLSTM with Attention Tuning  
echo "Tuning BiLSTM with Attention..."
mlflow run . -e tune_model \
  -P model_type=bi_lstm_attention \
  -P dataset_name=new_york \
  --experiment-name 'Hyperparameter Tuning'

# 3. Fusion Model Tuning (for 3-hour prediction)
echo "Tuning Fusion Model (3-hour prediction)..."
mlflow run . -e tune_model \
  -P model_type=long_term_fusion \
  -P dataset_name=new_york \
  --experiment-name 'Hyperparameter Tuning' \
  -P hours_out=3

# Wait for all tuning to complete
echo "All tuning jobs submitted. Waiting for completion..."
wait

# Extract best parameters (example - adjust paths as needed)
echo "Extracting best parameters..."
TUNING_DIR="mlruns/0"  # Default MLflow tracking URI

# Get the most recent tuning run IDs
BI_LSTM_RUN=$(ls -t $TUNING_DIR | grep "bi_lstm" | head -1)
ATTN_RUN=$(ls -t $TUNING_DIR | grep "bi_lstm_attention" | head -1)  
FUSION_RUN=$(ls -t $TUNING_DIR | grep "long_term_fusion" | head -1)

# Copy parameter files to organized location
mkdir -p tuning_results
cp $TUNING_DIR/$BI_LSTM_RUN/artifacts/tuning_results/*.json tuning_results/
cp $TUNING_DIR/$ATTN_RUN/artifacts/tuning_results/*.json tuning_results/
cp $TUNING_DIR/$FUSION_RUN/artifacts/tuning_results/*.json tuning_results/

echo "Tuning complete. Best parameters saved to tuning_results/"
