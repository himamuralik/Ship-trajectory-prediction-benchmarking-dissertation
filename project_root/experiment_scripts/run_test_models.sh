#!/usr/bin/env bash
set -euo pipefail

#---------------------------------------------------
#  Load best hyperparameters from JSON into shell vars
#---------------------------------------------------
eval $(jq -r '
  .long_term_fusion
  | to_entries[]
  | "LTF_\(.key)=\(.value)"
' best_params.json)

eval $(jq -r '
  .bilstm
  | to_entries[]
  | "BIL_\(.key)=\(.value)"
' best_params.json)

eval $(jq -r '
  .bilstm_attention
  | to_entries[]
  | "BAT_\(.key)=\(.value)"
' best_params.json)


#---------------------------------------------------
# 1) Long‑Term Fusion evaluation
mlflow run . \
  -e test_time_gaps \
  -P time_gap=30 \
  -P length_of_history=3 \
  -P hours_out=3 \
  -P loss=mse \
  -P time_of_day=hour_day \
  -P weather=currents \
  -P dataset_name=new_york \
  --experiment-name "Test Models" \
  -P model_type=long_term_fusion \
  -P sog_cog=raw \
  -P rnn_to_dense_connection=all_nodes \
  -P batch_size=$LTF_batch_size \
  -P direction=bidirectional \
  -P layer_type=lstm \
  -P learning_rate=$LTF_learning_rate \
  -P number_of_dense_layers=$LTF_number_of_dense_layers \
  -P number_of_rnn_layers=$LTF_number_of_rnn_layers \
  -P rnn_layer_size=$LTF_rnn_layer_size \
  -P dense_layer_size=$LTF_dense_layer_size \
  -P extended_recurrent_idxs=vt_dst_and_time \
  -P number_of_fusion_weather_layers=$LTF_number_of_fusion_weather_layers

#---------------------------------------------------
# 2) BiLSTM evaluation
mlflow run . \
  -e test_time_gaps \
  -P time_gap=30 \
  -P length_of_history=3 \
  -P loss=mse \
  -P time_of_day=hour_day \
  -P weather=ignore \
  -P dataset_name=new_york \
  --experiment-name "Test Models" \
  -P model_type=bilstm \
  -P sog_cog=raw \
  -P rnn_to_dense_connection=all_nodes \
  -P batch_size=$BIL_batch_size \
  -P direction=bidirectional \
  -P layer_type=lstm \
  -P learning_rate=$BIL_learning_rate \
  -P number_of_dense_layers=$BIL_number_of_dense_layers \
  -P number_of_rnn_layers=$BIL_number_of_rnn_layers \
  -P rnn_layer_size=$BIL_rnn_layer_size \
  -P dense_layer_size=$BIL_dense_layer_size

#---------------------------------------------------
# 3) BiLSTM + Attention evaluation
mlflow run . \
  -e test_time_gaps \
  -P time_gap=30 \
  -P length_of_history=3 \
  -P loss=haversine \
  -P time_of_day=hour_day \
  -P weather=ignore \
  -P dataset_name=new_york \
  --experiment-name "Test Models" \
  -P model_type=bilstm_attention \
  -P sog_cog=raw \
  -P batch_size=$BAT_batch_size \
  -P direction=forward_only \
  -P layer_type=lstm \
  -P learning_rate=$BAT_learning_rate \
  -P number_of_rnn_layers=$BAT_number_of_rnn_layers \
  -P rnn_layer_size=$BAT_rnn_layer_size
