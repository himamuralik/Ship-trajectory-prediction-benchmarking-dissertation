# hypermodel.py

import kerastuner as kt
import tensorflow as tf
from keras import Input, Model
from keras.layers import (
    GRU as GRUKeras, LSTM as LSTMKeras,
    Bidirectional, Flatten, Dense, Dropout
)
from keras.optimizers import Adam

from models.losses import HaversineLoss
from loading import Normalizer

class RNNFusionHyperModel(kt.HyperModel):
    def __init__(self, 
                 input_ts_length, 
                 input_num_features, 
                 static_feature_dim,
                 output_dim,
                 normalization_factors):
        """
        :param input_ts_length: length of your time series
        :param input_num_features: # features per timestep
        :param static_feature_dim: # static inputs (0 if none)
        :param output_dim:  usually 2 (lat/lon)
        :param normalization_factors: dict, passed to HaversineLoss & unnormalizer
        """
        self.input_ts_length = input_ts_length
        self.input_num_features = input_num_features
        self.static_feature_dim = static_feature_dim
        self.output_dim = output_dim
        self.normalization_factors = normalization_factors

    def build(self, hp):
        # 0) pick architecture
        model_type = hp.Choice("model_type",
                              ["bilstm", "bilstm_attention", "long_term_fusion"],
                              default="bilstm")

        # 1) shared RNN choices
        cell_str = hp.Choice("cell", ["LSTM","GRU"], default="LSTM")
        RNN = LSTMKeras if cell_str=="LSTM" else GRUKeras

        direction = hp.Choice("direction", ["forward_only","bidirectional"],
                              default="bidirectional")
        rnn_layers = hp.Int("rnn_layers", 1, 5, step=1, default=2)
        rnn_units  = hp.Int("rnn_units", 32, 256, step=32, default=128)

        # 2) fusion‐only params
        fusion_layers = fusion_units = feat_layers = feat_units = 0
        if model_type=="long_term_fusion":
            fusion_layers = hp.Int("fusion_layers", 0, 3, step=1, default=1)
            fusion_units  = hp.Int("fusion_units", 32, 256, step=32, default=128)
            feat_layers   = hp.Int("feat_layers", 0, 2, step=1, default=1)
            feat_units    = hp.Int("feat_units", 0, 256, step=32, default=64)

        # 3) head depth
        dense_layers = hp.Int("dense_layers", 1, 3, step=1, default=1)
        dense_units  = hp.Int("dense_units", 32, 256, step=32, default=128)

        # 4) optimizer
        lr = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log", default=1e-3)

        # --- build graph ---
        # time‐series trunk
        x_in = Input(shape=(self.input_ts_length, self.input_num_features),
                     name="recurrent_input")
        x = x_in
        for i in range(rnn_layers):
            return_seq = (i < rnn_layers-1) or (model_type=="bilstm_attention")
            layer = RNN(rnn_units, return_sequences=return_seq)
            x = (Bidirectional(layer)(x)
                 if direction=="bidirectional" else layer(x))

        # apply attention vs flatten vs mean-pool
        if model_type=="bilstm_attention":
            att = Dense(rnn_units, activation="tanh")(x)
            x = tf.reduce_mean(att, axis=1)
        elif model_type=="bilstm":
            x = x[:, -1, :]
        else:  # long_term_fusion: mean‐pool
            x = tf.reduce_mean(x, axis=1)

        inputs = [x_in]

        # static branch for fusion
        if model_type=="long_term_fusion":
            s_in = Input(shape=(self.static_feature_dim,), name="static_input")
            s = s_in
            # optional feature extractor
            for _ in range(feat_layers):
                s = Dense(feat_units, activation="relu")(s)
                s = Dropout(0.2)(s)
            # fusion layers
            for _ in range(fusion_layers):
                s = Dense(fusion_units, activation="relu")(s)
                s = Dropout(0.2)(s)
            inputs.append(s_in)
            x = tf.concat([x, s], axis=1)

        # final dense head
        for _ in range(dense_layers):
            x = Dense(dense_units, activation="relu")(x)
            x = Dropout(0.2)(x)

        out = Dense(self.output_dim, activation="linear", name="pred_lat_lon")(x)
        model = Model(inputs=inputs, outputs=out)

        # compile
        loss_fn = "mse" if model_type!="long_term_fusion" else HaversineLoss(self.normalization_factors).haversine_loss
        model.compile(optimizer=Adam(learning_rate=lr), loss=loss_fn)
        return model

