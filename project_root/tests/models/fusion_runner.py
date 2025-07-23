import tensorflow as tf
from haversine import haversine_vector, Unit
from keras import Input, Model
from keras.layers import Bidirectional, LSTM, Flatten, Dense, Dropout
from keras.optimizer_v2.adam import Adam as AdamKeras
from keras.regularizers import L1, L2

from loading import Normalizer
from models.losses import HaversineLoss
from models.model_runner import ModelRunner


class FusionModelRunner(ModelRunner):
    """
    Bi‑LSTM trunk + static one‑hot “vessel_group” branch.
    """
    def __init__(
        self,
        n_rnn_layers: int,
        rnn_units: int,
        n_static_layers: int,
        static_units: int,
        n_dense_layers: int,
        dense_units: int,
        direction: str,
        ts_length: int,
        n_recurrent_feats: int,
        n_static_feats: int,
        output_dim: int,
        normalization_factors: dict,
        y_idxs: list,
        columns,
        learning_rate: float,
        rnn_to_dense: str = "last",
        regularization: str = None,
        regularization_app: str = None,
        regularization_coeff: float = None,
    ):
        # Direction
        if direction not in ("forward_only", "bidirectional"):
            raise ValueError("direction must be 'forward_only' or 'bidirectional'")
        self.direction = direction

        # Hyperparams
        self.n_rnn_layers     = n_rnn_layers
        self.rnn_units        = rnn_units
        self.n_static_layers  = n_static_layers
        self.static_units     = static_units
        self.n_dense_layers   = n_dense_layers
        self.dense_units      = dense_units
        self.ts_length        = ts_length
        self.n_recurrent_feats= n_recurrent_feats
        self.n_static_feats   = n_static_feats
        self.output_dim       = output_dim
        self.rnn_to_dense     = rnn_to_dense

        # Regularization
        self._config_regularization(regularization,
                                    regularization_app,
                                    regularization_coeff)

        # Loss & optimizer
        self.normalization_factors = normalization_factors
        self.y_idxs       = y_idxs
        self.columns      = columns
        self.optimizer    = AdamKeras(learning_rate=learning_rate)
        self.loss         = HaversineLoss(normalization_factors).haversine_loss

        # Build, compile
        self._build_model()
        self.compile()

    def _config_regularization(self, method, app, coeff):
        if method == "dropout":
            self.rnn_reg     = {"recurrent_dropout": coeff} if app=="recurrent" else {"dropout": coeff}
            self.dense_reg   = {}
            self.dense_dropout = 0.0 if app=="recurrent" else coeff
        elif method in ("l1","l2"):
            cls = L1 if method=="l1" else L2
            key = f"{app}_regularizer" if app else "kernel_regularizer"
            self.rnn_reg     = {key: cls(coeff)}
            self.dense_reg   = {key: cls(coeff)}
            self.dense_dropout = 0.0
        else:
            self.rnn_reg     = {}
            self.dense_reg   = {}
            self.dense_dropout = 0.0

    def _build_model(self):
        # --- recurrent trunk ---
        seq_in = Input(shape=(self.ts_length, self.n_recurrent_feats), name="seq_input")
        x = seq_in
        for i in range(self.n_rnn_layers):
            return_seq = (self.rnn_to_dense=="all") or (i < self.n_rnn_layers-1)
            lstm = LSTM(self.rnn_units, return_sequences=return_seq, **self.rnn_reg)
            x = Bidirectional(lstm)(x) if self.direction=="bidirectional" else lstm(x)

        if self.rnn_to_dense=="all":
            x = Flatten(name="flatten_seq")(x)
        else:
            x = x[:,-1,:]  # last timestep

        # --- static “vessel_group” branch ---
        static_in = Input(shape=(self.n_static_feats,), name="static_input")
        s = static_in
        for _ in range(self.n_static_layers):
            s = Dense(self.static_units, activation="relu", **self.dense_reg)(s)
            s = Dropout(self.dense_dropout)(s)

        # --- fusion ---
        x = tf.concat([x, s], axis=-1)

        # --- final dense head ---
        for _ in range(self.n_dense_layers):
            x = Dense(self.dense_units, activation="relu", **self.dense_reg)(x)
            x = Dropout(self.dense_dropout)(x)

        out = Dense(self.output_dim, activation="linear", name="pred_lat_lon")(x)
        self.model = Model(inputs=[seq_in, static_in], outputs=out, name="fusion_runner")

    def predict(self, X, Y, args=None):
        # X = [sequence_array, static_array], Y = ground truth
        Y_hat = self.model.predict(X)
        Y_hat = Normalizer().unnormalize(Y_hat, self.normalization_factors)
        Y_true= Normalizer().unnormalize(Y, self.normalization_factors)
        d = haversine_vector(Y_true, Y_hat, Unit.KILOMETERS)
        return [Y_hat], [d], [d.mean()]
