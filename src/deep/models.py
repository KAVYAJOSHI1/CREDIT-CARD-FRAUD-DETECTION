"""Deep architectures for tabular fraud detection (Keras 3)."""
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="fraud")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.75):
    """Binary focal loss (Lin et al. 2017): down-weights easy negatives so the
    0.17% fraud class dominates the gradient."""
    y_true = ops.cast(y_true, y_pred.dtype)
    y_pred = ops.clip(y_pred, 1e-7, 1 - 1e-7)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    a = y_true * alpha + (1 - y_true) * (1 - alpha)
    return ops.mean(-a * ops.power(1 - pt, gamma) * ops.log(pt))


# ---------------------------------------------------------------- Residual MLP
def residual_mlp(n_features, width=128, blocks=4, drop=0.2):
    """Pre-activation residual MLP with BatchNorm, GELU and dropout (MC-dropout capable)."""
    inp = layers.Input((n_features,))
    x = layers.Dense(width)(inp)
    for _ in range(blocks):
        h = layers.BatchNormalization()(x)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(drop)(h)
        h = layers.Dense(width)(h)
        h = layers.BatchNormalization()(h)
        h = layers.Activation("gelu")(h)
        h = layers.Dense(width)(h)
        x = layers.Add()([x, h])
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(drop)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="resmlp")


# ------------------------------------------------- FT-Transformer (Gorishniy 2021)
@keras.saving.register_keras_serializable(package="fraud")
class FeatureTokenizer(layers.Layer):
    """Embeds every scalar feature into its own d-dim token (x_i * W_i + b_i) and
    prepends a learnable [CLS] token."""

    def __init__(self, n_features, d_model, **kw):
        super().__init__(**kw)
        self.n_features, self.d_model = n_features, d_model

    def build(self, shape):
        self.w = self.add_weight(shape=(self.n_features, self.d_model), initializer="glorot_uniform", name="w")
        self.b = self.add_weight(shape=(self.n_features, self.d_model), initializer="zeros", name="b")
        self.cls = self.add_weight(shape=(1, 1, self.d_model), initializer="glorot_uniform", name="cls")

    def call(self, x):
        tok = ops.expand_dims(x, -1) * self.w + self.b
        cls = ops.tile(self.cls, [ops.shape(x)[0], 1, 1])
        return ops.concatenate([cls, tok], axis=1)

    def get_config(self):
        return {**super().get_config(), "n_features": self.n_features, "d_model": self.d_model}


def tabular_transformer(n_features, d_model=32, heads=4, blocks=3, drop=0.1):
    inp = layers.Input((n_features,))
    x = FeatureTokenizer(n_features, d_model)(inp)
    for _ in range(blocks):
        h = layers.LayerNormalization()(x)
        h = layers.MultiHeadAttention(heads, d_model // heads, dropout=drop)(h, h)
        x = layers.Add()([x, h])
        h = layers.LayerNormalization()(x)
        h = layers.Dense(d_model * 2, activation="gelu")(h)
        h = layers.Dropout(drop)(h)
        h = layers.Dense(d_model)(h)
        x = layers.Add()([x, h])
    cls = layers.LayerNormalization()(x[:, 0])
    cls = layers.Activation("relu")(cls)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(cls), name="ft_transformer")


# ------------------------------------------------------- Denoising autoencoder
def denoising_autoencoder(n_features, bottleneck=8, noise=0.1):
    """Trained on legitimate transactions only. Fraud reconstructs poorly -> high error."""
    inp = layers.Input((n_features,))
    x = layers.GaussianNoise(noise)(inp)
    for u in (64, 32):
        x = layers.Dense(u, activation="gelu")(x)
    z = layers.Dense(bottleneck, name="bottleneck")(x)
    x = z
    for u in (32, 64):
        x = layers.Dense(u, activation="gelu")(x)
    return keras.Model(inp, layers.Dense(n_features)(x), name="dae")
