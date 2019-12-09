import tensorflow as tf
from tensorflow import keras


def balanced_binary_focal_loss(alpha: float = 0.75, gamma: float = 2.0):

    assert 0.0 <= alpha <= 1.0

    def loss_fn(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(keras.backend.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_t = tf.where(keras.backend.equal(y_true, 1.0), alpha_factor, 1.0 - alpha_factor)
        cross_entropy = -keras.backend.log(p_t)
        weight = alpha_t * keras.backend.pow((1.0 - p_t), gamma)
        loss = weight * cross_entropy
        loss = keras.backend.sum(loss, axis=-1)
        return loss

    loss_fn.__name__ = 'balanced_binary_focal_loss'

    return loss_fn
