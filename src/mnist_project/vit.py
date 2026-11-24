"""Trackable positional embeddings, including .keras save/load support."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='mnist_project')
class PositionEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.positions = self.add_weight(name='positions', shape=(input_shape[1],input_shape[2]),
                                         initializer='random_normal', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.positions
