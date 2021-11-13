import tensorflow as tf


def euclidean_distance(x):
    sum_square = tf.math.reduce_sum(tf.math.square(x[0] - x[1]), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


class Optimizer:
    def __init__(self, optimizer, name, learning_rate):
        self.optimizer = optimizer
        self.name = name
        self.learning_rate = learning_rate


class BaseModel:
    def __init__(self, model, name, batchnorm, dropout, embedding_size):
        self.model = model
        self.name = name
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.embedding_size = embedding_size


class Dataset:
    def __init__(self, name, X_train, Y_train, X_test, Y_test):
        self.name = name
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test


class DistanceLambda:
    def __init__(self, name, l):
        self.name = name
        self.l = l


class OptimizerNotFound(Exception):
    pass


class DistanceLambdaNotFound(Exception):
    pass
