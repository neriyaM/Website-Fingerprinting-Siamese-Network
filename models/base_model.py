from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


def build_base_model(input_shape, batchnorm, dropout, embedding_size):
    bias_initializer = RandomNormal(mean=0.5, stddev=0.01)
    kernel_initializer = RandomNormal(mean=0.0, stddev=0.01)
    kernel_regularizer = l2(2e-4)

    model = Sequential()
    model.add(Dense(12, input_shape=input_shape, activation='relu',
                    bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
    if batchnorm:
        model.add(BatchNormalization())

    model.add(Dense(8, activation='relu', bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer))
    if batchnorm:
        model.add(BatchNormalization())

    if dropout:
        model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(embedding_size))

    return model
