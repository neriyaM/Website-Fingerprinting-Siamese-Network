from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2


def build_base_model(input_shape, batchnorm, dropout):
    bias_initializer = RandomNormal(mean=0.5, stddev=0.01)
    kernel_initializer = RandomNormal(mean=0.0, stddev=0.01)
    kernel_regularizer = l2(2e-4)

    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
    if batchnorm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
    if batchnorm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
    if batchnorm:
        model.add(BatchNormalization())

    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', bias_initializer=bias_initializer,
                     kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
    if batchnorm:
        model.add(BatchNormalization())

    if dropout:
        model.add(Dropout(rate=0.2))

    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid'))

    return model
