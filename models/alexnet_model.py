# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
                  weights=None):
    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv1D(96, 11, input_shape=img_shape,
                       padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 2
    alexnet.add(Conv1D(256, 5, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 3
    alexnet.add(ZeroPadding1D(1))
    alexnet.add(Conv1D(512, 3, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 4
    alexnet.add(ZeroPadding1D(1))
    alexnet.add(Conv1D(1024, 3, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))

    # Layer 5
    alexnet.add(ZeroPadding1D(1))
    alexnet.add(Conv1D(1024, 3, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling1D(pool_size=2))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(3072))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(128))

    if weights is not None:
        alexnet.load_weights(weights)

    return alexnet
