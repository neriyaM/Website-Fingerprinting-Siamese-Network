from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
from consts import *
import time
import numpy as np
import os
import cv2


class SiameseNetwork:
    def __init__(self, base_model, input_shape, loss, optimizer, metrics):
        first_input = Input(input_shape)
        second_input = Input(input_shape)
        first_model = base_model(first_input)
        second_model = base_model(second_input)

        # L1 distance layer
        distance_lambda = Lambda(lambda x: abs(x[0] - x[1]))
        distance_layer = distance_lambda([first_model, second_model])

        output_layer = Dense(1, activation='sigmoid')(distance_layer)
        network = Model(inputs=[first_input, second_input], outputs=output_layer)
        network.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.network = network

    def train(self, X_train, Y_train, X_test, Y_test, validation_split, epochs,
              verbose, callbacks, batch_size, shuffle):
        start_time = time.time()
        self.network.fit(X_train, Y_train, validation_split=validation_split,
                         epochs=epochs, verbose=verbose,
                         callbacks=callbacks, batch_size=batch_size, shuffle=shuffle)
        duration = time.time() - start_time

        predicts = self.network.predict(X_test)
        predicts = [int(np.round(x)) for x in predicts]
        test_accuracy = accuracy_score(Y_test, predicts)
        return duration, test_accuracy
