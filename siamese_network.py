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
        record_samples(predicts, X_test, Y_test, SAMPLES_DIR)
        return duration, test_accuracy


def record_samples(predicts, X_test, Y_test, output_dir):
    correct_dir_path = os.path.join(output_dir, SAMPLES_CORRECT_DIR)
    incorrect_dir_path = os.path.join(output_dir, SAMPLES_INCORRECT_DIR)
    if not os.path.exists(correct_dir_path):
        os.makedirs(correct_dir_path)
    if not os.path.exists(incorrect_dir_path):
        os.makedirs(incorrect_dir_path)

    correct_num, incorrect_num = 0, 0
    i = 0
    while correct_num < 10 or incorrect_num < 10:
        pair = (X_test[0][i], X_test[1][i])
        if Y_test[i] == predicts[i]:
            if correct_num < 10:
                correct_num = correct_num + 1
                save_pair(pair, correct_num, correct_dir_path)
        else:
            if incorrect_num < 10:
                incorrect_num = incorrect_num + 1
                save_pair(pair, incorrect_num, incorrect_dir_path)

        i = i + 1


def save_pair(pair, index, output_dir):
    imageA_path = os.path.join(output_dir, '{}A.png'.format(index))
    imageB_path = os.path.join(output_dir, '{}B.png'.format(index))
    cv2.imwrite(imageA_path, pair[0].squeeze())
    cv2.imwrite(imageB_path, pair[1].squeeze())
