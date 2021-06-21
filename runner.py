from consts import *
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
import csv
from base_model import build_base_model
from siamese_network import SiameseNetwork
from tensorflow.keras.backend import clear_session


class Runner:
    def __init__(self, name, batch_size, validation_size, epochs, batchnorm,
                 dropout, optimizer, optimizer_name, learning_rate, callbacks):
        self.name = name
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.epochs = epochs
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.callbacks = callbacks

        base_model = build_base_model(INPUT_SHAPE_GRAYSCALE, batchnorm, dropout)
        self.siamese_network = SiameseNetwork(base_model, INPUT_SHAPE_GRAYSCALE,
                                              'binary_crossentropy', optimizer, 'binary_accuracy')

    def run(self, X_train, Y_train, X_test, Y_test):
        duration, test_accuracy = self.siamese_network.train(X_train, Y_train, X_test, Y_test,
                                                             validation_split=self.validation_size, epochs=self.epochs,
                                                             verbose=True,
                                                             callbacks=self.callbacks, batch_size=self.batch_size,
                                                             shuffle=True)

        self.save_results(duration, test_accuracy)

    def save_results(self, duration, test_accuracy):
        result = {
            BATCH_SIZE_KEY: str(self.batch_size),
            VALIDATION_SIZE_KEY: str(self.validation_size),
            EPOCHS_KEY: str(self.epochs),
            BATCHNORM_KEY: str(self.batchnorm),
            DROPOUT_KEY: str(self.dropout),
            OPTIMIZER_KEY: self.optimizer_name,
            LEARNING_RATE_KEY: str(self.learning_rate),
            TEST_ACCURACY_KEY: str(test_accuracy),
            DURATION_KEY: str(duration)
        }

        with open(RESULTS_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
            writer.writerow(result)

    @staticmethod
    def clear():
        clear_session()


def load_runners(runners_config):
    runners = []
    for runner_config in runners_config:
        runner_name = runner_config[RUNNER_NAME_KEY]
        batch_size = runner_config[BATCH_SIZE_KEY]
        validation_size = runner_config[VALIDATION_SIZE_KEY]
        epochs = runner_config[EPOCHS_KEY]
        batchnorm = runner_config[BATCHNORM_KEY]
        dropout = runner_config[DROPOUT_KEY]
        learning_rate = runner_config[LEARNING_RATE_KEY]
        optimizer_name = runner_config[OPTIMIZER_KEY]
        optimizer = get_optimizer(optimizer_name, learning_rate)
        callbacks = get_callbacks(runner_name, runner_config[CALLBACKS_KEY])
        runners.append(Runner(runner_name, batch_size, validation_size, epochs,
                              batchnorm, dropout, optimizer, optimizer_name, learning_rate, callbacks))

    return runners


def get_optimizer(name, learning_rate):
    if name == "Adam":
        return Adam(lr=learning_rate)
    elif name == "Adagrad":
        return Adagrad(lr=learning_rate)
    elif name == "RMSprop":
        return RMSprop(lr=learning_rate)

    raise OptimizerNotFound


def get_callbacks(runner_name, callbacks_name):
    callbacks = []
    for name in callbacks_name:
        if name == "EarlyStopping":
            callbacks.append(EarlyStopping(patience=3))
        elif name == "TensorBoard":
            log_dir = os.path.join(TENSORBOARD_LOGS_DIR, runner_name)
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1,
                                         write_graph=True, write_images=True, update_freq='batch'))

    return callbacks


class OptimizerNotFound(Exception):
    pass
