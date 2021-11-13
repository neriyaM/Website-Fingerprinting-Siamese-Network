from consts import *
import csv
from siamese_network import SiameseNetwork
from tensorflow.keras.backend import clear_session


class Runner:
    def __init__(self, name, base_model, dataset, input_size, batch_size, validation_size, epochs,
                 optimizer, distance_lambda, callbacks):
        self.name = name
        self.base_model = base_model
        self.dataset = dataset
        self.input_size = input_size
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.distance_lambda = distance_lambda
        self.optimizer = optimizer
        self.siamese_network = SiameseNetwork(base_model.model, input_size, distance_lambda.l,
                                              'binary_crossentropy', optimizer.optimizer, 'binary_accuracy')

    def run(self):
        duration, test_accuracy = self.siamese_network.train(self.dataset.X_train, self.dataset.Y_train,
                                                             self.dataset.X_test, self.dataset.Y_test,
                                                             validation_split=self.validation_size,
                                                             epochs=self.epochs,
                                                             verbose=True,
                                                             callbacks=self.callbacks, batch_size=self.batch_size,
                                                             shuffle=True)

        self.save_results(duration, test_accuracy)

    def save_results(self, duration, test_accuracy):
        result = {
            BASE_MODELS_KEY: self.base_model.name,
            DATASETS_KEY: self.dataset.name,
            DISTANCE_LAMBDAS_KEY: self.distance_lambda.name,
            BATCH_SIZES_KEY: str(self.batch_size),
            EMBEDDING_SIZES_KEY: str(self.base_model.embedding_size),
            VALIDATION_SIZES_KEY: str(self.validation_size),
            EPOCHS_KEY: str(self.epochs),
            BATCHNORM_KEY: str(self.base_model.batchnorm),
            DROPOUT_KEY: str(self.base_model.dropout),
            OPTIMIZERS_KEY: self.optimizer.name,
            LEARNING_RATES_KEY: str(self.optimizer.learning_rate),
            TEST_ACCURACY_KEY: str(test_accuracy),
            DURATION_KEY: str(duration)
        }

        with open(RESULTS_FILE, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
            writer.writerow(result)

    @staticmethod
    def clear():
        clear_session()
