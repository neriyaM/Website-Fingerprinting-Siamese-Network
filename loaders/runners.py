import os
from consts import *
import itertools
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import utils
from loaders.data import load_data
from runner import Runner
from loaders.models import load_base_models
import numpy as np


def load_runners(runners_config):
    runners = []
    base_models = runners_config[BASE_MODELS_KEY]
    datasets = runners_config[DATASETS_KEY]
    embedding_sizes = runners_config[EMBEDDING_SIZES_KEY]
    batch_sizes = runners_config[BATCH_SIZES_KEY]
    distance_lambdas = runners_config[DISTANCE_LAMBDAS_KEY]
    optimizers = runners_config[OPTIMIZERS_KEY]
    validation_sizes = runners_config[VALIDATION_SIZES_KEY]
    epochs_values = runners_config[EPOCHS_KEY]

    for values in itertools.product(base_models, datasets, embedding_sizes,
                                    batch_sizes, distance_lambdas,
                                    optimizers, validation_sizes, epochs_values):
        base_model_name = values[0]
        dataset_name = values[1]
        embedding_size = values[2]
        batch_size = values[3]
        distance_lambda_name = values[4]
        optimizers_config = values[5]
        validation_size = values[6]
        epochs = values[7]

        dataset = load_dataset(dataset_name)
        input_size = INPUT_SIZES[dataset_name]
        base_models = load_base_models(base_model_name, input_size, embedding_size)
        optimizers = get_optimizers(optimizers_config)
        distance_lambda = get_distance_lambda(distance_lambda_name)
        for vals in itertools.product(base_models, optimizers):
            base_model = vals[0]
            optimizer = vals[1]
            runner_name = build_runner_name(base_model, dataset_name, batch_size,
                                            distance_lambda_name, optimizer, validation_size, epochs)
            callbacks = get_callbacks(runner_name, runners_config[CALLBACKS_KEY])
            runner = Runner(runner_name, base_model, dataset, input_size, batch_size, validation_size,
                            epochs, optimizer, distance_lambda, callbacks)
            runners.append(runner)

    return runners


def load_dataset(dataset_name):
    match_pairs_train = os.path.join(dataset_name, TRAIN_KEY, PAIRS_KEY, MATCH_FILE_KEY)
    mismatch_pairs_train = os.path.join(dataset_name, TRAIN_KEY, PAIRS_KEY, MISMATCH_FILE_KEY)
    features_train = os.path.join(dataset_name, TRAIN_KEY, FEATURES_KEY)
    X_train, Y_train = load_data(match_pairs_train, mismatch_pairs_train, features_train)

    match_pairs_test = os.path.join(dataset_name, TEST_KEY, PAIRS_KEY, MATCH_FILE_KEY)
    mismatch_pairs_test = os.path.join(dataset_name, TEST_KEY, PAIRS_KEY, MISMATCH_FILE_KEY)
    features_test = os.path.join(dataset_name, TEST_KEY, FEATURES_KEY)
    X_test, Y_test = load_data(match_pairs_test, mismatch_pairs_test, features_test)

    return utils.Dataset(dataset_name, X_train, Y_train, X_test, Y_test)


def get_optimizers(optimizers_config):
    optimizers = []
    optimizer_name = optimizers_config[NAME_KEY]
    for learning_rate in optimizers_config[LEARNING_RATES_KEY]:
        optimizer = get_optimizer(optimizer_name, learning_rate)
        optimizers.append(optimizer)

    return optimizers


def get_optimizer(name, learning_rate):
    if name == "Adam":
        optimizer = Adam(lr=learning_rate)
    elif name == "Adagrad":
        optimizer = Adagrad(lr=learning_rate)
    elif name == "RMSprop":
        optimizer = RMSprop(lr=learning_rate)
    elif name == "SGD":
        optimizer = SGD(lr=learning_rate)
    else:
        raise utils.OptimizerNotFound

    return utils.Optimizer(optimizer, name, learning_rate)


def get_distance_lambda(name):
    if name == "L1":
        return utils.DistanceLambda(name, lambda x: abs(x[0] - x[1]))
    elif name == "L2":
        return utils.DistanceLambda(name, utils.euclidean_distance)

    raise utils.DistanceLambdaNotFound


def build_runner_name(base_model, dataset_name, batch_size, distance_lambda_name,
                      optimizer, validation_size, epochs):
    return f"{base_model.name}_{dataset_name}_{batch_size}_{distance_lambda_name}_{optimizer.name}_" \
           f"{validation_size}_{epochs}_{base_model.batchnorm}_{base_model.dropout}_{base_model.embedding_size}"


def get_callbacks(runner_name, callbacks_name):
    callbacks = []
    for name in callbacks_name:
        if name == "EarlyStopping":
            callbacks.append(EarlyStopping(patience=3))
        elif name == "TensorBoard":
            log_dir = os.path.join(TENSORBOARD_LOGS_DIR, runner_name)
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1,
                                         write_graph=True, write_images=True, update_freq='batch'))
        elif name == "ModelCheckpoint":
            callbacks.append(
                ModelCheckpoint(filepath="model.h5", monitor='val_binary_accuracy', mode='max', save_best_only=True,
                                verbose=1))
        elif name == "ReduceLR":
            callbacks.append(ReduceLROnPlateau(factor=np.sqrt(0.1), patience=5))

    return callbacks
