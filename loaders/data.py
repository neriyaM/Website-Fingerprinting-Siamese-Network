import numpy as np
import os
import csv


def load_data(match_path, mismatch_path, data_dir_path):
    features_vectors = load_feature_vectors(data_dir_path)
    X_match, Y_match = load_match_pairs(features_vectors, match_path)
    X_mismatch, Y_mismatch = load_mismatch_pairs(features_vectors, mismatch_path)
    X = [*X_match, *X_mismatch]
    Y = [*Y_match, *Y_mismatch]

    # shuffle match and mismatch examples:
    X = np.array(X)
    shuffle_index = np.arange(len(Y))
    np.random.shuffle(shuffle_index)
    Y = np.array(Y)[shuffle_index]
    X = np.array(X)[shuffle_index]
    X = [X.swapaxes(0, 1)[0], np.array(X).swapaxes(0, 1)[1]]
    return X, np.array(Y)


def load_match_pairs(features_vectors, path):
    X = []
    Y = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            person = line[0]
            first_index = int(line[1])
            second_index = int(line[2])
            first_vector = features_vectors[person][first_index]
            second_vector = features_vectors[person][second_index]

            X.append([first_vector, second_vector])
            Y.append(0)

    return X, Y


def load_mismatch_pairs(features_vectors, path):
    X = []
    Y = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            first_person = line[0]
            second_person = line[1]
            first_index = int(line[2])
            second_index = int(line[3])
            first_vector = features_vectors[first_person][first_index]
            second_vector = features_vectors[second_person][second_index]

            X.append([first_vector, second_vector])
            Y.append(1)

    return X, Y


def load_feature_vectors(path):
    features_vectors = dict()
    filenames = get_file_names(path)
    for filename in filenames:
        file_path = os.path.join(path, filename)
        features = get_features(file_path)
        features_vectors[filename.split(".")[0]] = [[float(j) for j in i] for i in features]

    return features_vectors


def get_features(file_path):
    file = open(file_path)
    reader = csv.reader(file)
    return list(reader)


def get_file_names(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


def convert_to_var_cnn(X):
    num_samples = X[0].shape[0]
    first_inputs = []
    second_inputs = []

    first_dir_inputs = []
    first_time_inputs = []
    first_metadata_inputs = []

    second_dir_inputs = []
    second_time_inputs = []
    second_metadata_inputs = []

    for i in range(0, num_samples):
        first_feature_vector = X[0][i]
        second_feature_vector = X[1][i]
        first_dir_inputs.append(first_feature_vector[0:1000])
        first_time_inputs.append(first_feature_vector[1000:2000])
        first_metadata_inputs.append(first_feature_vector[2000:2007])
        #first_inputs.append([first_dir_input, first_time_input, first_metadata_input])

        second_dir_inputs.append(second_feature_vector[0:1000])
        second_time_inputs.append(second_feature_vector[1000:2000])
        second_metadata_inputs.append(second_feature_vector[2000:2007])
        #second_inputs.append([second_dir_input, second_time_input, second_metadata_input])

    first_dir_inputs = np.array(first_dir_inputs)
    first_time_inputs = np.array(first_time_inputs)
    first_metadata_inputs = np.array(first_metadata_inputs)

    second_dir_inputs = np.array(second_dir_inputs)
    second_time_inputs = np.array(second_time_inputs)
    second_metadata_inputs = np.array(second_metadata_inputs)

    first_inputs = [first_dir_inputs,
                    first_time_inputs,
                    first_metadata_inputs]
    second_inputs = [second_dir_inputs,
                     second_time_inputs,
                     second_metadata_inputs]
    return [first_inputs, second_inputs]
