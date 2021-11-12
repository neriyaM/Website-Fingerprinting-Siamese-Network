import numpy as np
from os import listdir
from os.path import isfile, join
import csv


def load_feature_vectors(path):
    features_vectors = dict()
    filenames = get_file_names(path)
    for filename in filenames:
        file_path = join(path, filename)
        features = get_features(file_path)
        features_vectors[filename.split(".")[0]] = [[int(float(j)) for j in i] for i in features]

    return features_vectors


def get_features(file_path):
    file = open(file_path)
    reader = csv.reader(file)
    return list(reader)


def get_file_names(dir_path):
    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


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

            X.append([np.array(first_vector), second_vector])
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
