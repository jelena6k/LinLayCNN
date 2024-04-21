import random
import numpy as np

import matplotlib.pyplot as plt

def generate_2D_features_from_raw_dataset(data_set, max_length, max_objects_types):
    """
   za svaki objekat ima jedan feature da kaze gde se nalazi  ako je smesten,
   i ako treba da se smesti koja je duzina,
   u nekim slucajevima imamo i dodatne features kao sto je duzina prostora

    Returns
    -------:
   2D_dataset numpy array of size: data_set_size x max_objects_types*2 x max_length

    """
    additional_features = 0

    if "space_length" in data_set.keys():
        additional_features = additional_features + 1
    if "features" in data_set.keys():
        additional_features = additional_features + 1

    num_features = max_objects_types * 2 + additional_features
    print("broj features")
    print(num_features)

    print("max")
    print(max_objects_types)
    nn_dataset = []
    for example_id, example in enumerate(data_set["objects"]):
        nn_example = np.zeros([num_features, max_length])
        if "space_length" in data_set.keys():
            nn_example[0, :data_set["space_length"][example_id]] = np.ones(data_set["space_length"][example_id])
        if "features" in data_set.keys():
            nn_example[1, data_set["features"][example_id]] = 1
        for index, object_type in enumerate(example):
            feature_id = index + additional_features
            nn_example[feature_id, object_type[0]:object_type[0] + object_type[1]] = np.ones(object_type[1])
            nn_example[max_objects_types + feature_id, :object_type[1]] = np.ones(object_type[1])
        nn_dataset.append(nn_example)
    return np.asarray(nn_dataset)


def make_features_and_labels_one_object_type(nn_dataset, object_id, max_objects_types):
    """
    one training example is for one object, so if we have 10 layouts where each contain 5 objects,
    than we will have 10*5 = 50 trainng examples
    for one object from one layout feautre map (neural_net_data output), makes features and labels
    *heree is expected that object positions are predicted in order(3 after 2 and 1, 2 after 1...)
    """
    Y = nn_dataset[:, object_id, :]

    X = np.zeros(nn_dataset.shape)
    X[:, object_id + max_objects_types, :] = nn_dataset[:, object_id + max_objects_types, :]
    X[:, :object_id, :] = nn_dataset[:, :object_id, :]

    return X, Y


def make_features_and_labels(nn_dataset, object_types, length_feature=1, num_features=0, max_object_types = 1):
    """
        one training example is for one object, so if we have 10 layouts where each contain 5 objects,
    than we will have 10*5 = 50 trainng examples, first 10 examples are for object one, next 10 are for object 2 etc

    Returns:
    ------
    X_train size: nn_dataset_size x max_object_types x  max_objects_types*2 x max_length
    Y_train_size: nn_dataset_size  x max_length
    """
    X_train = []
    Y_train = []
    for i in object_types:
        object_id = length_feature + i + num_features
        X, Y = make_features_and_labels_one_object_type(nn_dataset, object_id, max_object_types)
        X_train.extend(X)
        Y_train.extend(Y)
    return np.asarray(X_train), np.asarray(Y_train)