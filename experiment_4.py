
import importlib
imported_module = importlib.import_module("testing")
importlib.reload(imported_module)
from testing import *
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[2]:


from generate_dataset import *
from neural_net import *
from plotting import *
from postprocessing_dataset import *
from predicting import *
from preprocessing_dataset import *
from testing import *
import json


def import_dataset(filename):
    f = open(filename)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    data = json.loads(data)
    objects = []
    spaces = []
    features = []
    types = []
    for key,value in data["objects"].items():
        objects.append(value)
    for key,value in data["space_length"].items():
        spaces.append(value)
    for key,value in data["features"].items():
        features.append(value)

    if "types" in data:
        for key, value in data["types"].items():
            types.append(value)
    data = {'objects': np.asarray(objects), 'space_length': np.asarray(spaces),'features': np.asarray(features), "types":np.asarray(types)}
    return data


def export_dataset(dataset, name="data.json"):
    idx = 0
    jsonss = {}
    jsons_space = {}
    jsons_features = {}
    typess = {}
    for layout in dataset["objects"]:
        jsonss[idx] = layout

        idx = idx + 1

    idx = 0
    for layout in dataset["space_length"]:
        jsons_space[idx] = layout

        idx = idx + 1


    idx = 0
    for layout in dataset["features"]:
        jsons_features[idx] = layout

        idx = idx + 1

    idx = 0

    if "types" in dataset:
        for layout in dataset["types"]:
            typess[idx] =  layout
            idx = idx + 1
        print(typess)

    with open(name, 'w') as f:
        json.dump(json.dumps({"objects": {k: v.tolist() for k, v in jsonss.items()},
                    "space_length": {k: v.tolist() for k, v in jsons_space.items()},
                              "features": {k: v.tolist() for k, v in jsons_features.items()}        ,
                             "types": {k: v.tolist() for k, v in typess.items()}}
                             ), f)


# In[4]:

def make_train_test(data, train_size, test_size):
    p = np.random.permutation(len(data["objects"]))

    objects = data["objects"][p]
    spaces = data["space_length"][p]
    features = data["features"][p]
    types = data["types"][p]
    test_obj = objects[:test_size]
    test_space = spaces[:test_size]
    test_features = features[:test_size]
    test_types = types[:test_size]
    train_obj = objects[test_size:train_size + test_size]
    train_spaces = spaces[test_size:train_size + test_size]
    train_features = features[test_size:train_size + test_size]
    train_types = types[test_size:train_size + test_size]
    return {'objects': train_obj, 'space_length': train_spaces, 'features':train_features,'types':train_types}, {'objects': test_obj, 'space_length': test_space,'features':test_features,'types':test_types}


def make_train(data, train_size):
    objects = data["objects"]
    spaces_len = data["space_length"]
    features = data["features"]
    types = data["types"]

    p = np.random.permutation(len(data["objects"]))

    objects  = objects[p]
    objects = objects[:train_size]

    spaces_len  = spaces_len[p]
    spaces_len = spaces_len[:train_size]

    featu = features[p]
    featu = featu[:train_size]

    type = types[p]
    type = type[:train_size]
    return {'objects': objects,"space_length":spaces_len,"features":featu,"types":type}




def overlaping_predictions(predictions, features):
    overll = []
    distance = []
    for idx in range(0,predictions.shape[0]):
        # overlaping = np.logical_and(predictions[idx], obj).sum()

        length02  = np.where(predictions[idx] == 1)[0]
        lengtho1 = np.where(features[idx, 2, :] == 1)[0]
        # print(length02[0] -  lengtho1[-1])
        # print(lengtho1[0] -  length02[-1])
        dist = 0
        overlaping = 0

        if length02.sum() != 0:
            dist =  max([length02[0] -  lengtho1[-1], lengtho1[0] -  length02[-1]])


            if dist <0 :
                overlaping = -dist
                dist = -1


            elif dist > 0:
                overlaping = -1


        distance.append(dist)
        overll.append(overlaping)
    return overll,distance

def overlaping_sizes(pred_seq, features):
    suma_pip = []
    dist = []
    for i in range(0, pred_seq.shape[0]):  ### ovde ide broj razlicitih velicina trening skupa, bilo hardkodirano pa sam ga promenila, ako nekad javlja gresku mozda je to
        a,b= overlaping_predictions(pred_seq[i], features[i])
        suma_pip.append(a)
        dist.append(b)
    return np.nan_to_num(np.asarray(suma_pip)),np.nan_to_num(np.asarray(dist))


# In[21]:


def overl_dist_iterations(pred_it, X_test_iterations):
    suma = []
    distance = []
    for i in range(pred_it.shape[0]):
        suma_seq,dist = overlaping_sizes(pred_it[i], X_test_iterations[i])
        suma.append(suma_seq)
        distance.append(dist)

    return np.asarray(suma),np.asarray(distance)