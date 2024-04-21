
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
    for key,value in data["objects"].items():
        objects.append(value)
    for key,value in data["space_length"].items():
        spaces.append(value)
    data = {'objects': np.asarray(objects), 'space_length': np.asarray(spaces)}
    return data


def export_dataset(dataset, name="data.json"):
    idx = 0
    jsonss = {}
    jsons_space = {}
    for layout in dataset["objects"]:
        jsonss[idx] = layout

        idx = idx + 1

    idx = 0
    for layout in dataset["space_length"]:
        jsons_space[idx] = layout

        idx = idx + 1

    j = json.dumps({"objects": {k: v.tolist() for k, v in jsonss.items()},
                    "space_length": {k: v.tolist() for k, v in jsons_space.items()}})

    with open(name, 'w') as f:
        json.dump(json.dumps({"objects": {k: v.tolist() for k, v in jsonss.items()},
                    "space_length": {k: v.tolist() for k, v in jsons_space.items()}}), f)


# In[4]:




def make_train_test(data, train_size, test_size):
    p = np.random.permutation(len(data["objects"]))

    objects = data["objects"][p]
    spaces = data["space_length"][p]
    test_obj = objects[:test_size]
    test_space = spaces[:test_size]

    train_obj = objects[test_size:train_size + test_size]
    train_spaces = spaces[test_size:train_size + test_size]

    return {'objects': train_obj, 'space_length': train_spaces}, {'objects': test_obj, 'space_length': test_space}


def make_train(data, train_size):
    objects = data["objects"]
    spaces_len = data["space_length"]

    p = np.random.permutation(len(data["objects"]))

    objects  = objects[p]
    objects = objects[:train_size]

    spaces_len  = spaces_len[p]
    spaces_len = spaces_len[:train_size]

    return {'objects': objects,"space_length":spaces_len}

### distance to the right edge
### ovde da se preuredi da daje za distance izmedju objekata
def space_type(Y_pred, X ):  ###
    """
    za odredjeni objekat za svaki trening primer proverava koliki je prostor izmedju njega i ivice prostora

    """
    overll = []
    length_feature = 1.
    for i in range(0, X.shape[0]):  # za svaki trening primer

        pred = Y_pred[i]
        overll_per_example = []

        obj = X[i, 0, :]
        pred_arr = np.where(pred == 1)[0]
        featu_arr = np.where(obj == 1)[0]

        end = pred_arr[-1] if len(pred_arr) > 0 else 1000
        edge = featu_arr[-1] if len(featu_arr) > 0 else -1000
        space = edge - end
        space = 0 if space < 0 else space
        overll.append(space)
    return np.asarray(overll)



def space_sizes(pred_seq, X_predictions):  # avg_spaces_seq = space_sizes(pred_seq,X_test,pip = False)

    suma = []
    for i in range(0, 5):# train set sizes
        features = X_predictions[i]
        a = space_type(np.asarray(pred_seq[i]), np.asarray(features))

        suma.append(np.asarray(a))

    return np.nan_to_num(np.asarray(suma))


# In[25]:


def avg_iterations_space(pred_it, X_test_iterations):
    suma = []
    avg = []
    for i in range(0,pred_it.shape[0]): # iterations
        it = space_sizes(pred_it[i], X_test_iterations[i])
        avg.append(it)

    return np.asarray(avg)

###distance to the left edge

def test_ex_distance_left_edge(Y_pred):  ###
    """
    za odredjeni objekat za svaki trening primer proverava koliki je prostor izmedju njega i ivice prostora

    """
    distances = []
    for i in range(0, Y_pred.shape[0]):  # za svaki trening primer

        pred = Y_pred[i]

        dist = np.where(pred == 1)[0]

        dist = dist[0] if len(dist)>0 else 0

        distances.append(dist)
    return np.asarray(distances)



def training_size_distance_left_edge(pred_seq):  # avg_spaces_seq = space_sizes(pred_seq,X_test,pip = False)

    suma = []
    for i in range(0, pred_seq.shape[0]):# train set sizes
        a = test_ex_distance_left_edge(np.asarray(pred_seq[i]))

        suma.append(np.asarray(a))

    return np.nan_to_num(np.asarray(suma))



def iterations_distance_left_edge(pred_it):
    suma = []
    avg = []
    for i in range(0,pred_it.shape[0]): # iterations
        it = training_size_distance_left_edge(pred_it[i])
        avg.append(it)

    return np.asarray(avg)


###overlapping



def overlaping_predictions(predictions, features):
    overll = []
    distance = []
    for idx in range(0,predictions.shape[0]):
        obj = features[idx, 1, :]
        # overlaping = np.logical_and(predictions[idx], obj).sum()

        length02  = np.where(predictions[idx] == 1)[0]
        lengtho1 = np.where(features[idx, 1, :] == 1)[0]
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


### to the left/right
def to_the_left_to_o1(o1,o2):
    def begining(a):
        return np.where(a == 1)[0][0]

    beggining01 = np.apply_along_axis(begining, 3, o1)
    beggining02 = np.apply_along_axis(begining, 3, o2)
    left = (beggining01 - beggining02) > 0
    right = ~left
    return left, right

###violation

def violation(pred,X_test):
    viol = []
    avg = 0
    correct = []
    dist = []
    for i in range(pred.shape[0]):
        if (pred[i].sum() != 0) :
            end = np.where(pred[i]==1)[0][-1]
            length = np.where(X_test[i][0]==1)[0][-1]
        #         print(end,length)
            cor = 1
            viola = 0
            dis = 0
            if end > length:
                viola = end - length
                cor = 0
                dis = -1
            elif end == length:
                cor = 1
                viola = 0
                dis = 0
            else:
                viola = -1
                dis = length -end
                cor = 0
        correct.append(cor)
        dist.append(dis)
        viol.append(viola)
    return np.asarray(viol),np.asarray(correct),np.asarray(dist)
# def violation_iteration_size(pred,X_test):
#     violations = []
#     for i in range(pred.shape[0]):#iteration
#         violations.append([violation(pred[i][j],X_test[i][j]) for j in range(0,pred.shape[1])])#size
#     return np.asarray(violations)

def violation_iteration_avg(pred,X_test):
    violations_all = []
    dist_all = []
    cor_all =[]
    for i in range(pred.shape[0]):#iteration
        violations  = []
        dists = []
        corrects = []
        for j in range(0,pred.shape[1]):
            viola,correct,dist = violation(pred[i][j],X_test[i][j])
#            nepreskace = niz == 0

            #a = np.ma.masked_where(nepreskace,niz).mean(axis = 0)
            violations.append(viola)
            corrects.append(correct)
            dists.append(dist)
#            violations_count.append((niz!=0).sum()/niz.shape[0]*100)
#            corrects.append(correct.shape[0]/niz.shape[0]*100)
#        violations_all_count.append(violations_count)
        violations_all.append(violations)
        dist_all.append(dists)
        cor_all.append(corrects)

        #        cor_all.append(corrects)
    return np.asarray(violations_all),np.asarray(dist_all),np.asarray(cor_all)



#violations_all vraca prosecno preskakanje po iteraciji za razlicite  velicine tr skupa
#violations_all_count - vraca koliko preskacu po iteraciji za razlicite velicine tr skupa

# length to mi je trebalo kad sam radila predikcije na drugi nacin, sad ne treba ali neka ga zlu ne trebalo
# def length_diffrence(predictions, labels):
#     length_diff = []

#     for i in range(0,5):#sizes
#         lengths = np.abs(predictions[i].sum(axis = 1) - labels.sum(axis = 1))


#         length_diff.append(lengths)
#     return length_diff

# def avg_iterations_length(pred_it,Y_test):
#     lenghts = []
#     for i in range(2):
#         length = length_diffrence(pred_it[i],Y_test[i])
#         lenghts.append(length)


#     return np.asarray(lenghts)

# ###length

# avg_length = avg_iterations_length(preds_test,Y_types_test_iterations[:,0,:,:])# zato sto su za jednu iteraciju isti test primeri za sve velicine trening skupa, sto je  poenata da se uporedi
# avg_length.mean(axis = 0).mean(axis = 1)
# plt.plot(["100","200","500","1000","5000"],avg_length.mean(axis = 0).mean(axis = 1),"o-b" )
# plt.savefig("length_diff")