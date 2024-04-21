#!/usr/bin/env python
# coding: utf-8

# In[1]:


import importlib

import predicting

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


# In[3]:


def export_dataset(dataset, name = "data.json"):
    idx =0
    jsonss  = {}
    for layout in dataset["objects"]:



            jsonss[idx] = layout

            idx = idx+1


    j = json.dumps({k: v.tolist() for k, v in jsonss.items()})

    with open(name, 'w') as f:
        json.dump(j, f)


# In[4]:



# In[5]:

def import_dataset(filename):
    f = open(filename)

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    data = json.loads(data)
    objects = []
    for key, value in data.items():
        objects.append(value)
    data = {'objects': np.asarray(objects)}
    return data

def make_train_test(data, train_size, test_size):
    array = data["objects"][:5000]
    np.random.shuffle(array)
    test = array[:test_size]

    train = array[test_size:train_size+test_size]
    return {'objects':train},{'objects':test}
def make_train(data, train_size):
    array = data["objects"][:5000]
    np.random.shuffle(array)
    train = array[:train_size]

    return {'objects':train}

###PROVERA


# In[6]:


def data_preprocessing(raw_data, max_obj_types = 10,lenght_feature = 0, object_types_prediction = None  ,feature = 0):
    object_types_prediction  = list(range(0,max_obj_types)) if object_types_prediction is None else object_types_prediction
    D2_dataset = generate_2D_features_from_raw_dataset(raw_data,600,max_obj_types)
    X_nn_data,Y__nn_data = make_features_and_labels(D2_dataset,object_types_prediction,lenght_feature,feature,max_obj_types)
    return X_nn_data,Y__nn_data


# def model_predict(model, X_nn_data,max_object_types = 10):
#
#     max_space_length = 600
#     length_feature = 0
#     test_conv_nn = X_nn_data.reshape(-1, 1, max_space_length, max_object_types*2+length_feature)
#     pred = model.predict(test_conv_nn)
#     return pred>0.5

#stari nacin ne valja
# def model_predict(model, X_nn_data,length_feature =0,max_object_types = 10):
#     max_space_length = 600
#     test_conv_nn = X_nn_data.reshape(-1, 1, max_space_length, max_object_types*2+length_feature)
#     pred = model.predict(test_conv_nn)
#     return pred>0.5


def model_predict(model, X_nn_data,length_feature =0,max_object_types = 10,features = 0):
    max_space_length = 600
    test_conv_nn = X_nn_data.reshape(-1, 1, max_space_length, max_object_types*2+length_feature + features)
    pred = model.predict(test_conv_nn)
    return pred

def arrange_one(pred,length):
    # pred = np.random.rand(1,600)
    # length = 5
    if(pred.sum() == 0):
        return pred
    else:
        length = int(length)

        print(length)
        cumu = np.cumsum(pred)
        roled = np.roll(cumu,length)
        value = cumu -roled
        pos = np.argmax(value)
        arranged = np.zeros((600))
        arranged[pos - length: pos] = np.ones((1,length))

        return arranged

def arrange_object(predictions, Y_test):
    arranged_objects = []
    for i in range(0,predictions.shape[0]):
        arranged = arrange_one(predictions[i], Y_test[i].sum())
        arranged_objects.append(arranged)
    return np.squeeze(np.asarray(arranged_objects))
# In[368]:



# data["objects"][0]
# data["objects"][1]
# D2_dataset = generate_2D_features_from_raw_dataset(data,600,10)
# X_nn_data,Y__nn_data = make_features_and_labels(D2_dataset,object_types_prediction,0,0,10)
# Y__nn_data[90000].sum()
# np.argwhere(Y__nn_data[90000]==1)
# X_nn_data[90000][9]


# In[6]:


# def predictions_pipeline(X_test_types,model,object_types_prediction, length_feature = 1):
#     predictions = []
#     teste = []
#     for X_test_obj in X_test_types:
#         for oidx,pred_types in enumerate(predictions):
#             idx = object_types_prediction[oidx]
#             X_test_obj[:,idx+length_feature,:] = pred_types
#         test_conv_nn = X_test_obj.reshape(-1, 1, X_test_obj.shape[2], X_test_obj.shape[1])
#         pred_type = model.predict(test_conv_nn)
#         pred1_type = pred_type>0.5
#         predictions.append(pred1_type)
#
#     return predictions


# In[7]:


# def experiment_A_1_per_ex(data_train,data_test, train_set_sizes,testset_size):
#     def reset_weights(model):
#         import keras.backend as K
#         session = K.get_session()
#         for layer in model.layers:
#             if hasattr(layer, 'kernel_initializer'):
#                 layer.kernel.initializer.run(session=session)
#             if hasattr(layer, 'bias_initializer'):
#                 layer.bias.initializer.run(session=session)
#     precissions_types_train, recall_types_train,precissions_types_test, recall_types_test = [],[],[],[]
#     object_types_prediction  = list(range(0,data_train["objects"].shape[1]))
#     train_raw_dataset = make_train(data_train, train_set_sizes[0])
#     X_test,Y_test = data_preprocessing(data_test)
#     for train_size_id in range(len(train_set_sizes)):
#         if train_size_id > 0:
#             train_raw_dataset = make_train(data_train, train_set_sizes[train_size_id])
#             reset_weights(model)
#
#         X_train,Y_train = data_preprocessing(train_raw_dataset)
#         model = run_nn((X_train, Y_train), (X_test, Y_test),5 )
#         pred_train = model_predict(model, X_train)
#         pred_test = model_predict(model, X_test)
#         precision_train_types, recall_train_types = precisions_recalls_types_ex(pred_train,Y_train,object_types_prediction)
#         precision_test_types, recall_test_types = precisions_recalls_types_ex(pred_test,Y_test,object_types_prediction)
#         precissions_types_train.append(precision_train_types)
#         recall_types_train.append(recall_train_types)
#         precissions_types_test.append(precision_test_types)
#         recall_types_test.append(recall_test_types)
#
#
#
#     return precissions_types_train,recall_types_train,precissions_types_test,recall_types_test


# In[8]:


### data diversity
### ovde odradim precission i recall za svaki trening primer za svaki tip sa svakim test primerom
### all_train_examp sadrzi za svaki trening primer precc i recall za svaki tip sa svakim test primerom
def measure_prec_recall_between_datasets(X_train, X_test):

    prec_recall_between_dataset = []
    idx = 0
    for i in range(X_train.shape[0]):
        train_ex = []
        print(idx)
        idx = idx + 1
        for j in range(0,X_train.shape[1]):

            train = np.full((X_test.shape[0],600),X_train[i,j])
            prec,recall = precission_recall_per_ex(train,X_test[:,j,:])
            train_ex.append([prec,recall])
        prec_recall_between_dataset.append(train_ex)
 
    return np.asarray(prec_recall_between_dataset)

def measure_f1_between_datasets(prec_recall_between_dataset):
        return np.nan_to_num( 2 * (prec_recall_between_dataset[:,:,0,:] * prec_recall_between_dataset[:,:,1,:])/(prec_recall_between_dataset[:,:,0,:] + prec_recall_between_dataset[:,:,1,:]))
# with open('all_train_examp_vs_train_np.npy', 'wb') as f:
#     np.save(f, all_train_examp_vs_train_np1) 
# all_train_examp_vs_train_np1 = np.load('all_train_examp_vs_train_np.npy')
def leave_diagonal(datasetf1):
    data_f1 = []
    for i in range(0,datasetf1.shape[0]):
        data_f1.append( np.concatenate((datasetf1[i,:i],datasetf1[i,i+1:])))
    return np.asarray(data_f1)
def plot_pdf_dataset(dataset):
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    mean=dataset.mean()
    sigma = dataset.std()
    x_axis = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    plt.plot(x_axis, stats.norm.pdf(x_axis, mean, sigma))
    return (mean,sigma)

# OVAKO SE KORISTI
# data = experiment_1.import_dataset("data_B.json")
# data_train,data_test = experiment_1.make_train_test(data, 5000, 0)
# X_train = experiment_1.generate_2D_features_from_raw_dataset(data_train,600,10)
# dataset_prec_recall_per_type = experiment_1.measure_prec_recall_between_datasets(X_train[:,:10,:],X_train[:,:10,:])
# dataset_f1_per_type = experiment_1.measure_f1_between_datasets(dataset_prec_recall_per_type)
# a = (dataset_f1_per_type[:,:,:]).mean(axis = 1)
# ld = experiment_1.leave_diagonal(a)
# mean,sigma = experiment_1.plot_pdf_dataset(ld)

#save
# with open('data_diversity_layout2.npy', 'wb') as f:
#     np.save(f, ld)
#load
# all_train_examp_vs_train_np1 = np.load('all_train_examp_vs_train_np.npy')


# In[9]:


def dataset_distribution(dataset):
    for i in range(1,10):
        mean =dataset["objects"][:,i,0].mean()
        std = dataset["objects"][:,i,0].std()
        x_axis = np.linspace(mean - 3*std, mean + 3*std, 100)
        plt.plot(x_axis, stats.norm.pdf(x_axis, mean, std))
    plt.legend(["object 2","object 3","object 4","object 5","object 6","object 7","object 8","object 9","object 10"])    
    plt.savefig('starting_position_distributino.png', edgecolor='none')


# In[10]:


### Model performances, measured on predictions
### KAd racunam metriku na trening skupu nisu mi iste velicine trening skupa i onda ne mogu sve rezultate da 
###smestim u numpy array, pa mi je zato sizeid kao argument

def f1_obj(prec,rec):
    return np.nan_to_num(2*prec*rec/(prec+rec))

def f1_avg_1(predictions_prec,predictions_recall, iterationid,sizeid):#za iteraciju iteration i za velicinu 
    # trening skupa size racuna f1 za svaki objekat pa 
    # f1 za layout kao sredinu f1 za svaki objekat
    prec = np.asarray(predictions_prec[iterationid][sizeid])
    rec = np.asarray(predictions_recall[iterationid][sizeid])
    f1 = np.nan_to_num(2*prec*rec/(prec+rec))
    f1 = f1.mean(axis = 0)
    return f1

#daje bolje rezulAate od prethodne
def f1_avg_2(predictions_prec,predictions_recall,iterationid, sizeid):#za iteraciju iteration i za velicinu trening skupa sizeid
    #racuna prec i recall za  svaki layout kao srednji prec i recall po objektima iz tog layouta pa onda racuna f1 
    # na osnovu tih usrednjenih vrenosti
    prec = np.asarray(predictions_prec[iterationid][sizeid]).mean(axis = 0)
    rec = np.asarray(predictions_recall[iterationid][sizeid]).mean(axis = 0)
    f1 = np.nan_to_num(2*prec*rec/(prec+rec))
    return f1


def f1_avg_2_a(predictions_prec,predictions_recall):#
    #racuna prec i recall za  svaki layout kao srednji prec i recall po objektima iz tog layouta pa onda racuna f1 
    # na osnovu tih usrednjenih vrenosti
    prec = np.asarray(predictions_prec).mean(axis = 2)
    rec = np.asarray(predictions_recall).mean(axis = 2)
    f1 = np.nan_to_num(2*prec*rec/(prec+rec))
    return f1

# def mean_f1_avg_2_along_iterations(predictions_prec,predictions_recall, size):#ovde usrednjavam performanse(f1, ili prec ili recall) 
#     #po iteracijama
#     """returns: f1 measure for the given dataset size"""
    
#     suma = f1_avg_2(predictions_prec,predictions_recall,0,size).mean() + f1_avg_2(predictions_prec,predictions_recall,1,size).mean() + \
#            + f1_avg_2(predictions_prec,predictions_recall,2,size).mean() + f1_avg_2(predictions_prec,predictions_recall,3,size).mean() + f1_avg_2(predictions_prec,predictions_recall,4,size).mean()
#     return (suma/5)

def mean_f1_avg_2_along_iterations(predictions_prec,predictions_recall, size):#ovde usrednjavam performanse(f1, ili prec ili recall)
    #po iteracijama
    """returns: f1 measure for the given dataset size"""
    suma = 0
    for i in range(0,predictions_prec.shape[0]):
        suma = suma + f1_avg_2(predictions_prec,predictions_recall,i,size).mean()  
    return (suma/predictions_prec.shape[0])

def mean_metric_along_iterations_per_type(predictions_performances, size):#ovde usrednjavam performanse(f1, ili prec ili recall) 
    #po iteracijama za svaki objekat
    num_types = predictions_performances.shape[2]
    suma = 0
    print(predictions_performances.shape[0])
    for i in range(0,predictions_performances.shape[0]):
        suma = suma + np.concatenate(predictions_performances[i][size]).reshape(num_types,-1).mean(axis = 1)

    return (suma/predictions_performances.shape[0])



# def mean_metric_along_iterations(predictions_performances):#ovde usrednjavam performanse(f1, ili prec ili recall) 
#     #po iteracijama
#     """returns: f1 measure for the given dataset size"""

#     suma = np.asarray(predictions_performances[0]).mean() + np.asarray(predictions_performances[1]).mean() + \
#            np.asarray(predictions_performances[2]).mean() + np.asarray(predictions_performances[3]).mean() + \
#            np.asarray(predictions_performances[4]).mean()
#     print(np.asarray(predictions_performances[0]).mean().shape)
#     return (suma/5)
# def mean_f1_along_iterations(func, size):#ovde usrednjavam po iteracijama, idemo sa f1avg2 jer daje bolje od f1avg1
#     suma = func(0,size).mean()+ func(1,size).mean()+ func(2,size).mean()+ func(3,size).mean()+ func(4,size).mean()
#     return (suma/5)

# def f1_avg_types(iteration, sizeid):#za iteraciju iteration i za velicinu trening skupa size racuna prec i recall za
#     # svaki layout kao srednji prec i recall po objektima iz tog layouta pa onda racuna f1 na osnovu tih usrednjenih vrenosti
#     prec = np.asarray(precissions_types_test_iterations[iteration][sizeid])
#     rec = np.asarray(recall_types_test_iterations[iteration][sizeid])
#     f1 = np.nan_to_num(2*prec*rec/(prec+rec))
#     return f1
# def f1_avg_types_train(iteration, sizeid):#za iteraciju iteration i za velicinu trening skupa size racuna prec i recall za
#     # svaki layout kao srednji prec i recall po objektima iz tog layouta pa onda racuna f1 na osnovu tih usrednjenih vrenosti
#     prec = np.asarray(precissions_types_train_iterations[iteration][sizeid])
#     rec = np.asarray(recall_types_train_iterations[iteration][sizeid])
#     f1 = np.nan_to_num(2*prec*rec/(prec+rec))
#     return f1
# def f1_avg_avg_train(iteration, sizeid):#za iteraciju iteration i za velicinu trening skupa size racuna prec i recall za
#     # svaki layout kao srednji prec i recall po objektima iz tog layouta pa onda racuna f1 na osnovu tih usrednjenih vrenosti
#     prec = np.asarray(precissions_types_train_iterations[iteration][sizeid]).mean(axis = 0)
#     rec = np.asarray(recall_types_train_iterations[iteration][sizeid]).mean(axis = 0)
#     f1 = np.nan_to_num(2*prec*rec/(prec+rec))
#     return f1


# In[516]:





# In[53]:


### Plot model performances
def plot_f1_by_size(predictions_prec,predictions_recall,colour,iterations = 5,linestylee='solid',):
    
    plt.scatter(["100","200","500","1000","4000"],[mean_f1_avg_2_along_iterations(predictions_prec,predictions_recall,sizeid) for sizeid in range(0,iterations)],color = colour)
    np.set_printoptions(precision=2)
    plt.plot([mean_f1_avg_2_along_iterations(predictions_prec,predictions_recall,sizeid) for sizeid in range(0,iterations)],color = colour,linestyle = linestylee)
    print([mean_f1_avg_2_along_iterations(predictions_prec,predictions_recall,sizeid)*100 for sizeid in range(0,iterations)])


###crata mean pr i recall po iteracijama po tipu
def plot_metric_by_type_and_size(dataset,title):
    sizes = dataset.shape[1]
    for i in range(0, sizes):
        plt.plot(mean_metric_along_iterations_per_type(dataset, i))

    plt.yticks([0.5,0.6,0.7,0.8,0.9,1])
    plt.title(title)
    plt.legend(["100","200","500","1000","4000"])


# In[89]:




# In[20]:


# Dataset generation for experiment 1

# train_raw_dataset = generate_raw_dataset_A(10000,10,600)
# export_dataset(train_raw_dataset)


# In[272]:




# ### Measuring dataset diversity

# In[59]:




def experiment_A_2_per_ex(data_train,data_test, train_set_sizes,testset_size, max_object_types = 10):
    def reset_weights(model):
        import keras.backend as K
        session = K.get_session()
        for layer in model.layers: 
            if hasattr(layer, 'kernel_initializer'): 
                layer.kernel.initializer.run(session=session)
            if hasattr(layer, 'bias_initializer'):
                layer.bias.initializer.run(session=session)
    max_space_length = 600

    length_feature = 0
    precissions_types_pip, recall_types_pip,precissions_types, recall_types = [],[],[],[]
    pipeline_predictions, separate_predictions = [],[]
    X_predictions = []
    # object_types_prediction  = list(range(0,10))
    train_raw_dataset = make_train(data_train, train_set_sizes[0])
    X_test,Y_test = data_preprocessing(data_test,max_object_types)
    for train_size_id in range(len(train_set_sizes)):
        if train_size_id > 0:        
            train_raw_dataset = make_train(data_train, train_set_sizes[train_size_id])
            reset_weights(model)

        X_train,Y_train = data_preprocessing(train_raw_dataset,max_object_types)
        model = run_nn((X_train, Y_train), (X_test, Y_test),5 )
        X_test_types, Y_types = split_dataset_by_type(X_test, testset_size),split_dataset_by_type(Y_test, testset_size)
        object_types_prediction  = list(range(0,max_object_types))
        predictions_pip_types,predictions_features = predicting.predictions_pipeline(X_test_types,model,object_types_prediction, Y_test,length_feature = 0)
        print(np.asarray(predictions_pip_types).shape)
        predictions_pip = np.concatenate(predictions_pip_types,0)
        pipeline_predictions.append(predictions_pip)
        
        predictions_features = np.concatenate(predictions_features)
        X_predictions.append(predictions_features)
        
#         precision_train_types, recall_train_types = precisions_recalls_types_ex(pred_train,Y_train,object_types_prediction)
        
        precision_test_types, recall_test_types = precisions_recalls_types_ex(predictions_pip,Y_test,object_types_prediction)
#         precissions_types_train.append(precision_train_types)
#         recall_types_train.append(recall_train_types)
        precissions_types_pip.append(precision_test_types)
        recall_types_pip.append(recall_test_types)

        ###sequential
        test_conv_nn = X_test.reshape(-1, 1, max_space_length, max_object_types*2+length_feature)
        pred = model.predict(test_conv_nn,max_object_types)
        # pred1 = pred>0.5
        pred1 = experiment_1.arrange_object(pred, Y_test)
        separate_predictions.append(pred1)
        precisions, recalls = precisions_recalls_types_ex(pred1,Y_test,object_types_prediction)
        precissions_types.append(precisions)
        recall_types.append(recalls)
    return X_predictions,X_test,Y_test,np.asarray(pipeline_predictions),np.asarray(separate_predictions),np.asarray(precissions_types_pip),np.asarray(recall_types_pip),np.asarray(precissions_types),np.asarray(recall_types)


# In[97]:



def predictions_pipeline(X_test_types,model,object_types_prediction, length_feature = 1):
    predictions = []
    test_features = []
    for X_test_obj in X_test_types:
        for oidx,pred_types in enumerate(predictions):
            idx = object_types_prediction[oidx]
            X_test_obj[:,idx+length_feature,:] = pred_types
        test_conv_nn = X_test_obj.reshape(-1, 1, X_test_obj.shape[2], X_test_obj.shape[1])
        pred_type = model.predict(test_conv_nn,len(object_types_prediction))
        pred1_type = pred_type>0.5
        predictions.append(pred1_type)
        test_features.append(X_test_obj)
    return predictions,test_features



def change_shape(niz):
    return niz.reshape(1,5,10,1000)



def overlapping_type(Y_pred, X,obj_id,length_feature = 1):### 
    """
    za odredjeni objekat za svaki trening primer proverava koliko se sece sa objektima ciji je id manji od njegovog
    i za svaki tr_primer vraca kolike je overlaping sa svakim od prethodnih 
    ----ostalima, dakle vraca samo jedan broj
    """
    overll = []

    for i in range(0,X.shape[0]):#za svaki trening primer

        pred= Y_pred[i]
        overll_per_example = []### za svaki trening primer gledam da li se preseko sa tipovima pre njega
        for j in range(0,obj_id):# overlaping sa tipovma pre njega
            obj = X[i,length_feature+j,:]
            overll_per_example.append(np.logical_and(pred,obj).sum())
        overll.append(overll_per_example)
    return np.asarray(overll)

def overlapping_types(predictions_array, test_array, object_types,length_feature = 1):
    """vraca listu np.arrays gde za svaki objekat iz object_types imamo niz overlaping-a sa objektima pre njega
    dakle za svaki object_id imamo overlapping sa object_id-1 objekata"""
    overlaping = []
#     interval = int(test.shape[0]/len(object_types))

#     for idx in range(0,len(object_types)):
#         overlap = overlapping_type( predictions_array[idx],test_array[idx],object_types[idx],0, test_array[idx].shape[0],length_feature)
#         overlaping.append(overlap)

    for idx in object_types:
        overlap = overlapping_type( predictions_array[idx],test_array[idx],idx,length_feature)
        overlaping.append(overlap)
        
    return  overlaping

def overlaping_predictions(predictions, features):
    ### ne gleda overlaping z aprvi objekat posto je on svakako nula jer pre njega nema nikoga
    testset_size = 1000
    object_types_prediction  = list(range(0,10))
    pred_types, X_types = split_dataset_by_type(np.asarray(predictions), testset_size),split_dataset_by_type(np.asarray(features), testset_size)
    # pred_types, X_types = split_dataset_by_type(np.asarray(pred_pip[0]), testset_size)[1:],split_dataset_by_type(np.asarray(X_predictions[0]), testset_size)[1:]

    overl = overlapping_types(np.asarray(pred_types),np.asarray(X_types), object_types_prediction[1:],length_feature = 0)
    sum_overllap = []
    avg_overllap = []
    for overl_objid in overl:
        sum_overllap.append(np.sum(overl_objid,axis =  1))
        avg_overllap.append(np.mean(overl_objid,axis =  1))
    return np.asarray(sum_overllap),np.asarray(avg_overllap)
    
    sum_overllap = np.asarray(sum_overllap)  
    sum_overllap_hist = sum_overllap.reshape(-1,1)
    hist = plt.hist(sum_overllap_hist, bins = [*range(0, int(max(sum_overllap_hist)), 1)],label="overlaping_sum")
    plt.legend()
def overlaping_sizes(pred_seq,X_predictions,pip = True):
        suma_pip = []
        avg_pip = []
        for i in range (0,5):
            features = X_predictions[i] if pip else X_predictions
            a,b = overlaping_predictions(pred_seq[i], features)
            suma_pip.append(a)
            avg_pip.append(b)
        return np.nan_to_num(np.asarray(suma_pip)), np.nan_to_num(np.asarray(avg_pip))





# In[21]:


def avg_iterations(pred_it,X_test_iterations,pip = True):
    suma = []
    avg = []
    for i in range(5):
        suma_seq, avg_seq = overlaping_sizes(pred_it[i],X_test_iterations[i],pip)
        avg.append(avg_seq)
        suma.append(suma_seq)
    return np.asarray(avg),np.asarray(suma)


# In[22]:


#avg, suma = avg_iterations([results[i][3] for i in range(0,5)],[results[i][0] for i in range(0,5)],pip = True)




def space_type(Y_pred, X,obj_id,length_feature = 1):### 
    """
    za odredjeni objekat za svaki trening primer proverava koliko se sece sa objektima ciji je id manji od njegovog
    i za svaki tr_primer vraca kolike je overlaping sa svakim od prethodnih 
    ----ostalima, dakle vraca samo jedan broj
    """
    overll = []

    for i in range(0,X.shape[0]):#za svaki trening primer

        pred= Y_pred[i]
        overll_per_example = []

        obj = X[i,length_feature+obj_id-1,:]
        pred_arr = np.where(pred == 1)[0]
        featu_arr  = np.where(obj == 1)[0]

        start = pred_arr[0] if len(pred_arr)>0 else -1000
        end =  featu_arr[-1] if len(featu_arr)>0 else 1000
        space = start - end
        space = 0 if space < 0 else space
        overll.append(space)
    return np.asarray(overll)

def space_types(predictions_array, test_array, object_types,length_feature = 1):
    """vraca listu np.arrays gde za svaki objekat iz object_types imamo niz overlaping-a sa objektima pre njega
    dakle za svaki object_id imamo overlapping sa object_id-1 objekata"""
    overlaping = []
#     interval = int(test.shape[0]/len(object_types))

#     for idx in range(0,len(object_types)):
#         overlap = overlapping_type( predictions_array[idx],test_array[idx],object_types[idx],0, test_array[idx].shape[0],length_feature)
#         overlaping.append(overlap)

    for idx in object_types:
        overlap = space_type( predictions_array[idx],test_array[idx],idx,length_feature)
        overlaping.append(overlap)
        
    return  overlaping
def space_predictions(predictions, features):
    ### ne gleda overlaping z aprvi objekat posto je on svakako nula jer pre njega nema nikoga
    testset_size = 1000
    object_types_prediction  = list(range(0,5))
    pred_types, X_types = split_dataset_by_type(np.asarray(predictions), testset_size),split_dataset_by_type(np.asarray(features), testset_size)
    # pred_types, X_types = split_dataset_by_type(np.asarray(pred_pip[0]), testset_size)[1:],split_dataset_by_type(np.asarray(X_predictions[0]), testset_size)[1:]

    overl = space_types(np.asarray(pred_types),np.asarray(X_types), object_types_prediction[1:],length_feature = 0)
    return overl 

def space_sizes(pred_seq,X_predictions,pip = True): #avg_spaces_seq = space_sizes(pred_seq,X_test,pip = False)

        suma_pip = []
        for i in range (0,5):
            features = X_predictions[i] if pip else X_predictions
            a = space_predictions(pred_seq[i], features)
            print(len(a[0]))
            
            suma_pip.append(np.asarray(a))

        return np.nan_to_num(np.asarray(suma_pip))


# In[25]:


def avg_iterations_space(pred_it,X_test_iterations,pip = True):
    suma = []
    avg = []
    for i in range(5):
        avg_seq = space_sizes(pred_it[i],X_test_iterations[i],pip)
        avg.append(avg_seq)
        
    return np.asarray(avg)

#avg_space_pip = avg_iterations_space([results[i][3] for i in range(0,5)],[results[i][0] for i in range(0,5)],pip = True)
#avg_space_seq = avg_iterations_space([results[i][4] for i in range(0,5)],[results[i][1] for i in range(0,5)],pip = False)


# In[56]:




def length_diffrence(predictions, labels):
    length_diff = []

    for i in range(0,5):#sizes
        lengths = np.abs(predictions[i].sum(axis = 1) - labels.sum(axis = 1))

        
        testset_size = 1000
        object_types_prediction  = list(range(0,10))
        pred_types = split_dataset_by_type(np.asarray(lengths), testset_size)
        print(np.asarray(lengths).shape)
        print(pred_types.shape)
        length_diff.append(pred_types[1:])
    return length_diff

def avg_iterations_length(pred_it,Y_test):
    lenghts = []
    for i in range(5):
        length = length_diffrence(pred_it[i],Y_test[i])
        lenghts.append(length)


    return np.asarray(lenghts)

#avg_length_pip = avg_iterations_length([results[i][3] for i in range(0,5)],[results[i][2] for i in range(0,5)])
#avg_length_seq = avg_iterations_length([results[i][4] for i in range(0,5)],[results[i][2] for i in range(0,5)])


# In[107]:



