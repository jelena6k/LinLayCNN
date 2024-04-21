import numpy as np
def precission_recall_all(pred1,Y_test,start, end):
    ### concatenate all predictions, and calculate precision and recall as there is just one prediction
    recall = np.logical_and(pred1[start:end],Y_test[start:end]).sum()/Y_test[start:end].sum()
#     print(pred1[start:end].sum())
    precission =  np.logical_and(pred1[start:end],Y_test[start:end]).sum()/pred1[start:end].sum() if pred1[start:end].sum()!=0 else 0
#     print(start)
#     print(precission,recall)
    return precission,recall


def precission_recall(pred1,Y_test,start, end):
    """mean precision and recall on Y_test"""
    recall = np.nan_to_num(np.logical_and(pred1[start:end],Y_test[start:end]).sum(axis = 1)/(Y_test[start:end].sum(axis = 1))).mean()
#     print(pred1[start:end].sum())
    precission =  np.nan_to_num(np.logical_and(pred1[start:end],Y_test[start:end]).sum(axis = 1)/(pred1[start:end].sum(axis = 1))).mean() if pred1[start:end].sum()!=0 else 0
#     print(start)
#     print(precission,recall)
    return precission,recall

def precission_recall_ex(pred1,Y_test,start, end):
    """mean precision and recall on Y_test"""
    recall = np.nan_to_num(np.logical_and(pred1[start:end],Y_test[start:end]).sum(axis = 1)/(Y_test[start:end].sum(axis = 1)))
#     print(pred1[start:end].sum())
    precission =  np.nan_to_num(np.logical_and(pred1[start:end],Y_test[start:end]).sum(axis = 1)/(pred1[start:end].sum(axis = 1))) if pred1[start:end].sum()!=0 else 0
#     print(start)
#     print(precission,recall)
    return precission,recall
def precisions_recalls_types_ex(predictions, test, object_types):# preciznost i recall po  tipu
    precisions = []
    recalls = []
    interval = int(test.shape[0]/len(object_types))
    start = 0
    for i in object_types:
        a,b = precission_recall_ex(predictions,test,start, start+interval)
        start = start+interval
        precisions.append(a)
        recalls.append(b)
    return precisions, recalls
# def precisions_recalls_types(predictions, test, max_object_types):
#     precisions = []
#     recalls = []
#     interval = int(test.shape[0]/max_object_types)
#     start = 0
#     for i in range(0, max_object_types):
#         a,b = precission_recall(predictions,test,start, start+interval)
#         start = start+interval
#         precisions.append(a)
#         recalls.append(b)
#     return precisions, recalls

def precisions_recalls_types(predictions, test, object_types):# preciznost i recall po  tipu
    precisions = []
    recalls = []
    interval = int(test.shape[0]/len(object_types))
    start = 0
    for i in object_types:
        a,b = precission_recall(predictions,test,start, start+interval)
        start = start+interval
        precisions.append(a)
        recalls.append(b)
    return precisions, recalls

def precission_recall_per_ex(predictions,test):
    """koristi se za crtanje histograma preciznosti i recalla po trening poprimeru
    """
    precisions = []
    recalls = []
    for i in range(0,predictions.shape[0]):
        a,b = precission_recall(predictions,test,i, i+1)
        precisions.append(a)
        recalls.append(b)
    return np.asarray(precisions), np.asarray(recalls)


def overlapping_o2_vs_o1(X,Y_pred): ### ovo je samo da proveri dda li se drugi sa prvim sece
    overll = []

    for i in range(0,Y_pred.shape[0]):
        pred= Y_pred[i]
        obj = X[i,1,:]
        overll.append(np.logical_and(pred,obj).sum())
    return np.asarray(overll)

def overlapping_type(X,Y_pred,obj_id,start, end,length_feature = 1):###
    """
    za odredjeni objekat za svaki trening primer proverava koliko se sece sa objektima ciji je id manji od njegovog
    i za svaki tr_primer vraca kolike je overlaping sa svakim od prethodnih
    ----ostalima, dakle vraca samo jedan broj
    """
    overll = []
    ts = X[start:end]
    rs = Y_pred[start:end]
    for i in range(0,rs.shape[0]):

        pred= rs[i]
        overll_per_example = []### za svaki trening primer gledam da li se preseko sa tipovima pre njega
        for j in range(0,obj_id):
            obj = ts[i,length_feature+j,:]
            overll_per_example.append(np.logical_and(pred,obj).sum())
        overll.append(overll_per_example)
    return np.asarray(overll)

def overlapping_types(predictions_array, test_array, object_types,length_feature = 1):
    """vraca listu np.arrays gde za svaki objekat iz object_types imamo niz overlaping-a sa objektima pre njega
    dakle za svaki object_id imamo overlapping sa object_id-1 objekata"""
    overlaping = []
#     interval = int(test.shape[0]/len(object_types))

    for idx in range(0,len(object_types)):
        overlap = overlapping_type( predictions_array[idx],test_array[idx],object_types[idx],0, test_array[idx].shape[0],length_feature)
        overlaping.append(overlap)
    return  overlaping


def violation(Y_test,pred,X_test):
    suma = []
    avg = 0
    for i in range(Y_test.shape[0]):
        if (pred[i].sum() != 0) :
            end = np.where(pred[i]==1)[0][-1]
            length = np.where(X_test[i][0]==1)[0][-1]
        #         print(end,length)
            if end > length:
                suma.append(end - length)
            else:
                suma.append(0)

    return suma