import numpy as np
def split_dataset_by_type(dataset, interval):
    splitted = []
    for i in range(0, int(dataset.shape[0]/interval)):
        start = i*interval
        end = start + interval
        end = end if end < dataset.shape[0] else  dataset.shape[0]
        splitted.append(dataset[start:end])
    return np.asarray(splitted)