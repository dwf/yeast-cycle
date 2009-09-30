from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import sys

def get_these(sortedidx, rawdata, likelihoods, h5file, threshold=np.inf):
    results = []
    for index, row in izip(sortedidx, rawdata[sortedidx]):
        if likelihoods[index] >= threshold:
            break
        labels, numfound = ndi.label(h5file.getNode(row['image']).read())
        objslices = ndi.find_objects(labels)
        theslice = objslices[row['objnum'] - 1] #starts from 0
        arr = labels[theslice] == row['objnum']
        results.append((arr, likelihoods[index]))
    return results
    
def get_low_ranks(rawdata, likelihoods, h5file, threshold = -26, lower=-50):
    results = []
    idx = np.where((likelihoods < threshold) & (likelihoods > lower))[0]
    lowest_to_highest = np.argsort(likelihoods[idx])
    sortedidx = idx[lowest_to_highest]
    print len(sortedidx)
    
    sys.stdout.flush()
    results = get_these(sortedidx, rawdata, likelihoods, h5file, threshold)
    return results

def get_top_n(rawdata, likelihoods, h5file, n=9):
    results = []
    idx = np.argsort(likelihoods)
    print idx
    sortedidx = idx[-n:][::-1]
    return get_these(sortedidx, rawdata, likelihoods, h5file)

def get_bottom_n(rawdata, likelihoods, h5file, n=9):
    results = []
    idx = np.argsort(likelihoods)
    print idx
    sortedidx = idx[:n][::-1]
    return get_these(sortedidx, rawdata, likelihoods, h5file)

def get_top_in_class(loglik):
    