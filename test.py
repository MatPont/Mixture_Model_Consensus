import os
from coclust.coclustering import CoclustInfo, CoclustMod, CoclustSpecMod
from coclust.clustering import SphericalKmeans
import Cluster_Ensembles as CE
from scipy import io
import sys
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from random import *

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

datasets = ["classic4", "cstr", "ng5", "rcv4", "ng20"]

def run(dataset_name):
    print(dataset_name)
    
    ##############
    # Load Data
    ##############
    mat_file = "./Dataset2/"+dataset_name+"_ti_n.mat"
    mat2_file = "./Dataset3/"+dataset_name+".mat"

    # Load dataset
    mat = io.loadmat(mat_file)
    #X = mat['X']
    #y = mat['y']
    X = mat['dtm']
    print(X.shape)

    # Load label
    mat2 = io.loadmat(mat2_file)
    if 'gnd' in mat2:
        y = mat2['gnd']
    elif 'labels' in mat2:
        y = mat2['labels']
    number_of_classes = len(np.unique(y))
    print(y.shape)
    print(number_of_classes)
    
    model = CoclustInfo(n_row_clusters=number_of_classes, n_col_clusters=number_of_classes*3, max_iter=200, n_init=50)
    model.fit(X)
    
    res = model.row_labels_
    
    print(nmi(res, y.ravel()))
    print(ari(res, y.ravel()))
    
    
    
if __name__ == "__main__":
    for dataset_name in datasets:
        run(dataset_name)
