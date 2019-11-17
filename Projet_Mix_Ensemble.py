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



#################################################
# Parameters
#################################################
col_mult = 1

algos_pair = [(CoclustInfo, "CoclustInfo"), (CoclustSpecMod, "CoclustSpecMod"), (CoclustMod, "CoclustMod")]

datasets = ["classic4", "cstr", "ng5", "rcv4"]
#################################################



#################################################
# Functions
#################################################
def make_co_assoc(row_labels):
    co_assoc = np.zeros((row_labels.shape[1], row_labels.shape[1]))
    for i in range(row_labels.shape[0]):
        labels = row_labels[i,]
        #temp = np.array([[int(i == j) for i in labels] for j in labels])
        n_values = np.max(labels) + 1
        temp = np.eye(n_values)[labels]
        temp = np.dot(temp, temp.T)
        co_assoc += temp
    return co_assoc


def run_cluster_ensembles(row_labels, number_of_classes, y):
    res = CE.cluster_ensembles(cluster_runs=row_labels, N_clusters_max=number_of_classes)
    
    #nmis.append(nmi(res, y.ravel()))
    #aris.append(ari(res, y.ravel()))
    
    return res

    
def run_coclust_co_assoc(row_labels, number_of_classes, n_col_clusters, y, algo, algo_name):
    co_assoc = make_co_assoc(row_labels)
        
    if algo_name == "CoclustInfo":
        model = algo(n_row_clusters=number_of_classes, n_col_clusters=n_col_clusters, n_init=20, max_iter=100)
    else:
        model = algo(n_clusters=number_of_classes, n_init=20, max_iter=100)    
    model.fit(co_assoc)

    #nmis.append(nmi(model.row_labels_, y.ravel()))
    #aris.append(ari(model.row_labels_, y.ravel()))

    return model.row_labels_    
#################################################



#################################################
# RUN
#################################################
def run(dataset_name, algo_pair):
    algo = algo_pair[0]
    algo_name = algo_pair[1]
    nmis = []
    aris = []
    
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
    
    n_col_clusters = number_of_classes * col_mult

    ##############
    # 1
    ##############
    print("#################################################")
    print("# 1 - Run "+algo_name+"...")
    print("#################################################")

    criterions = []
    t_row_labels = []
    
    epochs = 10 if algo == CoclustSpecMod else 200
    n_init = 20 if algo == CoclustSpecMod else 1

    for i in range(epochs):
        if i % int(epochs/10) == 0:
            print(i)
        if algo == CoclustInfo:
            model = algo(n_row_clusters=number_of_classes, n_col_clusters=n_col_clusters, max_iter=100, n_init=n_init)
        else:
            model = algo(n_clusters=number_of_classes, max_iter=100, n_init=n_init)    
        model.fit(X)
        #print(model.criterion)
        if "criterion" in model.__dict__:
            criterions.append(model.criterion)
        elif "modularity" in model.__dict__:
            criterions.append(model.modularity)
        else:
            criterions.append(random())
        t_row_labels.append(model.row_labels_)

    row_labels = np.array(t_row_labels)
    order = list(np.flip(np.argsort(criterions))[:10])

    row_labels = row_labels[order]
    print(row_labels.shape)

    ##############
    # 2
    ##############
    print("#################################################")
    print("# 2 - Run cluster ensembles...")
    print("#################################################")

    res = run_cluster_ensembles(row_labels, number_of_classes, y)
    nmis.append(nmi(res, y.ravel()))
    aris.append(ari(res, y.ravel()))

    ##############
    # 3
    ##############
    print("#################################################")
    print("# 3 - Run "+algo_name+" on co-association matrix...")
    print("#################################################")

    res = run_coclust_co_assoc(row_labels, number_of_classes, n_col_clusters, y, algo, algo_name)
    nmis.append(nmi(res, y.ravel()))
    aris.append(ari(res, y.ravel()))    

    ##############
    # 4
    ##############
    print("#################################################")
    print("# 4 - Run spherical k-means...")
    print("#################################################")
    model = SphericalKmeans(n_clusters=number_of_classes, n_init=20, max_iter=100)
    model.fit(X)

    #row_labels = np.concatenate([row_labels, np.matrix(model.row_labels_)], axis=0)
    temp = list(row_labels)
    temp.append(model.row_labels_)
    row_labels = np.array(temp)

    res = run_cluster_ensembles(row_labels, number_of_classes, y)
    nmis.append(nmi(res, y.ravel()))
    aris.append(ari(res, y.ravel()))
        
    res = run_coclust_co_assoc(row_labels, number_of_classes, n_col_clusters, y, algo, algo_name)
    nmis.append(nmi(res, y.ravel()))
    aris.append(ari(res, y.ravel()))    

    ##############
    # Results
    ##############
    print("#################################################")
    print("# Results")
    print("#################################################")

    print(nmis)
    print(aris)
    return nmis, aris
#################################################



datasets = ["cstr"]

for dataset in datasets:
    print("#################################################")
    print("# DATASET: "+dataset)
    print("#################################################")    
    
    for algo_pair in algos_pair:
        file_name = dataset+"_"+algo_pair[1]+"_"+str(col_mult)
        if os.path.exists(file_name):
            continue
            
        print("#################################################")
        print("# ALGO: "+algo_pair[1])
        print("#################################################")  
        
        best_nmis, best_aris = [0], [0]  
        for _ in range(5):
            nmis, aris = run(dataset, algo_pair)
            if np.mean([np.mean(nmis), np.mean(aris)]) > np.mean([np.mean(best_nmis), np.mean(best_aris)]):
                best_nmis, best_aris = nmis, aris
                
        myfile = open(file_name, "a")
        content = ', '.join(str(t) for t in best_nmis) + "\n" + ', '.join(str(t) for t in best_aris)
        myfile.write(content)
        myfile.close()
