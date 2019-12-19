#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Chargement des datasets
def load_datasets():
    datasetA, metaA = arff.loadarff(open('R15.arff','r'))
    datasetB, metaB = arff.loadarff(open('cluto-t8-8k.arff','r'))
    datasetC, metaC = arff.loadarff(open('spiralsquare.arff','r'))
    datasetD, metaD = arff.loadarff(open('elly-2d10c13s.arff','r'))
    datasetE, metaE = arff.loadarff(open('cure-t2-4k.arff','r'))
    datasetF, metaF = arff.loadarff(open('square5.arff','r'))
    datasetG, metaG = arff.loadarff(open('rings.arff','r'))
    datasetH, metaH = arff.loadarff(open('disk-1000n.arff','r'))
    datasetI, metaI = arff.loadarff(open('complex8.arff','r'))
    datasetJ, metaJ = arff.loadarff(open('complex9.arff','r'))
    
    return ([datasetA, datasetB, datasetC, datasetD, datasetE, datasetF, datasetG, datasetH, datasetI, datasetJ],[metaA, metaB, metaC, metaD, metaE, metaF, metaG, metaH, metaI, metaJ])


# In[6]:


data, metadata = load_datasets()


# In[23]:


print(data[1])
print(metadata[0])


# In[7]:


def plot_datasets(datasets, metas):
    for i in range(len(datasets)):
        dataset = datasets[i]
        meta = metas[i]
        try:
            plt.scatter(dataset[meta.names()[0]],
                    dataset[meta.names()[1]],
                    c = dataset[meta.names()[2]],
                    s = 1, cmap = 'nipy_spectral')
        except ValueError:
            plt.scatter(dataset[meta.names()[0]],
                        dataset[meta.names()[1]],
                        c = [int(s) if s.isdigit() else 8 for str in dataset[meta.names()[2]] for s in str.split()],
                        s = 1, cmap = 'nipy_spectral')
        plt.show()
plot_datasets(data,metadata)


# In[3]:


### Kmeans iteratif sur dataset
def kmeans_iteratif(dataset, nb_max=50):
    
    duration = []
    sil_score = []
    db_score = []
    chi_score = []
    klabels = []

    X = [[x,y] for (x,y,c) in dataset]

    for k in range(2,nb_max):
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, init='k-means++').fit(X)
        elapsed = time.time() - start_time
        labels = kmeans.labels_
        klabels.append(labels)
        sil = metrics.silhouette_score(X, labels)
        db = metrics.davies_bouldin_score(X, labels)
        chi = metrics.calinski_harabaz_score(X, labels)
        sil_score.append(sil)
        db_score.append(db)
        chi_score.append(chi)
        duration.append(elapsed)

    best_k_sil = sil_score.index(max(sil_score)) + 2
    best_k_db = db_score.index(min(db_score)) + 2
    best_k_chi = chi_score.index(max(chi_score)) + 2

    plt.plot(range(2,nb_max),sil_score,label = "silhouette coefficient")
    plt.plot(range(2,nb_max),db_score,label = "DB index")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.legend()
    plt.title("Score with silhouette coefficient and DB index")
    plt.show()

    plt.plot(range(2,nb_max),chi_score,label = "Calinski and Harabasz index")
    plt.xlabel('k')
    plt.ylabel('score')
    plt.legend()
    plt.title("Score with Calinski and Harabasz index")
    plt.show()

    plt.plot(range(2,nb_max),duration, label = "Execution time")
    plt.xlabel('k')
    plt.ylabel('time')
    plt.title("Execution time")
    plt.show()
      
    return ((best_k_sil, best_k_db, best_k_chi),klabels)


# In[39]:


for i in range(len(data)):
    dataset = data[i]
    meta = metadata[i]
    print(meta)
    ((k_sil,k_db,k_chi),klabels) = kmeans_iteratif(dataset)
    print("best k with silhouette coefficient: ", k_sil)
    print("best k with DB index: ", k_db)
    print("best k with Calinski and Harabasz index: ", k_chi)
    best_labels_sil = klabels[k_sil-2]
    best_labels_db = klabels[k_db-2]
    best_labels_chi = klabels[k_chi-2]
    plt.scatter(dataset[meta.names()[0]],
                    dataset[meta.names()[1]],
                    c = best_labels_sil,
                    s = 1, cmap = 'nipy_spectral')
    plt.title("Labels with silhouette coefficient")
    plt.show()
    
    plt.scatter(dataset[meta.names()[0]],
                    dataset[meta.names()[1]],
                    c = best_labels_db,
                    s = 1, cmap = 'nipy_spectral')
    plt.title("Labels with DB index")
    plt.show()
    
    plt.scatter(dataset[meta.names()[0]],
                    dataset[meta.names()[1]],
                    c = best_labels_chi,
                    s = 1, cmap = 'nipy_spectral')
    plt.title("Labels with Calinski and Harabasz index")
    plt.show()
    

