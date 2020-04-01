# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:00:17 2020

@author: VCZL048
"""

from keras_facenet import *
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()


data = np.concatenate((trainX, testX), axis=0)
data_label = np.concatenate((trainy, testy), axis=0)
model_clustering_7 = DBSCAN(metric='euclidean', min_samples=1, eps=0.7)
model_clustering_7.fit(data)
model_clustering_6 = DBSCAN(metric='euclidean', min_samples=1, eps=0.6)
model_clustering_6.fit(data)
model_clustering_8 = DBSCAN(metric='euclidean', min_samples=1, eps=0.8)
model_clustering_8.fit(data)
model_clustering_85 = DBSCAN(metric='euclidean', min_samples=1, eps=0.85)
model_clustering_85.fit(data)
model_clustering_865 = DBSCAN(metric='euclidean', min_samples=1, eps=0.865)
model_clustering_865.fit(data)
model_clustering_885 = DBSCAN(metric='euclidean', min_samples=1, eps=0.895)
model_clustering_885.fit(data)
model_clustering_9 = DBSCAN(metric='euclidean', min_samples=1, eps=0.9)
model_clustering_9.fit(data)

# Clacul des metriques
score_6 = adjusted_rand_score(data_label, model_clustering_6.labels_)
score_7 = adjusted_rand_score(data_label, model_clustering_7.labels_)
score_8 = adjusted_rand_score(data_label, model_clustering_8.labels_)
score_85 = adjusted_rand_score(data_label, model_clustering_85.labels_)
score_865 = adjusted_rand_score(data_label, model_clustering_865.labels_)
score_885 = adjusted_rand_score(data_label, model_clustering_885.labels_)
score_9 = adjusted_rand_score(data_label, model_clustering_9.labels_)


print('score 0,6: ', score_6)
print('score 0,7: ', score_7)
print('score 0,8: ', score_8)
print('score 0,85: ', score_85)
print('score 0,865: ', score_865)
print('score 0,885: ', score_885)
print('score 0,9: ', score_9)


out_encoder.inverse_transform(np.array([34]))
data_label
model_clustering.labels_


u, indices = np.unique(model_clustering_7.labels_, return_counts=True)
u[indices > 1]

# Déterminer epsilon; ici trop peu d'observation je pense
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1
