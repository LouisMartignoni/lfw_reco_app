# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:23:29 2020

@author: VCZL048
"""

# example of loading the keras facenet model
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import numpy as np
import pandas as pd
from PIL import *
from matplotlib import pyplot
from os import listdir, scandir
from os.path import isdir
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import *
from random import choice
import os
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cosine, euclidean
import streamlit as st
from facenet_pytorch import MTCNN as py_mtcnn, InceptionResnetV1
import re
# load the model
# summarize input and output shape


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    
    
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    # extract the face
    face = pixels[y1:y2, x1:x2]
    
    
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def load_faces(p):
    mtcnn_torch = py_mtcnn(post_process=False, image_size=160, thresholds=[0.5, 0.7, 0.7])
    
    img = Image.open(p)
    face = mtcnn_torch(img)
    #name = n
    if face is not None:
        print('load_faces')
        return face.unsqueeze(0)
        #names.append(name)
        
    return None


def load_faces2(data):
    faces = []
    names = []
    mtcnn_torch = py_mtcnn(post_process=False, image_size=160, thresholds=[0.5, 0.7, 0.7])
    
    for p, n in zip(data.image_path, data.name):
        img = Image.open(p)
        face = mtcnn_torch(img)
        name = n
        if face is not None:
            faces.append(face.unsqueeze(0))
            names.append(name)
            print('une tete ajoutée')
    return faces, names


def load_dataset2(faces):
    labels = [subdir for _ in range(len(faces))]
    print('>loaded %d examples for class: %s' % (len(faces), subdir))
    X.extend(faces)
    y.extend(labels)
    return np.asarray(X), np.asarray(y)



# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in [ f.name for f in scandir(directory)]:
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)



# get the face embedding for one face
def get_embedding(model, face_pixels):
    
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std   
    emb = model(face_pixels)
    
    return emb

# convert each face in the train set to an embedding
def get_embeddings(model, data):
    res = list()
    for face_pixels in data:
        embedding = get_embedding(model, face_pixels)
        res.append(embedding.detach().numpy()[0])
    res = np.asarray(res)
    return res

def load_images(lfw_allnames):
    os.chdir('lfw')
    lfw_allnames = pd.read_csv(lfw_allnames)

    # Créer une table pour créer train/test
    image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
    image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
    image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
    image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
    image_paths = image_paths.drop("images",1)
    
    path_louis = r'Louis\louis_train.jpg'
    name_louis = 'LouisLeBoss'
    
    dataframe = pd.DataFrame([[name_louis, path_louis]], columns = ['name', 'image_path'])
    
    # Créer Train/Test
    lfw_train, lfw_test = train_test_split(image_paths, test_size=0.006)
    lfw_train, lfw_test = train_test_split(lfw_test, test_size=0.3)
    lfw_train = lfw_train.reset_index().drop("index",1)
    lfw_test = lfw_test.reset_index().drop("index",1)
    
    lfw_train = lfw_train.append(dataframe, ignore_index=True)
    
    # verify that there is a mix of seen and unseen individuals in the test set
    print(len(set(lfw_train.name).intersection(set(lfw_test.name))))
    print(len(set(lfw_test.name) - set(lfw_train.name)))
    
    
    
    os.chdir('lfw-deepfunneled')
    
    # Detection des visages, obtention de nos matrices 128x128x3 de pixels  
    trainX, trainy = load_faces2(lfw_train)    
    testX, testy = load_faces2(lfw_test) 
    
    #np.savez_compressed('lfw.npz', trainX, trainy, testX, testy)
    
    
    #Face embedding(representation of features in a vector)
    #data = np.load('lfw.npz')
    
    
    
    #Face embedding(representation of features in a vector)
    #trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    return trainX, trainy, testX, testy

def embeddings(path_facenet, trainX, trainy, testX, testy):
    print('load facenet')    
    # load the facenet model
    model_facenet = load_model(path_facenet)
    print('embeddings')
    # Standardization et embedding
    newTrainX = get_embeddings(model_facenet, trainX)
    newTestX = get_embeddings(model_facenet, testX)
    # save arrays to one file in compressed format
    np.savez_compressed('lfw.npz', newTrainX, trainy, newTestX, testy)
    return newTrainX, trainy, newTestX, testy, model_facenet
    
    
def modeling(trainX, trainy, testX, testy):   
    #Classification
    # load dataset
   # data = np.load('lfw.npz')
   # trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    
    
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    
    
    # label encode targets
    labels = np.concatenate((trainy, testy), axis=None)
    out_encoder = LabelEncoder()
    out_encoder.fit(labels)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    
    print('SVM')
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    return model, out_encoder


def score_image(path, model_facenet, model, out_encoder, name='unknown'):
    image_dataframe = pd.DataFrame([[name, path]], columns = ['name', 'image_path'])
    imageX, imagey = load_faces2(image_dataframe)
    
    newImageX = get_embeddings(model_facenet, imageX)
    
    
    # normalize input vectors
   # in_encoder = Normalizer(norm='l2')
   # imageX_norm = in_encoder.transform(newImageX)
    
    # predict
    image_class = model.predict(newImageX)
    image_prob = model.predict_proba(newImageX)
    # SVC predict_proba utilise la méthode de Platt qui ne fonctionne pas avec peu de donées,
    # ceci explique que la classe attribuée ne correspond pas au max de la proba
    
    prob_per_class_dic = dict(zip(model.classes_, image_prob[0]))
    class_probability = prob_per_class_dic.get(image_class[0]) *100
    print('TEST', class_probability)
    # get name
    class_index = image_class[0]
    print('class_index', class_index)
    print('class_image', image_class)
    print('class_image', image_prob)
   # class_probability = image_prob[0,class_index] * 100
    print("classe:", image_class)
    predict_names = out_encoder.inverse_transform(image_class)
    
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    return imageX[0], predict_names[0], class_probability


def cosine_class(path, model_facenet, out_encoder, emb, y, name='unknown'):
    image_dataframe = pd.DataFrame([[name, path]], columns = ['name', 'image_path'])
    imageX, imagey = load_faces2(image_dataframe)
    
    newImageX = get_embeddings(model_facenet, imageX)
    
    # normalize input vectors
    #in_encoder = Normalizer(norm='l2')
    #imageX_norm = in_encoder.transform(newImageX)
    i = 0
    j = 0
    min=1000
    latest_iteration = st.empty()
    bar = st.progress(0)
    for vec in emb:
        dist = cosine(newImageX, vec)
        if dist<min:
            i=j
            min=dist
        j = j+1
        latest_iteration.text(f'{j/len(emb)*100}%')
        bar.progress(j/len(emb))
    predict_names = out_encoder.inverse_transform([y[i]])
    return imageX[0], dist, predict_names



def euclidean_class(path, model_facenet, out_encoder, emb, y, all_paths, name='unknown'):
    image_dataframe = pd.DataFrame([[name, path]], columns = ['name', 'image_path'])
    #print(image_dataframe)
    imageX, imageY = load_faces2(image_dataframe)
    print('image', imageX)
    newImageX = get_embeddings(model_facenet, imageX)
    
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    imageX_norm = in_encoder.transform(newImageX)
    i1 = 0
    i2 = 0
    i3 = 0
    j = 0
    d1=1000
    d2=1000
    d3=1000
    faces = []
    predict_names = []
    dists = []
    dist2 = []
    latest_iteration = st.empty()
    bar = st.progress(0)
    for vec in emb:
        dist = euclidean(imageX_norm, vec)
        dist2.append(dist)
        if dist<d1:
            d3=d2
            i3=i2
            d2=d1
            i2=i1
            i1=j
            d1=dist
        elif dist<d2:
            d3=d2
            i3=i2
            d2=dist
            i2=j
        elif dist<d3:
            d3=dist
            i3=j
        j = j+1
        latest_iteration.text(f'{j/len(emb)*100}%')
        bar.progress(j/len(emb))
    print('nb iterations:',j)
    print('min:', min(dist2))
    faces.append(Image.open(path))
    os.chdir(r'C:/Users/vczl048/keras_facenet/lfw/lfw-deepfunneled')
    for k in [i1, i2, i3]:
        #predict_names.append(out_encoder.inverse_transform([y[k]]))
        predict_names.append([y[k]])
        faces.append(Image.open(all_paths[k]))
    dists.extend([d1, d2, d3])
    
    return faces, dists, predict_names, newImageX


def saveImage(new_name, new_emb, file_jpg, all_emb, all_path, all_y):
    new_im = Image.open(file_jpg)
    if not os.path.exists(r"C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\\" + new_name):
        os.makedirs(r"C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\\" + new_name)
    n = len([name for name in os.listdir(r"C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\\" + new_name)])
    new_path = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\\' + new_name + '\\' + new_name +'_' + str(n+1) + '.jpg'
    print('all path type', type(all_path))
    all_emb = np.concatenate((all_emb, new_emb), axis=0)
    all_path = all_path.tolist()
    all_path.append(new_path)
    print('new name', new_name)
    print('len y', len(all_y.tolist()))
    all_y = all_y.tolist()
    all_y.append(new_name)
    print('len y2', len(all_y))
    print('nb emb:', len(all_emb))
    print('new emb:', new_emb)
    np.savez_compressed(r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\embs.npz', all_emb, all_y, all_path)
    new_im.save(new_path)
    st.write('Image sauvegardée!')
    return

def valid(data, threshold):
    indices = list()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for d in data:
        
        dist = euclidean()
    
    return TP, FP, FN, TN