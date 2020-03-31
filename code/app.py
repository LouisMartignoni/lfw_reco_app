# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:09:49 2020

@author: VCZL048
"""

import streamlit as st
from keras_facenet import *
import pickle
import matplotlib.pyplot as plt
    
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

emb = []
new_emb = []
y = []
all_paths = []

st.title('Face ID')
path_facenet = r'C:\Users\vczl048\keras_facenet\facenet_keras.h5'


encoder_filename = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\encoder.sav'
out_encoder = pickle.load(open(encoder_filename, 'rb'))

filename = file_selector(r'C:\Users\vczl048\Desktop\images_facenet_test')
option = st.sidebar.selectbox('What classification method do you want to use?',
                     ('Support Vector Classification', 'Cosine distance',
                      'Euclidean distance'))
c1 = st.sidebar.button('Confirmer')
print('valeur c11', c1)
if c1:
    st.write('You selected `%s`' % filename)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    if (option == 'Support Vector Classification'):
        model_filename = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\svc.sav'
        model = pickle.load(open(model_filename, 'rb'))

        im, predict_name, class_probab = score_image(filename, resnet, model, out_encoder)
        print('predict_name', predict_name)
        title = '%s (%.3f)' % (predict_name, class_probab)
        st.image(im.squeeze(0).permute(1, 2, 0).int().numpy(), caption=title)
        
    elif(option == 'Cosine distance'):
        data = np.load(r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\embeddings.npz')
        emb, y = data['arr_0'], data['arr_1']
        
        im, dist, predict_name = cosine_class(filename, resnet, out_encoder, emb, y)
        
        title = '%s (%.3f)' % (predict_name, dist)
        st.image(im.squeeze(0).permute(1, 2, 0).int().numpy(), caption=title)
        
    else:
        data = np.load(r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\embeddings.npz')
        emb, y, all_paths = data['arr_0'], data['arr_1'], data['arr_2']
        print('all_emb', emb)
        im, dist, predict_name, new_emb = euclidean_class(filename, resnet, out_encoder, emb, y, all_paths)
        
        title = 'Image testée'
        st.image(im[0], caption=title)
        
        for i, x in enumerate(im[1:]):
            print(i)
            title = '%s (%.3f)' % (predict_name[i], dist[i])
            st.image(x, caption=title)
        print('valeur c1', c1)
c2 = st.sidebar.text_input('Entrez le nom de la personne que vous voulez sauvegarder')
c3 = st.sidebar.button('Sauvegarde')
if c3:
    print('bouton sauve')
    print('new_emb', new_emb)
    saveImage(c2, new_emb, filename, emb, all_paths, y)
                    
