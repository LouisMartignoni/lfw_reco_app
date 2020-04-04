# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:09:49 2020

@author: VCZL048
"""

import streamlit as st
from keras_facenet import *
#import pickle
#import matplotlib.pyplot as plt

def main():
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

    #filename = file_selector(r'C:\Users\vczl048\Desktop\images_facenet_test')
    #option = st.sidebar.selectbox('What classification method do you want to use?',
    #                     ('Support Vector Classification', 'Cosine distance',
    #                      'Euclidean distance'))
    #c1 = st.sidebar.button('Confirmer')

    file_jpeg = st.sidebar.file_uploader("Déposez l'image", type=['png', 'jpg', 'jpeg'])

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    save = st.sidebar.checkbox('Voulez vous sauvegarder l''image?')
    c2 = st.sidebar.text_input('Entrez le nom de la personne que vous voulez sauvegarder')
    up = st.sidebar.button('Confirmer', key='dragndrop')

    if up:
        #data = np.load(r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\embeddings.npz')
        data = np.load(r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\embs.npz')
        emb, y, all_paths = data['arr_0'], data['arr_1'], data['arr_2']
        im, dist, predict_name, new_emb = euclidean_class(file_jpeg, resnet, out_encoder, emb, y, all_paths)
        print('new emb done', new_emb)
        title = 'Image testée'
        st.image(im[0], caption=title, width=300)
        t = []
        for i, x in enumerate(im[1:]):
            t.append( '%s (%.3f)' % (predict_name[i], dist[i]))
        st.image(im[1:], caption=t, width=160)
        if save:
            print('new_emb', new_emb)
            saveImage(c2, new_emb, file_jpeg, emb, all_paths, y)

if __name__ == '__main__':
    main()