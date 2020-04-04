# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:09:49 2020

@author: VCZL048
"""

import streamlit as st
from keras_facenet import *

def main():

    emb = []
    new_emb = []
    y = []
    all_paths = []

    st.title('Face ID')

    file_jpeg = st.sidebar.file_uploader("Déposez l'image", type=['png', 'jpg', 'jpeg'])

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    save = st.sidebar.checkbox('Voulez vous sauvegarder l''image?')
    c2 = st.sidebar.text_input('Entrez le nom de la personne que vous voulez sauvegarder')
    up = st.sidebar.button('Confirmer', key='dragndrop')

    if up:
        #data = np.load(r'lfw\lfw-deepfunneled\embeddings.npz')
        data = np.load(r'lfw\lfw-deepfunneled\embs.npz')
        emb, y, all_paths = data['arr_0'], data['arr_1'], data['arr_2']
        #emb = np.delete(emb, -1, 0)
        #y = np.delete(y, -1, 0)
        #all_paths = np.delete(all_paths, -1, 0)
        
        #np.savez_compressed(r'lfw/lfw-deepfunneled/embs.npz', emb, y, all_paths)

        im, dist, predict_name, new_emb = euclidean_class(file_jpeg, resnet, emb, y, all_paths)
        title = 'Image testée'
        st.image(im[0], caption=title, width=300)
        t = []
        for i, x in enumerate(im[1:]):
            t.append( '%s (%.3f)' % (predict_name[i], dist[i]))
        st.image(im[1:], caption=t, width=160)
        if save:
            if c2 == "":
                st.write("Veuillez renseigner le nom de la personne")
            else:
                saveImage(c2, new_emb, file_jpeg, emb, all_paths, y)

if __name__ == '__main__':
    main()