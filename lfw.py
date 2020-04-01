# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:23:00 2020

@author: VCZL048
"""
import os
os.getcwd()
os.chdir('lfw')
from keras_facenet import *
from sklearn.model_selection import train_test_split
import pickle


# Data read-in and cleaning

lfw_allnames = pd.read_csv("lfw_allnames.csv")
matched_pairsTrain = pd.read_csv('matchpairsDevTrain.csv')
mismatched_pairsTrain = pd.read_csv('mismatchpairsDevTrain.csv')
matched_pairsTest = pd.read_csv('matchpairsDevTest.csv')
mismatched_pairsTest = pd.read_csv('mismatchpairsDevTest.csv')


#Matched pairs
mpairTrain = pd.concat([matched_pairsTrain, matched_pairsTest], ignore_index=True) 
mpairTrain['paths_img1'] = mpairTrain.imagenum1.apply(lambda x: '{0:0>4}'.format(x))
mpairTrain['paths_img2'] = mpairTrain.imagenum2.apply(lambda x: '{0:0>4}'.format(x))
mpairTrain["paths_img1"] = mpairTrain.name + '/'  + mpairTrain.name + '_' + mpairTrain.paths_img1 + ".jpg"
mpairTrain["paths_img2"] = mpairTrain.name + '/' + mpairTrain.name + '_' + mpairTrain.paths_img2 + ".jpg"
mpairTrain['Y'] = 1
 #MisMatched pairs
mismatched_pairsTrain = pd.concat([mismatched_pairsTrain, mismatched_pairsTest], ignore_index=True)
mismatched_pairsTrain['paths_img1'] = mismatched_pairsTrain.imagenum1.apply(lambda x: '{0:0>4}'.format(x))
mismatched_pairsTrain['paths_img2'] = mismatched_pairsTrain.imagenum2.apply(lambda x: '{0:0>4}'.format(x))
mismatched_pairsTrain["paths_img1"] = mismatched_pairsTrain.name + '/'  + mismatched_pairsTrain.name + '_' + mismatched_pairsTrain.paths_img1 + ".jpg"
mismatched_pairsTrain["paths_img2"] = mismatched_pairsTrain['name.1'] + '/' + mismatched_pairsTrain['name.1'] + '_' + mismatched_pairsTrain.paths_img2 + ".jpg"
mismatched_pairsTrain['Y'] = 0

data = mpairTrain[['paths_img1', 'paths_img2', 'Y']].copy()
data = pd.concat([data, mismatched_pairsTrain[['paths_img1', 'paths_img2', 'Y']].copy()])
data['name'] = ''

data_test = data[:3]
l_emb = []
row = data_test[:1]
for i, row in data.iterrows():
    #embeddings
    print(row)
    image_dataframe = pd.DataFrame([[row.name, row.paths_img1]], columns = ['name', 'image_path'])
    image1X, image1y = load_faces2(image_dataframe)
    
    image_dataframe2 = pd.DataFrame([[row.name, row.paths_img2]], columns = ['name', 'image_path'])
    image2X, image2y = load_faces2(image_dataframe2)
        
    newImage1X = get_embeddings(resnet, image1X)
    newImage2X = get_embeddings(resnet, image2X)
        
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    image1X_norm = in_encoder.transform(newImage1X)
    image2X_norm = in_encoder.transform(newImage2X)
    
    d = {
            'emb_img1': image1X_norm,
            'emb_img2': image2X_norm,
            'Y': row.Y
            }
    l_emb.append(d)

df_emb = pd.DataFrame(l_emb)
    
image_dataframe = pd.DataFrame([[data.iloc[696].name, data.iloc[696].paths_img1]], columns = ['name', 'image_path'])
image1X, image1y = load_faces2(image_dataframe)

image_dataframe2 = pd.DataFrame([[data.iloc[696].name, data.iloc[696].paths_img2]], columns = ['name', 'image_path'])
image2X, image2y = load_faces2(image_dataframe2)
    
newImage1X = get_embeddings(resnet, image1X)
newImage2X = get_embeddings(resnet, image2X)
    
# normalize input vectors
in_encoder = Normalizer(norm='l2')
image1X_norm = in_encoder.transform(newImage1X)
image2X_norm = in_encoder.transform(newImage2X)


# Créer une table pour créer train/test
image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
image_paths = image_paths.drop("images",1)

path_louis = r'Louis\louis_train.jpg'
name_louis = 'LouisLeBoss'

dataframe = pd.DataFrame([[name_louis, path_louis]], columns = ['name', 'image_path'])

image_paths = image_paths.append(dataframe, ignore_index=True)
image_paths.reset_index().drop("index", 1)
# Créer Train/Test
lfw_train, lfw_test = train_test_split(image_paths, test_size=0.3)
#lfw_train, lfw_test = train_test_split(lfw_test, test_size=0.3)
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

trainX, trainy = load_faces2(image_paths)


pathsTrain = lfw_train.image_path.tolist()
pathsTest = lfw_test.image_path.tolist()

resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Standardization et embedding
#newTrainX = get_embeddings(model_facenet, trainX)
trainX = get_embeddings(resnet, trainX)
testX = get_embeddings(resnet, testX)


#Classification

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

#Save embeddings for distance computation app
embeddings = np.concatenate((trainX, testX), axis=0)
y = np.concatenate((trainy, testy), axis=0)
all_paths = np.concatenate((pathsTrain, pathsTest), axis=0)

np.savez_compressed('embeddings.npz', embeddings, y, all_paths)

path1 = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\Jacques_Chirac\Jacques_Chirac_0022.jpg'
path3 = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\Jacques_Chirac\Jacques_Chirac_0020.jpg'
path4 = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\Jacques_Chirac\Jacques_Chirac_0021.jpg'
path2 = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\Jacques_Chirac\Jacques_Chirac_0022.jpg'
path5 = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\Johnny_Htu\Johnny_Htu_0001.jpg'
paths = [path2, path3, path4, path5]
y2 = ['J', 'J', 'J', 'H']
dataframe = pd.DataFrame(list(zip(y2, paths)), columns = ['name', 'image_path'])

X1, y1 = load_faces2(dataframe)

X2 = get_embeddings(resnet, X1)
X2 = in_encoder.transform(X2)
im, dist, predict_name, new_emb = euclidean_class(path1, resnet, out_encoder, trainX, trainy, image_paths.image_path.tolist())

data = np.load(r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\embeddings.npz')
emb, y = data['arr_0'], data['arr_1']

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# Save model and encoder 
model_filename = 'svc.sav'
pickle.dump(model, open(model_filename, 'wb'))

encoder_filename = 'encoder.sav'
pickle.dump(out_encoder, open(encoder_filename, 'wb'))


# predict
yhat_train = loaded_model.predict(trainX)
yhat_test = loaded_model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

# Scoring d'une image particulière

path = r'C:\Users\vczl048\keras_facenet\lfw\lfw-deepfunneled\Test\Jacques_Chirac\Jacques_Chirac_0022.jpg'
name = 'Jacques_Chirac'


path_louis = r'C:\Users\vczl048\Desktop\images_facenet_test\louis.jpg'
name_louis = 'LouisLeBoss'

path_monroe = r'Marilyn_Monroe/Marilyn_Monroe_0001.jpg'
img = Image.open(path_monroe)
mtcnn_torch = py_mtcnn(post_process=False, image_size=160, thresholds=[0.5, 0.7, 0.7])
face, prob = mtcnn_torch(img, return_prob=True)
model_facenet, model = train_model("lfw_allnames.csv", r'C:\Users\vczl048\keras_facenet\facenet_keras.h5')

im, predict_name, class_probab = score_image(path_louis, model_facenet, model)

# plot for fun
pyplot.imshow(img)
title = '%s (%.3f)' % (predict_name, class_probab)
pyplot.title(title)
pyplot.show()


from facenet_pytorch import MTCNN, InceptionResnetV1


img = Image.open('Bill_Gates/Bill_Gates_0014.jpg')
mtcnn_torch = py_mtcnn(image_size=160)
face = mtcnn_torch(img)
resnet(face.unsqueeze(0))
faces.append(face)
names.append(name)