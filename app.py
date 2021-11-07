# """ IMPORTING ALL THE REQUIRED LIBRARIES"""

# Streamlit
import streamlit as st

# Pillow library for image
from PIL import Image

#
import cv2
# Numpy for converting image into arrays
import numpy as np

# Importing pickle for storing and loading the models
import pickle

# """IMPORTING TENSORFLOW AND SIMILAR LIBRARIES"""
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

# Import NEAREST NEIGHBORS Algorithm from Scikit-Learn
from sklearn.neighbors import NearestNeighbors
# To display progress
from numpy.linalg import norm
# Importing os for file path
import os

# """LOADING MODELS USING PICKLE
#  i) feature_list
#  ii) filenames """
feature_list = np.array(pickle.load(open('embeddings4.pkl','rb')))
filenames = pickle.load(open('filenames34.pkl','rb'))

# USING RESNet50 Model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False



model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])







#DROPDOWN for brands
option = st.sidebar.selectbox(
        'Choose your favourite BRANDS',
     ('WELCOME','ADDIDAS', 'AMERICAN TOURISTER','LAKME','LEVIS','NIVEA','PUMA'))


# """FUNCTION TO EXTRACT FEATURES from images Logo's of brands given """
def feature_extraction(img_path,model):
    # passing the images
    img = image.load_img(img_path, target_size=(224, 224))
    # """Converting into numpy array of images and expanding it and then proprocessing it"""
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    # predicting the model in result variable and flattening
    result = model.predict(preprocessed_img).flatten()
    # normalising the result
    normalized_result = result / norm(result)
    # Returning the normalized result
    return normalized_result

# """"FUNCTION FOR GETTING SIMILAR IMAGES BASED ON LOGO USING NEAREST NEIGHBORS ALGORITHMS"""
def recommend(features,feature_list):

    neighbors = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    # returning indices
    return indices


# FUNCTION FOR NIVEA
def NIVEA():
    # Uploading the image logo (NIVEA LOGO)
    original = Image.open('BrandImages/Nivea_logo.svg (1).png')
    st.image(original)
    st.title(option)
    # reading the NIVEA logo file
    image2 = cv2.imread('BrandImages/NIVEA.png')
    features = feature_extraction(os.path.join('BrandImages/NIVEA.png'), model)

    indices = recommend(features, feature_list)
    # MAKING 3 COLUMNS
    col1, col2, col3 = st.columns(3)


    with col1:
        st.image(filenames[indices[0][1]])
        st.caption('ARJUN RAMPAL')

    with col2:
        st.image(filenames[indices[0][11]])
        st.caption('TEEJAY SIDHU')

# first dropdown of WELCOME
def WELCOME():

    # st.caption('customer adviser')
    # LOADING THE BACKGROUND IMAGE
    st.image('BrandImages/BGround.png')

# FUNCTION FOR PUMA

def PUMA():
    original = Image.open('BrandImages/PUMA.png')
    st.title(option)
    st.image(original)

    image2 = cv2.imread('BrandImages/PUMA.png')
    features = feature_extraction(os.path.join('BrandImages/PUMA.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(filenames[indices[0][1]])
        st.caption('VIRAT KOHLI')

    with col2:
        st.image(filenames[indices[0][0]])
        st.caption('K.L RAHUL')

    with col3:
        st.image(filenames[indices[0][4]])
        st.caption('BANI J')


# FUNCTION LAKME

def LAKME():
    original = Image.open('BrandImages/LAKME.png')
    st.image(original)
    st.header(option)

    image2 = cv2.imread('BrandImages/LAKME.png')
    features = feature_extraction(os.path.join('BrandImages/LAKME.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3, col4, col5 = st.columns(5)





    with col1:
        st.image(filenames[indices[0][1]])
        st.caption('DIA MIRZA')


    with col2:
        st.image(filenames[indices[0][11]])
        st.caption('ARPITA MEHTA')




# AMERICAN TOURISTER function
def AT():
    original = Image.open('BrandImages/AmericanTourister.jpg')
    st.image(original)
    st.header(option)

    image2 = cv2.imread('BrandImages/AmericanTourister.jpg')
    features = feature_extraction(os.path.join('BrandImages/AmericanTourister.jpg'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(filenames[indices[0][0]])
        st.caption('RONALDO')


# ADDIDAS function

def ADDIDAS():
    original = Image.open('BrandImages/Addidas.jpg')
    st.image(original)
    st.title(option)


    image2 = cv2.imread('BrandImages/Addidas.jpg')
    features = feature_extraction(os.path.join('BrandImages/Addidas.jpg'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:
        st.image(filenames[indices[0][5]])
        st.caption('HIMA DAS')

# funtion for LEVI'S

def LEVI():
    original = Image.open('BrandImages/LEVI.png')
    st.image(original)
    st.title("LEVI'S")

    image2 = cv2.imread('BrandImages/LEVI.png')
    features = feature_extraction(os.path.join('BrandImages/LEVI.png'), model)

    indices = recommend(features, feature_list)
    # show
    col1, col2 = st.columns(2)


    with col1:
        st.image(filenames[indices[0][4]])
        st.caption('ZOYA AKHTAR')

    with col2:
        st.image(filenames[indices[0][5]])
        st.caption('HARSHVARDHAN KAPOOR')



#
# """BASED ON THE BRANDS SELECTED BY THE USER THE RESPETIVE FUNCTION IS CALLED
#  for example if user SELECTS NIVEA brand then NIVEA function is called and similarly"""

if option == 'WELCOME':
    WELCOME()
if option == 'NIVEA':
    NIVEA()

if option=='LAKME':
    LAKME()

if option =='PUMA':
    PUMA()

if option =='ADDIDAS':
    ADDIDAS()




if option=='AMERICAN TOURISTER':
    AT()


if option=='LEVIS':
    LEVI()











