import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from sklearn.metrics import pairwise_distances
import requests
import os
import urllib
import PIL.Image
import pickle
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from plotly import *
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
#import streamlit as st
#use the below library while displaying the images in jupyter notebook
from IPython.display import display, Image



# def get_similar_products_cnn(fashion_df,product_id, num_results):
#     if(df[df['ProductId']==product_id]['Gender'].values[0]=="Boys"):
#         extracted_features = boys_extracted_features
#         Productids = boys_Productids
#     elif(df[df['ProductId']==product_id]['Gender'].values[0]=="Girls"):
#         extracted_features = girls_extracted_features
#         Productids = girls_Productids
#     elif(df[df['ProductId']==product_id]['Gender'].values[0]=="Men"):
#         extracted_features = men_extracted_features
#         Productids = men_Productids
#     elif(df[df['ProductId']==product_id]['Gender'].values[0]=="Women"):
#         extracted_features = women_extracted_features
#         Productids = women_Productids
#     Productids = list(Productids)
#     doc_id = Productids.index(product_id)
#     pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1,-1))
#     indices = np.argsort(pairwise_dist.flatten())[0:num_results]
#     pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
#     print("="*20, "input product details", "="*20)
#     ip_row = df[['ImageURL','ProductTitle']].loc[df['ProductId']==Productids[indices[0]]]
#     for indx, row in ip_row.iterrows():
#         display(Image(url=row['ImageURL'], width = 224, height = 224,embed=True))
#         print('Product Title: ', row['ProductTitle'])
#     print("\n","="*20, "Recommended products", "="*20)
#     for i in range(1,len(indices)):
#         rows = df[['ImageURL','ProductTitle']].loc[df['ProductId']==Productids[indices[i]]]
#         for indx, row in rows.iterrows():
#             display(Image(url=row['ImageURL'], width = 224, height = 224,embed=True))
#             print('Product Title: ', row['ProductTitle'])
#             print('Euclidean Distance from input image:', pdists[i])



def predict():
    st.title("Content-based Image Retrieval")
    df = pd.read_csv("/dataset/new_data/fashion.csv")
#     st.write(df)
    
    zz=['-']
    zz.extend(df['Gender'].unique().tolist())
    
    gender = st.sidebar.selectbox("Enter the Gender",zz)
    df_gender=df[df['Gender']==gender]
    
    yy=['-']
    yy.extend(df_gender['SubCategory'].unique().tolist())
    subcategory=st.sidebar.selectbox("Enter the sub category",yy)
    
    df_subcategory=df_gender[df_gender['SubCategory']==subcategory]    
    
    product_id=st.selectbox('Enter the product id',df_subcategory['ProductId'].unique().tolist())
#     aa=['-']
#     aa.extend(df_subcategory['ProductType'].unique().tolist())
#     productype=st.sidebar.selectbox("Enter the product type",aa)
    
#     df_productype=df_subcategory[df_subcategory['ProductType']==productype]
    
#    product_id=st.selectbox('Enter the product id',df_productype['ProductId'].unique().tolist())
    
    boys_extracted_features = np.load('/dataset/Boys_ResNet_features.npy')
    boys_Productids = np.load('/dataset/Boys_ResNet_feature_product_ids.npy')
    girls_extracted_features = np.load('/dataset/Girls_ResNet_features.npy')
    girls_Productids = np.load('/dataset/Girls_ResNet_feature_product_ids.npy')
    men_extracted_features = np.load('/dataset/Men_ResNet_features.npy')
    men_Productids = np.load('/dataset/Men_ResNet_feature_product_ids.npy')
    women_extracted_features = np.load('/dataset/Women_ResNet_features.npy')
    women_Productids = np.load('/dataset/Women_ResNet_feature_product_ids.npy')
    df["ProductId"] = df["ProductId"].astype(str)

    def get_similar_products_cnn(product_id, num_results):
        if(df[df['ProductId']==product_id]['Gender'].values[0]=="Boys"):
            extracted_features = boys_extracted_features
            Productids = boys_Productids
        elif(df[df['ProductId']==product_id]['Gender'].values[0]=="Girls"):
            extracted_features = girls_extracted_features
            Productids = girls_Productids
        elif(df[df['ProductId']==product_id]['Gender'].values[0]=="Men"):
            extracted_features = men_extracted_features
            Productids = men_Productids
        elif(df[df['ProductId']==product_id]['Gender'].values[0]=="Women"):
            extracted_features = women_extracted_features
            Productids = women_Productids
        Productids = list(Productids)
        doc_id = Productids.index(product_id)
        pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1,-1))
        #st.write(pairwise_dist)
        indices = np.argsort(pairwise_dist.flatten())[0:num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
        ip_row = df[['ImageURL','ProductTitle']].loc[df['ProductId']==Productids[indices[0]]]
        for indx, row in ip_row.iterrows():
            image = PIL.Image.open(urllib.request.urlopen(row['ImageURL']))
            image = image.resize((224,224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
        st.write(f"""
             #### Top {num_results - 1} Recommended items
             """)
        New_images = []
        for i in range(1,len(indices)):
            rows = df[['ImageURL','ProductTitle']].loc[df['ProductId']==Productids[indices[i]]]
            for indx, row in rows.iterrows():
#                image = Image.open(Image(url=row['ImageURL'], width = 224, height = 224,embed=True))
                image = PIL.Image.open(urllib.request.urlopen(row['ImageURL']))
#                 image = image.resize((224,224))
#                 st.image(image)
                New_images.append(row['ImageURL'])
#                 st.write(f"Product Title: {row['ProductTitle']}")
#                 st.write(f"Euclidean Distance from input image: {pdists[i]}")


        col1,col2 = st.columns([1,1])
        
        with col1:
            image = PIL.Image.open(urllib.request.urlopen(New_images[0]))
            image = image.resize((224,224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
            st.write(f"Euclidean Distance: {pdists[0]}")
        with col2:
            image = PIL.Image.open(urllib.request.urlopen(New_images[1]))
            image = image.resize((224,224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
            st.write(f"Euclidean Distance: {pdists[1]}")
            
        col1,col2 = st.columns([1,1])
        
        with col1:
            image = PIL.Image.open(urllib.request.urlopen(New_images[2]))
            image = image.resize((224,224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
            st.write(f"Euclidean Distance: {pdists[2]}")
            
        with col2:
            image = PIL.Image.open(urllib.request.urlopen(New_images[3]))
            image = image.resize((224,224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
            st.write(f"Euclidean Distance: {pdists[3]}")
        
    user_input2 = 5


    button = st.button('Generate recommendations')
    if button:
        get_similar_products_cnn(str(product_id), int(user_input2))
    
        
if __name__ == "__main__":
    predict()