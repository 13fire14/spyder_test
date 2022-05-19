

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config('人脸识别系统')
st.title(":heart:人脸识别系统")
#%% load model
model=tf.keras.models.load_model('D:/model_save/model_cnn1_app.h5')
#%%上传图片
f=st.file_uploader('上传文件')
img=np.array(Image.open(f),dtype=np.uint8)
#st.write(img)
#st.image(img)
#%% 人脸检测
haar=cv2.CascadeClassifier('C:/Users/bianca/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
faces=haar.detectMultiScale(img)
for x,y,w,h in faces:
    face=img[y:y+h,x:x+w,:]
    #resize
    face2=cv2.resize(face,(32,32))
    #灰度图
    gray=cv2.cvtColor(face2,cv2.COLOR_RGB2GRAY)
    #model predict
    prob=model.predict(gray.reshape(1,32,32,1))
    #result
    st.write('女性' if prob>=0.5 else '男性')
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0))
    cv2.putText(img,'female' if prob>=0.5 else 'male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    st.image(img)
