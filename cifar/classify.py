from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

import streamlit as st

@st.cache_resource()
def get_model():
        model = load_model('CNN_cifar.h5', compile=False)
        print('Model Loaded')
        return model


def predict_class(image):
        loaded_model = get_model()
        image = load_img(image, target_size=(32, 32))
        image = img_to_array(image)
        image = image/255.0
        image = np.reshape(image,[1,32,32,3])

        return np.argmax(loaded_model.predict(image), axis=-1)

        