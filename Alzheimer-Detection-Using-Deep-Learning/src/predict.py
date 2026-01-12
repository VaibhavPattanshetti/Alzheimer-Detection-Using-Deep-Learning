import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import tensorflow as tf

def predict_image(image_path, model_path):
    model = tf.keras.models.load_model(model_path)

    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prob = model.predict(img)[0][0]
    label = "Alzheimer" if prob >= 0.5 else "Normal"

    return label, prob
