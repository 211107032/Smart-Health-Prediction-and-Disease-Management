import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

# Load the saved model
model = tf.keras.models.load_model("skin_disease_model.h5")

# Load class indices
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)
reverse_class_indices = {v: k for k, v in class_indices.items()}

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Function to predict the class of an input image
def predict_image(img_path, model, reverse_class_indices):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = reverse_class_indices[predicted_class_index]
    confidence = np.max(predictions)

    return predicted_class, confidence

# Path to the test image
sample_image_path = "./data/Vascular lesion/ISIC_0025578.jpg"  # Replace with the path to the image you want to test

# Predict
predicted_class, confidence = predict_image(sample_image_path, model, reverse_class_indices)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
