from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import io

# Initialize Flask app
app = Flask(__name__)

# Load the model and class indices
model = tf.keras.models.load_model("skin_disease_model.h5")
class_indices = joblib.load("class_indices.pkl")
IMG_HEIGHT, IMG_WIDTH = 224, 224


# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'image' not in request.files:
            return render_template("index.html", message="No image uploaded")

        # Read the uploaded image
        file = request.files['image']

        # Convert the uploaded file to a BytesIO object and load it using Keras
        img = image.load_img(io.BytesIO(file.read()), target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = list(class_indices.keys())[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return render_template("index.html", predicted_class=predicted_class, confidence=confidence)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
