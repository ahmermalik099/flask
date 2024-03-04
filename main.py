from flask import Flask, jsonify
import os

from flask import Flask, request, jsonify
#import tensorflow
from flask_cors import CORS
from keras.models import load_model  # TensorFlow is required for Keras to work
# from PIL import Image, ImageOps
import numpy as np
import os
import cv2

#from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow all origins by default (for development)


current_directory = os.path.dirname(os.path.abspath(__file__))

# Combine the directory with the filename
model_path = os.path.join(current_directory, "keras_model.h5")

# Load the model
model = load_model(model_path, compile=False)


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
#model = load_model("/keras_model.h5", compile=False)

# Load the labels
label_path= os.path.join(current_directory, "labels.txt")
class_names = open(label_path, "r").readlines()



@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

@app.route('/dummy', methods=['GET'])
def get_dummy_data():
    image = cv2.imread(f"{current_directory}/teeth.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize the image to be at least 224x224 and then cropping from the center
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict using the loaded model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    return jsonify({'class': str(class_name[2:]), 'confidence': str(confidence_score)}), 200


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))