import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("mobilenet_binary_model.h5")

# Path to test image
img_path = "bleached.png"   # <-- change this

# Load and preprocess
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)[0][0]

print("Raw output:", pred)

if pred > 0.5:
    print("Prediction: Healthy")
else:
    print("Prediction: Bleached")