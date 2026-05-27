import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenet_binary_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load image
img_path = "bleached.png"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Set input
interpreter.set_tensor(input_details[0]['index'], img_array)

# Run
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])

print("TFLite output:", output[0][0])