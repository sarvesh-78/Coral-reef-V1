# test_tflite.py
import tensorflow as tf
import numpy as np
from PIL import Image

# ✅ Class labels must match folder names during training
class_names = ["Bleached_Mild", "Bleached_Moderate", "Bleached_Severe", "Dead", "Healthy"]

# ✅ Load TFLite model
interpreter = tf.lite.Interpreter(model_path="coral_model_finetuned.tflite")  # 👈 use your latest model
interpreter.allocate_tensors()

# ✅ Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Images to test
test_images = ["test.jpg", "healthy.jpg", "dd.jpg","test2.jpg","test3.png","test4.jpg"]  # 👈 replace with your test images

for img_path in test_images:
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    pred_idx = np.argmax(output_data)
    pred_label = class_names[pred_idx]
    confidence = output_data[0][pred_idx] * 100

    print(f"✅ Image: {img_path}")
    print("Prediction scores:", np.round(output_data[0], 4))
    print(f"Predicted class: {pred_label} ({confidence:.2f}%)\n")
