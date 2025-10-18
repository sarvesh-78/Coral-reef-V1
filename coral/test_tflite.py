import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    "C:/Users/91998/Desktop/coral/dataset/train",
    target_size=(224,224),
    batch_size=32
)

print(train_generator.class_indices)
# Class labels
class_names = ["Bleached", "Dead","Healthy"]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="coral_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess image
img = Image.open("test.jpg").convert("RGB")
img = img.resize((224, 224))  # match model input size
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get predicted class index and label
pred_idx = np.argmax(output_data)
pred_label = class_names[pred_idx]

print("✅ Real image inference successful!")
print("Output:", output_data)
print("Predicted class:", pred_label)
