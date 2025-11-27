# evaluate_model.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# 🔹 Class labels (make sure it matches your folder structure)
class_names = ["Bleached_Mild", "Bleached_Moderate", "Bleached_Severe", "Dead", "Healthy"]

# 🔸 Load your trained model (use the fine-tuned one if you have)
model = load_model("coral_model_finetuned.h5")  # or coral_model.h5 if not fine-tuned

# 🪄 Validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    "dataset/val",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # important for correct labels
)

# 🧠 Run predictions
preds = model.predict(val_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

# 🧾 Classification report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# 🧮 Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Coral Bleaching Classification — Confusion Matrix')
plt.tight_layout()
plt.show()
