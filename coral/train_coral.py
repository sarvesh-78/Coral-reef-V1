# train_coral.py (IEEE version with graphs + metrics export)
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

# ===== Paths & Parameters =====
train_dir = "dataset/train"
val_dir = "dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

# ===== Data Augmentation =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

# ===== Load Datasets =====
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ===== Class Weights =====
print("Class indices:", train_generator.class_indices)
num_classes = len(train_generator.class_indices)
counts = Counter(train_generator.classes)
max_count = max(counts.values())
class_weights = {i: max_count / count for i, count in counts.items()}
print("Class weights:", class_weights)

# ===== Callbacks =====
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# ===== Build Model =====
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# ===== Initial Training =====
print("\n🔵 Starting initial training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ===== Save Initial Model =====
model.save("coral_model.h5")

# ===== Convert to TFLite =====
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("coral_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Initial model saved!")

# ================= GRAPH GENERATION =================

# Convert history to dataframe
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("training_metrics.csv", index=False)
print("📄 Saved training metrics to training_metrics.csv")

# --------- 1) Accuracy Graph ----------
plt.figure(figsize=(8,6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=300)
plt.close()
print("📊 Saved accuracy graph as accuracy_plot.png")

# --------- 2) Loss Graph ----------
plt.figure(figsize=(8,6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=300)
plt.close()
print("📉 Saved loss graph as loss_plot.png")

# --------- 3) Print Table for IEEE Paper ----------
print("\n===== IEEE Table: Final Epoch Metrics =====")
print(hist_df.tail(1))  # Shows last row

hist_df.tail(1).to_csv("final_epoch_table.csv", index=False)
print("📄 Saved final metrics table as final_epoch_table.csv")

print("\n🎉 All graphs + tables generated successfully!")
