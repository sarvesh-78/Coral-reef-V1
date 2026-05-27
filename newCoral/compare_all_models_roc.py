import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

test_dir = "dataset/test"
IMG_SIZE = (224, 224)

# =========================
# Custom CNN + MobileNetV2
# (trained with rescale=1./255)
# =========================
datagen_rescale = ImageDataGenerator(rescale=1./255)

test_gen_rescale = datagen_rescale.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

y_true = test_gen_rescale.classes

# =========================
# ResNet50
# (trained with preprocess_input)
# =========================
datagen_resnet = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_gen_resnet = datagen_resnet.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# =========================
# Load Models
# =========================
custom_model = tf.keras.models.load_model("custom_cnn_model.h5")
mobilenet_model = tf.keras.models.load_model("mobilenet_binary_model.h5")
resnet_model = tf.keras.models.load_model("resnet50_model.h5")

# =========================
# Predictions
# =========================
test_gen_rescale.reset()
custom_probs = custom_model.predict(test_gen_rescale)

test_gen_rescale.reset()
mobilenet_probs = mobilenet_model.predict(test_gen_rescale)

test_gen_resnet.reset()
resnet_probs = resnet_model.predict(test_gen_resnet)

# =========================
# ROC
# =========================
fpr_custom, tpr_custom, _ = roc_curve(y_true, custom_probs)
fpr_mobilenet, tpr_mobilenet, _ = roc_curve(y_true, mobilenet_probs)
fpr_resnet, tpr_resnet, _ = roc_curve(y_true, resnet_probs)

auc_custom = auc(fpr_custom, tpr_custom)
auc_mobilenet = auc(fpr_mobilenet, tpr_mobilenet)
auc_resnet = auc(fpr_resnet, tpr_resnet)

# =========================
# Plot
# =========================
plt.figure(figsize=(8,6))

plt.plot(fpr_custom, tpr_custom, label=f'Custom CNN (AUC = {auc_custom:.3f})')
plt.plot(fpr_mobilenet, tpr_mobilenet, label=f'MobileNetV2 (AUC = {auc_mobilenet:.3f})')
plt.plot(fpr_resnet, tpr_resnet, label=f'ResNet50 (AUC = {auc_resnet:.3f})')

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Comparative ROC Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=300)
plt.show()

print("\nAUC Scores:")
print("Custom CNN:", auc_custom)
print("MobileNetV2:", auc_mobilenet)
print("ResNet50:", auc_resnet)
