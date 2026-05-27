import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================
# PARAMETERS
# =========================
test_dir = "dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 1

# =========================
# LOAD TRUE LABELS (common)
# =========================

base_test_gen = ImageDataGenerator(rescale=1./255)

test_gen_base = base_test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = test_gen_base.classes

# =========================
# LOAD MODELS
# =========================
custom_model = tf.keras.models.load_model("custom_cnn_model.h5")
mobilenet_model = tf.keras.models.load_model("mobilenet_binary_model.h5")
resnet_model = tf.keras.models.load_model("resnet50_model.h5")

# =========================
# FUNCTION TO EVALUATE MODEL
# =========================
def evaluate_model(model, generator, model_name):
    generator.reset()
    probs = model.predict(generator)
    preds = (probs > 0.5).astype(int).flatten()

    report = classification_report(y_true, preds, output_dict=True)

    fpr, tpr, _ = roc_curve(y_true, probs)
    model_auc = auc(fpr, tpr)

    print(f"\n{model_name}")
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall:    {report['1']['recall']:.4f}")
    print(f"F1-score:  {report['1']['f1-score']:.4f}")
    print(f"AUC:       {model_auc:.4f}")
    print("-" * 40)

    return fpr, tpr, model_auc

# =========================
# GENERATORS PER MODEL
# =========================

# Custom CNN (rescale 1/255)
custom_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# MobileNetV2 (rescale 1/255)
mobilenet_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# ResNet50 (preprocess_input)
resnet_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# =========================
# EVALUATE
# =========================

fpr_custom, tpr_custom, auc_custom = evaluate_model(custom_model, custom_gen, "Custom CNN")
fpr_mobile, tpr_mobile, auc_mobile = evaluate_model(mobilenet_model, mobilenet_gen, "MobileNetV2")
fpr_resnet, tpr_resnet, auc_resnet = evaluate_model(resnet_model, resnet_gen, "ResNet50")

# =========================
# COMPARATIVE ROC PLOT
# =========================

plt.figure(figsize=(8,6))
plt.plot(fpr_custom, tpr_custom, label=f'Custom CNN (AUC = {auc_custom:.3f})')
plt.plot(fpr_mobile, tpr_mobile, label=f'MobileNetV2 (AUC = {auc_mobile:.3f})')
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

print("\nFinal AUC Scores:")
print("Custom CNN:", auc_custom)
print("MobileNetV2:", auc_mobile)
print("ResNet50:", auc_resnet)
