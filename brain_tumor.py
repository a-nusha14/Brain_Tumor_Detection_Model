import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Reshape,
    LSTM, Dense, Dropout, Input
)


dataset_path = r"C:\Users\Anusha Saha\OneDrive\Desktop\Brain_tumor_detection\brain_tumor_dataset\Training"
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
img_size = 64

data, labels = [], []

for label, tumor in enumerate(classes):
    folder = os.path.join(dataset_path, tumor)
    for img_name in os.listdir(folder):
        if not img_name.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        img = cv2.imread(os.path.join(folder, img_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        data.append(img)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)


X = data.reshape(-1, 64, 64, 1)
y = to_categorical(labels, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


input_layer = Input(shape=(64, 64, 1))

x = Conv2D(32, (3,3), activation='relu', padding='same', name='conv1')(input_layer)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same', name='conv2')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same', name='conv3')(x)  # Grad-CAM layer
x = MaxPooling2D((2,2))(x)

x = Reshape((64, 128))(x)

x = LSTM(64)(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1
)


loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(range(4), classes, rotation=45)
plt.yticks(range(4), classes)

for i in range(4):
    for j in range(4):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=classes))


y_test_bin = label_binarize(y_true, classes=[0,1,2,3])

plt.figure(figsize=(7,6))
for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.2f})")

plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC–AUC")
plt.legend()
plt.show()


plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.show()


def grad_cam(model, img_array, class_index, layer_name='conv3'):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


img = X_test[0]
img_input = img.reshape(1, 64, 64, 1)

pred = model.predict(img_input)
class_idx = np.argmax(pred)

heatmap = grad_cam(model, img_input, class_idx)


heatmap = cv2.resize(heatmap, (64, 64))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

original_img = np.uint8(img.squeeze() * 255)
original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
plt.title("Original MRI")
plt.imshow(original_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Tumor Localized")
plt.imshow(overlay)
plt.axis('off')

plt.show()


