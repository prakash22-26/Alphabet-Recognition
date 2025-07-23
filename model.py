import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 64
NUM_CLASSES = 26
DATASET_PATH = './archive/alphabet-dataset/Train'

def load_dataset(dataset_path):
    data, labels = [], []
    label_map = sorted(os.listdir(dataset_path))  # folder names A-Z
    for label, folder in enumerate(label_map):
        folder_path = os.path.join(dataset_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = np.expand_dims(img, axis=-1)  # shape: (64, 64, 1)
            data.append(img)
            labels.append(label)
    return np.array(data) / 255.0, np.array(labels)

# Load and split data
X, y = load_dataset(DATASET_PATH)
y = to_categorical(y, NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augmentation
datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(X_train)

# Input shape for grayscale
input_tensor = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = Conv2D(3, (3, 3), padding='same')(input_tensor)  # convert grayscale to 3-channel (for VGG)

# Base VGG19
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
vgg_output = base_model(x)
for layer in base_model.layers[:-4]:  # fine-tune last 4 conv layers
    layer.trainable = False

# Custom head
x = Flatten()(vgg_output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Full model
model = Model(inputs=input_tensor, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=13,
                    validation_data=(X_test, y_test))

# Plot accuracy/loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend(); plt.title('Accuracy'); plt.xlabel('Epoch')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title('Loss'); plt.xlabel('Epoch')
plt.tight_layout(); plt.show()

# Evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

# Save
model.save("model_v2.keras")
