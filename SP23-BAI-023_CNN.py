import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import random
import matplotlib.pyplot as plt
import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# Normalize pixel values from [0,255] to [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

idx = random.randint(0, X_train.shape[0])
plt.imshow(X_train[idx])

model = Sequential([
    Conv2D(24, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(48, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)


history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

plt.plot(history.history['loss'], label='Model Training Loss')
plt.plot(history.history['val_loss'], label='Model Validation Loss')
plt.ylim(0, 1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)  
y_test_labels = np.argmax(y_test, axis=1)  

# Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred)
cd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
cd.plot()

# Calculate metrics
accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred, average='macro')
recall = recall_score(y_test_labels, y_pred, average='macro')
f1 = f1_score(y_test_labels, y_pred, average='macro')

# Bar chart for metrics
plt.bar(["Accuracy", "Precision", "Recall", "F1-Score"], [accuracy, precision, recall, f1], color=['red', 'blue', 'yellow', 'green'])
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
plt.show()

# Print results
print('Confusion Matrix:')
print(cm)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1-Score: ', f1)

