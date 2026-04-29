# %%
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml

X,y = fetch_openml('mnist_784', version=1, return_X_y=True)

# %%
X = X.to_numpy().astype(int) / 255
y = y.to_numpy().astype(int)

# %%
y = y.reshape(-1, 1)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dense(96, activation='relu'))
model.add(Dense(10, activation='softmax'))

# %%
model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='sparse_categorical_crossentropy'
)

# %%
history = model.fit(X_train, y_train, epochs=10, validation_data=[X_test, y_test])

# %%
model.evaluate(X_test, y_test)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# %%



