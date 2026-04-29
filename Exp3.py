# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# %%
X, y = make_classification(n_samples=1000, n_classes=2, n_features=20)

# %%
X.shape

# %%
y = y.reshape(-1, 1)
y.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(20, )))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# %%
history = model.fit(X_train, y_train, epochs=20)

# %%
model.evaluate(X_test, y_test)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers

model = Sequential()

model.add(Dense(128, activation='relu', input_shape=(20, )))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='sigmoid'))

# %%
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
history = model.fit(X_train, y_train, epochs=20)

# %%
model.evaluate(X_test, y_test)

# %%
import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

# %%



