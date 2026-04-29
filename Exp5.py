# %%
import numpy as np
import pandas as pd
import tensorflow as tf

# %%
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# %%
X_train = X_train/255
X_test = X_test/255

X_train = tf.image.resize(X_train, (96, 96))
X_test = tf.image.resize(X_test, (96, 96))

# %%
base_model = tf.keras.applications.ResNet50(
    input_shape=(96, 96, 3),
    weights="imagenet",
    include_top=False
)

# %%
base_model.trainable = False

# %%
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(10, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# %%
model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

# %%



