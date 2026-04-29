# %%
import numpy as np
import pandas as pd
import tensorflow as tf

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128,128),
    batch_size=32
)
class_names = train_ds.class_names
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128,128),
    batch_size=32
)

# %%
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# %%
class_names

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Rescaling

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2,2)),


    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),


    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),


    Flatten(),


    Dense(128, activation='relu'),
    Dropout(0.5),


    Dense(1, activation='sigmoid')
])


# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

# %%



