# %%
import tensorflow as tf
import numpy as np
import pandas as pd

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset2/train',
    image_size=(128, 128),
    batch_size=32
)

class_names = train_ds.class_names

test_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset2/test',
    image_size=(128, 128),
    batch_size=32
)

# %%
normalize = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x,y : (normalize(x), y))
test_ds = test_ds.map(lambda x,y : (normalize(x), y))

# %%
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    weights="imagenet",
    include_top=False
)

# %%
base_model.trainable = False

# %%
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
history = model.fit(train_ds, epochs=10, validation_data=test_ds)

# %%



