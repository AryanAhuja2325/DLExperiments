# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("sentiment_analysis.csv")

# %%
X = df['text']
y = (df['sentiment'] == "positive").astype(int)

# %%
X = X.to_list()
y = y.to_list()

# %%
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

# %%
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# %%
vocab = len(tokenizer.word_index) + 1

# %%
len(X)

# %%
len(y)

# %%
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_seq, y, random_state=42, train_size=0.8)

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, padding="post")
X_test = pad_sequences(X_test, padding="post")

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, LSTM

model = Sequential([
    Embedding(input_dim=vocab, output_dim=128),
    LSTM(32),
    Dense(1, activation="sigmoid")
])

# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# %%
X_train = np.array(X_train).astype('int32')
y_train = np.array(y_train).astype('float32')

# %%
X_test = np.array(X_test).astype('int32')
y_test = np.array(y_test).astype('float32')

# %%
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# %%



