# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("AAMC.csv")
df.head()

# %%
data = df['Close'].to_numpy().reshape(-1, 1)

# %%
data.dtype

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# %%
data_scaled

# %%
def create_seq(data, time_step = 60):
    X,y = [],[]
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    
    return np.array(X), np.array(y)

# %%
X, y = create_seq(data_scaled, 60)

# %%
X.shape

# %%
y.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(40, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(40),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# %%
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %%
history = model.fit(X_train, y_train, epochs=25)

# %%
y_pred = model.predict(X_test)

# %%
y_pred = scaler.inverse_transform(y_pred)

# %%
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))

plt.plot(y_test_actual, label='Actual Price')
plt.plot(y_pred, label='Predicted Price')

plt.title("Stock Price Prediction using LSTM")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# %%


# %%



