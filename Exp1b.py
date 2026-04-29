# %%
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# %%
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)

df['target'] = (data.target != 0).astype(int)

print(df.head())

# %%
X = df.drop('target', axis=1).to_numpy()
y = df['target'].to_numpy()

# %%
y = y.reshape(-1, 1)
y.shape

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

# %%
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

# %%
w_h = np.random.rand(X.shape[1], 6)
b_h = np.random.rand(6)

w_o = np.random.rand(6, 1)
b_o = np.random.rand(1)

# %%
w_h

# %%
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sig_der(x):
    return x * (1-x)

# %%
epochs = 1000
lr = 0.01

for _ in range(epochs):
    z_hidden = X_train @ w_h + b_h
    z = sigmoid(z_hidden)

    z_output = z @ w_o + b_o
    y_pred = sigmoid(z_output)

    error = y_train-y_pred

    del_output = error * sig_der(y_pred)
    del_hidden = (del_output @ w_o.T) * sig_der(z)

    # print(del_hidden.shape)
    w_h -= lr * (X_train.T @ del_hidden)
    b_h -= lr * del_hidden.sum(axis=0)

    w_o -= lr * (z.T @ del_output)
    b_o -= lr * (del_output.sum(axis=0))

# %%
y_pred = []

for x in X_test:
    z_hidden = x @ w_h + b_h
    z = sigmoid(z_hidden)

    z_output = z @ w_o + b_o
    y_hat = sigmoid(z_output)

    y_pred.append(1 if y_hat >= 0.5 else 0)

# %%
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# %%



