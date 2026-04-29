# %%
import numpy as np
import pandas as pd

# %%
X = np.array([[0,0], [0, 1], [1, 0], [1,1]])

# %%
w_h = np.array([[1, -1], [-1, 1]])
b_h = np.array([-1, -1])

# %%
w_o = np.array([1, 1])
b_o = -1

# %%
def step(x):
    return 1 if x>=0 else 0

# %%
y_pred = []
for xi in X:
    h1 = step(np.dot(xi, w_h[0]) + b_h[0])
    h2 = step(np.dot(xi, w_h[1]) + b_h[1])

    y_pred.append(step(h1*w_o[0] + h2*w_o[1] + b_o))

print(y_pred)

# %%
w_h = np.array([[1, 1], [-1, -1]])
b_h = np.array([-2, 0.5])

# %%
y_pred = []
for xi in X:
    h1 = step(np.dot(xi, w_h[0]) + b_h[0])
    h2 = step(np.dot(xi, w_h[1]) + b_h[1])

    y_pred.append(step(h1*w_o[0] + h2*w_o[1] + b_o))

print(y_pred)

# %%



