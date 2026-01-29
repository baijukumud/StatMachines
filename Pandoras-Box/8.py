import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input

data = pd.DataFrame(np.sin(np.linspace(0, 100, 1000)), columns=['value'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X, y = [], []
for i in range(len(scaled_data) - 10):
  X.append(scaled_data[i:i+10, 0])
  y.append(scaled_data[i+10, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

inputs = Input(shape=(X_train.shape[1], 1))
x = LSTM(50)(inputs)
outputs = Dense(1)(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

predictions = scaler.inverse_transform(model.predict(X_test))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
plt.plot(y_test, label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
