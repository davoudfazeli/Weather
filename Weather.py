import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Read the CSV file and drop missing values
data_train = pd.read_csv("weatherHistory.csv").dropna()

# Extracting date information
data_train['Date_temp'] = pd.to_datetime(data_train['Formatted Date'].str[:10], format='%Y-%m-%d')

# Creating 'Date' and 'day_of_year' columns
data_train['Date'] = data_train['Date_temp'].dt.strftime('%Y%m%d')
data_train['day_of_year'] = data_train['Date_temp'].dt.dayofyear

# Grouping by date and day of the year, and calculating the mean temperature
data = data_train.groupby(['Date', 'day_of_year'])['Temperature (C)'].mean().reset_index()
mean_temp_day_of_year = pd.concat([data['Temperature (C)'], data['day_of_year']], axis=1)

x_total = np.array(data['day_of_year']).reshape(-1, 1)
y_total = np.array(data['Temperature (C)']).reshape(-1, 1)

# Normalize input data
scaler_x = StandardScaler()
x_total_scaled = scaler_x.fit_transform(x_total)

x_train, x_test, y_train, y_test = train_test_split(x_total_scaled, y_total, random_state=50, test_size=0.2)

model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

output = model.fit(x_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping], verbose=0)

plt.plot(output.history['loss'], label='Training Loss')
plt.plot(output.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", rmse)

