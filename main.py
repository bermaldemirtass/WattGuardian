import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM

# Veriyi yükle
df = pd.read_csv("energydata_complete.csv")

# Sadece sayısal verileri al (LSTM için)
df_scaled = df.drop(columns=["date"])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)

print("Veri yüklendi ve normalize edildi.")

### ------------------ ANOMALY DETECTION ------------------ ###
X_anom = df_scaled[["Appliances"]].values
X_train, X_test = train_test_split(X_anom, test_size=0.2, shuffle=False)

input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train,
                epochs=20,
                batch_size=64,
                validation_data=(X_test, X_test),
                shuffle=False)

reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

plt.figure(figsize=(10, 5))
plt.plot(mse, label="Reconstruction Error")
plt.hlines(threshold, xmin=0, xmax=len(mse), colors="red", label="Threshold")
plt.title("Anomaly Detection with Autoencoder")
plt.xlabel("Time")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("anomaly_detection_plot.png")
plt.close()

### ------------------ FORECASTING (LSTM) ------------------ ###
target_column = "Appliances"
X = df_scaled.drop(columns=[target_column])
y = df_scaled[target_column]

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_seq, y_seq = create_sequences(X, y, time_steps)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title("Energy Consumption Forecasting with LSTM")
plt.xlabel("Time Step")
plt.ylabel("Appliance Energy Usage")
plt.legend()
plt.tight_layout()
plt.savefig("forecasting_plot.png")
plt.close()

# Anomalileri çıkar ve kaydet
if 'is_anomaly' in df.columns:
    anomalies = df[df['is_anomaly'] == True]
    anomalies.to_csv("anomalies.csv", index=False)
    print("Anomaliler anomalies.csv olarak kaydedildi ✅")
else:
    print("❌ is_anomaly kolonu bulunamadı.")

import pandas as pd

# Veri setini oku
df = pd.read_csv("energydata_complete.csv", parse_dates=["date"])

# Appliances tüketiminde z-score kullanarak anomali tespiti
mean = df['Appliances'].mean()
std = df['Appliances'].std()

# Z-score yöntemi ile threshold belirleyelim
threshold = 3  # istersen 2.5 yapabilirsin
df['z_score'] = (df['Appliances'] - mean) / std
df['is_anomaly'] = df['z_score'].abs() > threshold

# Sadece anomalileri al
anomalies = df[df['is_anomaly'] == True]

# anomalies.csv olarak kaydet
anomalies.to_csv("anomalies.csv", index=False)
print(f"✅ {len(anomalies)} adet anomali anomalies.csv olarak kaydedildi.")

