import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv("energydata_complete.csv")
data = df[["Appliances"]].values

# Normalize et
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print("Veri yüklendi ve normalize edildi.")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Train-test ayır
X_train, X_test = train_test_split(data_scaled, test_size=0.2, shuffle=False)

# Autoencoder mimarisi
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Modeli eğit
history = autoencoder.fit(X_train, X_train,
                          epochs=20,
                          batch_size=64,
                          validation_data=(X_test, X_test),
                          shuffle=False)

print("Autoencoder eğitimi tamamlandı.")

# Yeniden yapılandırma hatalarını hesapla
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Eşik değeri belirle (ortalama + 2 std)
threshold = np.mean(mse) + 2 * np.std(mse)

# Anomalileri işaretle
anomalies = mse > threshold

# Sonuçları görselleştir
plt.figure(figsize=(10, 5))
plt.plot(mse, label="Reconstruction Error")
plt.hlines(threshold, xmin=0, xmax=len(mse), colors="red", label="Threshold")
plt.title("Anomaly Detection with Autoencoder")
plt.xlabel("Time")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("anomaly_detection_plot.png")
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Tüketimi tahmin edeceğimiz hedef sütun
target_column = "Appliances"

# Giriş verisi ve hedefi ayır
X = df_scaled.drop(target_column, axis=1)
y = df_scaled[target_column]

# Zaman serisi formatına getir
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_seq, y_seq = create_sequences(X, y, time_steps)

# Eğitim ve test ayırımı
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# LSTM modeli
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Tahmin ve karşılaştırma
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
plt.savefig("anomaly_plot.png")

plt.show()

