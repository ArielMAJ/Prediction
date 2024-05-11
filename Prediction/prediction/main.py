#pip install scikit-learn
#pip install tensorflow keras pandas numpy matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU

# Carregando a base de dados
df = pd.read_csv('D:\\Donwloads\\all_stocks_5yr.csv')

# Filtrar os dados para a ação escolhida
df = df[df['Name'] == 'CBG']

# Selecionar as colunas relevantes para previsão
data = df[['date','open','high','low', 'close','volume']].copy()

# Convertendo o formato da coluna date
data['date'] = pd.to_datetime(data['date'])
#fazer para ordenar os dados dos mais velhos aos mais velhos
# Ordenar os dados pela data
data.sort_values('date', inplace=True)
# Definir a data como índice
data.set_index('date', inplace=True)

# Normalizando os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

train_size = int(len(scaled_data) * 0.80)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i:(i + time_steps), 0]
        X.append(a)
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)

time_steps = 30  # ajustar conforme necessário
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

# RNN Simples
model_rnn = Sequential([
    SimpleRNN(50, input_shape=(time_steps, 1)),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
history_rnn =model_rnn.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

# LSTM
model_lstm = Sequential([
    LSTM(50, input_shape=(time_steps, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
history_lstm =model_lstm.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)

# GRU
model_gru = Sequential([
    GRU(50, input_shape=(time_steps, 1)),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mean_squared_error')
history_gru =model_gru.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=0)


def plot_training_time(history_rnn, history_lstm, history_gru):
    plt.figure(figsize=(10, 6))
    plt.plot(history_rnn.history['loss'], label='RNN')
    plt.plot(history_lstm.history['loss'], label='LSTM')
    plt.plot(history_gru.history['loss'], label='GRU')
    plt.title('Comparação do Tempo de Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.show()

plot_training_time(history_rnn, history_lstm, history_gru)
# Avaliar o tempo de treinamento
print("Tempo de treinamento RNN Simples:", model_rnn.history.history['loss'])
print("Tempo de treinamento LSTM:", model_lstm.history.history['loss'])
print("Tempo de treinamento GRU:", model_gru.history.history['loss'])

def plot_predictions(model, X_test, Y_test,title):
    predictions = model.predict(X_test)
    plt.title(title)
    plt.plot(Y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.show()

plot_predictions(model_rnn, X_test, Y_test,"Model RNN")

plot_predictions(model_lstm, X_test, Y_test,"Model LSTM")

plot_predictions(model_gru, X_test, Y_test,"Model GRU")

def plot_comparison(model_rnn, model_lstm, model_gru, X_test, Y_test):
    predictions_rnn = model_rnn.predict(X_test)
    predictions_lstm = model_lstm.predict(X_test)
    predictions_gru = model_gru.predict(X_test)

    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label='Actual', color='black')
    plt.title("Comparação de todos os modelos com o valor real")
    plt.plot(predictions_rnn, label='RNN Predicted', color='blue', linestyle='dashed')
    plt.plot(predictions_lstm, label='LSTM Predicted', color='green', linestyle='dotted')
    plt.plot(predictions_gru, label='GRU Predicted', color='red', linestyle='dashdot')

    plt.legend()
    plt.show()

plot_comparison(model_rnn, model_lstm, model_gru, X_test, Y_test)

