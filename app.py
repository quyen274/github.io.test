import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Đọc dữ liệu
@st.cache
def load_data():
    data = pd.read_csv('_Dữ_liệu_giao_dịch_ngày__202406152152.csv')
    data['Ngày'] = pd.to_datetime(data['Ngày'])
    data.drop_duplicates(inplace=True)
    data.drop(['Ghép 01', 'Ghép 02', 'Ngành nghề', 'Sàn'], axis=1, inplace=True)
    features = ['Mở cửa', 'Đóng cửa', 'Cao nhất', 'Thấp nhất', 'Trung bình', 'GD khớp lệnh KL']
    data = data[['Ngày', 'Mã CK'] + features]
    data = data.sort_values(by=['Mã CK', 'Ngày'])
    return data, features

data, features = load_data()

def prepare_data(data, features):
    data_scaled = data.copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled[features] = scaler.fit_transform(data[features])
    return data_scaled, scaler

def create_sequences(data, seq_length, feature_col):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        Y.append(data[i + seq_length][feature_col])
    return np.array(X), np.array(Y)

def train_lstm(X_train, Y_train, seq_length, epochs=100):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    epoch_range = tqdm(range(epochs), desc="Epochs")
    for epoch in epoch_range:
        model.fit(X_train, Y_train, epochs=5, batch_size=32, verbose=0)
        epoch_range.set_postfix({'loss': model.history.history['loss'][0]})

    return model

def evaluate_performance(model, test_data, scaler, seq_length):
    X_test, Y_test = create_sequences(test_data, seq_length, features.index('Đóng cửa'))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

    test_predict = model.predict(X_test)
    
    test_predict_transformed = np.zeros((test_predict.shape[0], len(features)))
    test_predict_transformed[:, features.index('Đóng cửa')] = test_predict.flatten()
    test_predict_transformed = scaler.inverse_transform(test_predict_transformed)[:, features.index('Đóng cửa')]

    Y_test_transformed = np.zeros((Y_test.shape[0], len(features)))
    Y_test_transformed[:, features.index('Đóng cửa')] = Y_test.flatten()
    Y_test_transformed = scaler.inverse_transform(Y_test_transformed)[:, features.index('Đóng cửa')]

    rmse = np.sqrt(mean_squared_error(Y_test_transformed, test_predict_transformed))
    mae = mean_absolute_error(Y_test_transformed, test_predict_transformed)
    mape = np.mean(np.abs((Y_test_transformed - test_predict_transformed) / Y_test_transformed)) * 100
    r2 = r2_score(Y_test_transformed, test_predict_transformed)

    return rmse, mae, mape, r2

def visualize_results(data, scaler, model, stock_symbol, seq_length):
    X, Y = create_sequences(data, seq_length, features.index('Đóng cửa'))
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    train_predict = model.predict(X)
    
    train_predict_transformed = np.zeros((train_predict.shape[0], len(features)))
    train_predict_transformed[:, features.index('Đóng cửa')] = train_predict.flatten()
    train_predict_transformed = scaler.inverse_transform(train_predict_transformed)[:, features.index('Đóng cửa')]

    actual_transformed = np.zeros((Y.shape[0], len(features)))
    actual_transformed[:, features.index('Đóng cửa')] = Y.flatten()
    actual_transformed = scaler.inverse_transform(actual_transformed)[:, features.index('Đóng cửa')]

    plt.figure(figsize=(10, 6))
    plt.plot(actual_transformed, label=f'Actual ({stock_symbol})', color='blue', alpha=0.6, linewidth=2)
    plt.plot(train_predict_transformed, label=f'Predicted ({stock_symbol})', color='orange', alpha=0.8, linewidth=2)
    plt.title(f'Predicted vs Actual Stock Prices ({stock_symbol})', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt.gcf())
    plt.clf()

def main():
    st.title('Stock Price Prediction with LSTM')
    st.write('This app predicts stock prices and displays top 10 best performing stocks.')

    seq_length = st.slider('Sequence Length', 10, 60, 20)
    epochs = st.slider('Epochs', 10, 200, 100)

    symbols = data['Mã CK'].unique()

    results = []

    for symbol in symbols:
        stock_data = data[data['Mã CK'] == symbol].copy()
        stock_data_scaled, scaler = prepare_data(stock_data, features)

        if len(stock_data_scaled) < seq_length:
            st.write(f"Skipping {symbol} due to insufficient data")
            continue

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_y_test = []
        all_y_pred = []

        for train_index, test_index in kf.split(stock_data_scaled):
            train_data, test_data = stock_data_scaled.iloc[train_index], stock_data_scaled.iloc[test_index]

            if len(train_data) < seq_length or len(test_data) < seq_length:
                st.write(f"Skipping split for {symbol} due to insufficient sequence data")
                continue

            X_train, Y_train = create_sequences(train_data[features].values, seq_length, features.index('Đóng cửa'))
            X_test, Y_test = create_sequences(test_data[features].values, seq_length, features.index('Đóng cửa'))

            lstm_model = train_lstm(X_train, Y_train, seq_length, epochs)
            rmse, mae, mape, r2 = evaluate_performance(lstm_model, test_data[features].values, scaler, seq_length)

            y_pred = lstm_model.predict(X_test)
            temp = np.zeros((y_pred.shape[0], len(features)))
            temp[:, features.index('Đóng cửa')] = y_pred.flatten()
            y_pred_transformed = scaler.inverse_transform(temp)[:, features.index('Đóng cửa')]
            Y_test_transformed = np.zeros((Y_test.shape[0], len(features)))
            Y_test_transformed[:, features.index('Đóng cửa')] = Y_test.flatten()
            Y_test_transformed = scaler.inverse_transform(Y_test_transformed)[:, features.index('Đóng cửa')]

            all_y_test.extend(Y_test_transformed)
            all_y_pred.extend(y_pred_transformed)

        if len(all_y_test) > 0 and len(all_y_pred) > 0:
            all_y_test = np.array(all_y_test)
            all_y_pred = np.array(all_y_pred)

            mse = mean_squared_error(all_y_test, all_y_pred)
            results.append((symbol, mse))

            visualize_results(stock_data_scaled[features].values, scaler, lstm_model, symbol, seq_length)

    results.sort(key=lambda x: x[1])
    top_3_symbols = results[:3]

    st.write('Top 10 best performing stocks:')
    for symbol, mse in top_3_symbols:
        st.write(f"{symbol}: MSE = {mse}")

if __name__ == "__main__":
    main()
