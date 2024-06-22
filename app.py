import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Định nghĩa hàm load_data và load_model
def load_data():
    data = pd.read_csv('_Dữ_liệu_giao_dịch_ngày__202406152152.csv')
    data['profit'] = data['Đóng cửa'] - data['Mở cửa']
    features = ['Mở cửa', 'Đóng cửa', 'Cao nhất', 'Thấp nhất', 'Trung bình', 'GD khớp lệnh KL']
    data = data.set_index('Ngày')
    data = data.sort_index()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    return scaled_data, data, features

def load_model():
    with open('best_lstm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Tạo chuỗi thời gian cho dự đoán
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Định nghĩa hàm recommend_stocks
def recommend_stocks(budget):
    scaled_data, data, features = load_data()
    model = load_model()
    
    seq_length = 60
    X_pred, _ = create_sequences(scaled_data, seq_length)
    
    predicted_profit = model.predict(X_pred)
    data['predicted_profit'] = np.nan
    data['predicted_profit'][seq_length:] = predicted_profit[:, 0]
    
    top_stocks = data.sort_values(by='predicted_profit', ascending=False).head(10)
    
    return top_stocks[['Mã CK', 'predicted_profit']]

# Xây dựng giao diện Streamlit
st.title("Stock Recommendation System")
budget = st.number_input("Enter your budget:", min_value=0, step=1000)
if st.button("Get Recommendations"):
    top_stocks = recommend_stocks(budget)
    st.write("Top recommended stocks for your budget:")
    st.write(top_stocks)
