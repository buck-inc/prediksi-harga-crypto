# app.py
import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Harga Crypto - Tokocrypto", layout="centered")
st.title("📈 Prediksi Harga Crypto (Tokocrypto)")

# Form input API key
st.sidebar.header("🔐 API Key Binance")
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret", type="password")

symbol = st.sidebar.text_input("Symbol", value="BTCUSDT")
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d"])
limit = st.sidebar.slider("Jumlah Data (candles)", 100, 1000, 500)

if st.sidebar.button("🚀 Ambil Data"):
    try:
        client = Client(api_key, api_secret)
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        st.subheader("📊 Data Harga")
        st.dataframe(df.tail(10))

        # Fitur dan Target
        X = df[['open', 'high', 'low', 'volume']]
        y = df['close']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        st.subheader("📈 Hasil Prediksi")
        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}, index=y_test.index)
        st.line_chart(pred_df)

    except Exception as e:
        st.error(f"❌ Gagal ambil data: {e}")
