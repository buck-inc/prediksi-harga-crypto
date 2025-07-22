import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# Ambil data dari Yahoo Finance
df = yf.download("BTC-USD", start="2023-01-01", end="2024-07-01", interval="1d")

# Penting! Reset index agar kolom 'Date' masuk ke kolom biasa
df.reset_index(inplace=True)

# Cek kolom (debugging)
print(df.columns)

# Temukan high/low lokal
n = 5
df['high_local'] = df['High'][argrelextrema(df['High'].values, np.greater_equal, order=n)[0]]
df['low_local'] = df['Low'][argrelextrema(df['Low'].values, np.less_equal, order=n)[0]]

# Simpan hasil
df.to_csv("hasil_market_structure.csv", index=False)
print("✅ Selesai! Data disimpan sebagai hasil_market_structure.csv")
