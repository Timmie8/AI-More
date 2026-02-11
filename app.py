import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Dashboard Setup
st.set_page_config(page_title="AI Swingtrade Dashboard", layout="wide")
st.title("📈 AI Swingtrade Dashboard (USA Market)")
st.sidebar.header("Instellingen")

# Input voor het aandeel
ticker = st.sidebar.text_input("Voer USA Ticker in (bijv. AAPL, TSLA, NVDA)", "AAPL")
period = st.sidebar.selectbox("Periode", ["6mo", "1y", "2y"])

# Data ophalen zonder API key
@st.cache_data
def load_data(symbol, p):
    df = yf.download(symbol, period=p, interval="1d")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns] # Fix voor nieuwe yfinance multi-index
    return df

try:
    data = load_data(ticker, period)
    
    # --- AI METHODE: LINEAIRE REGRESSIE VOOR TREND ---
    # We gebruiken de laatste 30 dagen om een trendlijn te voorspellen
    data['Days'] = np.arange(len(data))
    lookback = 30
    recent_data = data.tail(lookback)
    
    X = recent_data['Days'].values.reshape(-1, 1)
    y = recent_data['Close'].values
    model = LinearRegression().fit(X, y)
    
    # Voorspelling voor de komende 5 dagen (Swingtrade horizon)
    future_days = np.array([len(data) + i for i in range(5)]).reshape(-1, 1)
    future_preds = model.predict(future_days)
    
    # --- INDICATOREN BEREKENEN ---
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    # Layout: Kolommen
    col1, col2 = st.columns([3, 1])

    with col1:
        # Grafiek maken
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Prijs"))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name="20 MA", line=dict(color='orange', width=1)))
        
        # AI Trendlijn tekenen
        future_index = pd.date_range(start=data.index[-1], periods=6, freq='B')[1:]
        fig.add_trace(go.Scatter(x=future_index, y=future_preds, name="AI Voorspelling (5d)", line=dict(color='cyan', dash='dash')))
        
        fig.update_layout(title=f"Swingtrade Analyse: {ticker}", yaxis_title="Prijs (USD)", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("AI Signaal")
        laatste_prijs = data['Close'].iloc[-1]
        voorspelde_prijs = future_preds[-1]
        verschil = ((voorspelde_prijs - laatste_prijs) / laatste_prijs) * 100

        if verschil > 2:
            st.success(f"🚀 BULLISH\nVoorspelde stijging: {verschil:.2f}%")
        elif verschil < -2:
            st.error(f"📉 BEARISH\nVoorspelde daling: {verschil:.2f}%")
        else:
            st.warning("⚖️ NEUTRAAL\nGeen sterke trend")

        st.write("**Statistieken:**")
        st.write(f"Huidige prijs: ${laatste_prijs:.2f}")
        st.write(f"AI Target (5d): ${voorspelde_prijs:.2f}")

except Exception as e:
    st.error(f"Fout bij het ophalen van data. Controleer de ticker. Fout: {e}")
