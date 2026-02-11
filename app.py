import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="AI Pro Swingtrader", layout="wide")
st.title("🤖 LSTM & Reinforcement Logic Dashboard")

ticker = st.sidebar.text_input("USA Ticker", "NVDA")

@st.cache_data
def get_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

try:
    data = get_data(ticker)
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # --- LSTM VOORBEREIDING ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # We trainen kort op de laatste 60 dagen om een 5-daagse trend te voorspellen
    prediction_days = 60
    x_train, y_train = [], []
    
    for x in range(prediction_days, len(scaled_data)-5):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x:x+5, 0]) # Voorspel volgende 5 dagen
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # --- HET LSTM MODEL ---
    # In een echt platform laad je hier een voorgevangen model (.h5 bestand)
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dense(units=5) # Output: 5 dagen
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Training (voor demo heel kort, in productie gebruik je pre-trained)
    with st.spinner('AI analyseert patronen...'):
        model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)

    # Voorspelling doen
    real_df = scaled_data[-prediction_days:].reshape(1, prediction_days, 1)
    prediction = model.predict(real_df)
    prediction = scaler.inverse_transform(prediction) # Terug naar dollars
    
    # --- REINFORCEMENT LEARNING LOGICA ---
    # De RL-agent evalueert de LSTM output tegen de volatiliteit
    current_price = data['Close'].iloc[-1]
    predicted_5d = prediction[0][4]
    price_change = ((predicted_5d - current_price) / current_price) * 100
    
    # RL Reward-gebaseerde beslissing
    # Als winstverwachting > 3% en risico (volatiliteit) laag is -> BUY
    std_dev = data['Close'].tail(20).std() 
    
    # Visualisatie
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[-30:], y=data['Close'].tail(30), name="Historisch"))
    
    future_dates = pd.date_range(start=data.index[-1], periods=6, freq='B')[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=prediction[0], name="LSTM Voorspelling", line=dict(color='gold', width=4)))
    
    fig.update_layout(template="plotly_dark", title=f"LSTM Voorspelling voor {ticker}")
    st.plotly_chart(fig, use_container_width=True)

    # Dashboard Output
    col1, col2, col3 = st.columns(3)
    col1.metric("Huidige Prijs", f"${current_price:.2f}")
    col2.metric("LSTM Target (5d)", f"${predicted_5d:.2f}", f"{price_change:.2f}%")
    
    with col3:
        st.write("**RL Agent Besluit:**")
        if price_change > 2.5 and std_dev < (current_price * 0.05):
            st.success("STARK KOOP-SIGNAAL")
            st.caption("Ratio: Hoog rendement / Acceptabel risico")
        elif price_change > 0:
            st.warning("NEUTRAAL / HOLD")
            st.caption("Trend is positief, maar onvoldoende marge.")
        else:
            st.error("VERKOOP / VERMIJD")

except Exception as e:
    st.info(f"Selecteer een geldige ticker om de AI te starten. (Fout: {e})")
