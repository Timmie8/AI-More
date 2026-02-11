import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Swing Intelligence", layout="wide")

# --- UI ---
st.title("⚡ AI Swingtrader: XGBoost Intelligence")
ticker = st.sidebar.text_input("USA Ticker", "TSLA")
threshold = st.sidebar.slider("Target Profit (%)", 1.0, 5.0, 2.5) / 100

@st.cache_data
def load_smart_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # --- FEATURE ENGINEERING (De 'geheime saus') ---
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                                  -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['Vol_Change'] = df['Volume'].pct_change()
    df['Price_Speed'] = df['Close'].pct_change(periods=3) # Momentum over 3 dagen
    df['MA_Dist'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
    
    # TARGET: Is de prijs over 5 dagen > huidige prijs + threshold?
    df['Target'] = (df['Close'].shift(-5) > df['Close'] * (1 + threshold)).astype(int)
    
    return df.dropna()

try:
    df = load_smart_data(ticker)
    
    # --- MODEL TRAINING ---
    features = ['RSI', 'Vol_Change', 'Price_Speed', 'MA_Dist']
    X = df[features]
    y = df['Target']
    
    # We trainen op alles behalve de laatste 5 dagen (omdat we daar de uitkomst nog niet van weten)
    X_train = X[:-5]
    y_train = y[:-5]
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # --- VOORSPELLING VOOR VANDAAG ---
    current_state = X.tail(1)
    prob = model.predict_proba(current_state)[0][1] # Kans op winst
    
    # --- DASHBOARD ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-60:], y=df['Close'].tail(60), name="Prijs"))
        fig.update_layout(template="plotly_dark", title=f"Trend Analyse {ticker}")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("AI Verdict")
        score = round(prob * 100, 2)
        st.metric("Kans op Target", f"{score}%")
        
        if score > 65:
            st.success("🔥 STERK SIGNAAL: De AI herkent een winstgevend patroon voor de komende 5 dagen.")
        elif score > 45:
            st.warning("⚖️ NEUTRAAL: Geen sterke overtuiging. Wacht op betere condities.")
        else:
            st.error("📉 NEGATIEF: De kans op een stijging is statistisch klein.")

    # Feature Importance (Laat zien WAAROM de AI dit denkt)
    st.write("### Waar kijkt de AI naar?")
    importance = pd.DataFrame({'Feature': features, 'Weight': model.feature_importances_})
    st.bar_chart(importance.set_index('Feature'))

except Exception as e:
    st.info(f"Selecteer een USA ticker. (Fout: {e})")

