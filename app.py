import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# --- PAGINA CONFIGURATIE ---
st.set_page_config(page_title="Pro Trading Scanner", layout="wide", initial_sidebar_state="expanded")

# --- CSS VOOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNCTIE: DATA OPHALEN ---
@st.cache_data(ttl=300)
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty:
            return None
        # Fix voor yfinance MultiIndex kolommen
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Fout bij ophalen data: {e}")
        return None

# --- FUNCTIE: ANALYSE LOGICA ---
def perform_full_analysis(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    
    # 1. BEREKEN INDICATOREN (Bulk)
    # Bollinger Bands
    bb50 = ta.bbands(c, length=50, std=2); bb20 = ta.bbands(c, length=20, std=1)
    bb10 = ta.bbands(c, length=10, std=1); bb5 = ta.bbands(c, length=5, std=1)
    
    # Moving Averages
    sma5, sma10, sma20, sma50 = ta.sma(c, 5), ta.sma(c, 10), ta.sma(c, 20), ta.sma(c, 50)
    ema5, ema10, ema20 = ta.ema(c, 5), ta.ema(c, 10), ta.ema(c, 20)
    wma5, wma50 = ta.wma(c, 5), ta.wma(c, 50)
    dema10, dema20, dema50, dema100 = ta.dema(c, 10), ta.dema(c, 20), ta.dema(c, 50), ta.dema(c, 100)
    tema5, tema10, tema20 = ta.tema(c, 5), ta.tema(c, 10), ta.tema(c, 20)
    kama5, kama10, kama20 = ta.kama(c, 5), ta.kama(c, 10), ta.kama(c, 20)
    
    # Oscillators
    macd12 = ta.macd(c, 12, 26, 9); macd10 = ta.macd(c, 10, 20, 5)
    stoch_rsi = ta.stochrsi(c, length=14)
    mfi5 = ta.mfi(h, l, c, v, length=5); mfi7 = ta.mfi(h, l, c, v, length=7)
    cci20 = ta.cci(h, l, c, 20); cci40 = ta.cci(h, l, c, 40)
    aroon = ta.aroon(h, l, 10); adx9 = ta.adx(h, l, c, 9)
    sar = ta.sar(h, l)
    roc10 = ta.roc(c, 10); rsi = ta.rsi(c, 14)

    results = []

    # --- BULLISH RULES ---
    bull_defs = [
        ("MACD (12,26,9) Signal Cross", 95.0, macd12.iloc[-1, 0] > macd12.iloc[-1, 2]),
        ("KAMA (20) Price Cross", 93.3, c.iloc[-1] > kama20.iloc[-1]),
        ("Stoch RSI D Cross 50", 90.5, stoch_rsi.iloc[-1, 1] > 50),
        ("DEMA (20) Price Cross", 89.3, c.iloc[-1] > dema20.iloc[-1]),
        ("MACD (10,20,5) Signal Cross", 88.9, macd10.iloc[-1, 0] > macd10.iloc[-1, 2]),
        ("Aroon Oscillator > 0", 88.5, aroon.iloc[-1, 2] > 0),
        ("BB (50,2,2) Breakout", 88.2, c.iloc[-1] > bb50.iloc[-1, 2]),
        ("BB (10,1,1) Breakout", 87.0, c.iloc[-1] > bb10.iloc[-1, 2]),
        ("WMA (5) Crossover", 86.0, c.iloc[-1] > wma5.iloc[-1]),
        ("SMA (5) Crossover", 82.9, c.iloc[-1] > sma5.iloc[-1]),
        ("DEMA (10) Price Cross", 82.9, c.iloc[-1] > dema10.iloc[-1]),
        ("ADX (9) > 20 Strength", 80.4, adx9.iloc[-1, 0] > 20),
        ("EMA (10) Crossover", 76.7, c.iloc[-1] > ema10.iloc[-1]),
        ("SMA (50) Crossover", 70.0, c.iloc[-1] > sma50.iloc[-1]),
        ("SAR Parabolic Buy", 82.6, sar.iloc[-1] < c.iloc[-1])
    ]

    # --- BEARISH RULES ---
    bear_defs = [
        ("BB (50,2,2) Reversal", 88.2, c.iloc[-1] < bb50.iloc[-1, 0]),
        ("MFI (5) 80/20 Cross", 81.0, mfi5.iloc[-1] > 80),
        ("CCI (40) 100/-100 Cross", 81.0, cci40.iloc[-1] > 100),
        ("ADX (21) Reversal", 79.4, adx9.iloc[-1, 0] < 20),
        ("BB (20,1,1) Reversal", 76.5, c.iloc[-1] < bb20.iloc[-1, 0]),
        ("Stoch RSI K > 80", 72.2, stoch_rsi.iloc[-1, 0] > 80),
        ("CCI (20) 100/-100", 70.4, cci20.iloc[-1] > 100),
        ("ROC (10) Trend Weakness", 67.9, roc10.iloc[-1] < 0)
    ]

    for name, w, cond in bull_defs: results.append({"Indicator": name, "Type": "Bullish", "Weight": w, "Status": "✅" if cond else "❌", "Active": cond})
    for name, w, cond in bear_defs: results.append({"Indicator": name, "Type": "Bearish", "Weight": w, "Status": "✅" if cond else "❌", "Active": cond})
    
    return pd.DataFrame(results)

# --- UI START ---
st.title("📊 Programmeerpartner Trading Dashboard")
st.sidebar.header("Instellingen")
ticker = st.sidebar.text_input("Aandelen Ticker (bijv. PHM, AAPL, TSLA):", "PHM").upper()

if ticker:
    df_data = get_stock_data(ticker)
    
    if df_data is not None:
        analysis_df = perform_full_analysis(df_data)
        
        # Dashboard Score Sectie
        col1, col2 = st.columns(2)
        
        for i, t in enumerate(["Bullish", "Bearish"]):
            sub = analysis_df[analysis_df['Type'] == t]
            total_possible = sub['Weight'].sum()
            actual = sub[sub['Active']]['Weight'].sum()
            score = (actual / total_possible * 100) if total_possible > 0 else 0
            
            with [col1, col2][i]:
                color = "green" if t == "Bullish" else "red"
                st.metric(f"{t} Score", f"{score:.1f}%")
                st.progress(score / 100)
        
        st.divider()
        
        # Tabel Weergave
        st.subheader(f"Gedetailleerde Analyse voor {ticker}")
        st.dataframe(analysis_df[["Indicator", "Type", "Weight", "Status"]], use_container_width=True, height=500)
    else:
        st.error("Kon geen data vinden voor dit symbool. Controleer de ticker.")

st.sidebar.info("Tip: Installeer de modules via de terminal als je fouten ziet: `pip install streamlit yfinance pandas-ta pandas`.")



