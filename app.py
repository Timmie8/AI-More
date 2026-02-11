import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta

st.set_page_config(page_title="Ultimate 50+ Indicator Scanner", layout="wide")

@st.cache_data(ttl=300)
def get_clean_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_all_signals(df):
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    s = [] # Lijst voor alle resultaten

    # --- 1. PRE-CALCULATIONS (Voor snelheid) ---
    # Bollinger Bands
    bb50 = ta.bbands(c, length=50, std=2); bb20 = ta.bbands(c, length=20, std=1)
    bb10 = ta.bbands(c, length=10, std=1); bb5 = ta.bbands(c, length=5, std=1)
    
    # Moving Averages
    sma5 = ta.sma(c, 5); sma10 = ta.sma(c, 10); sma20 = ta.sma(c, 20); sma50 = ta.sma(c, 50)
    ema5 = ta.ema(c, 5); ema10 = ta.ema(c, 10); ema20 = ta.ema(c, 20)
    wma5 = ta.wma(c, 5); wma50 = ta.wma(c, 50)
    dema5 = ta.dema(c, 5); dema10 = ta.dema(c, 10); dema20 = ta.dema(c, 20); dema50 = ta.dema(c, 50); dema100 = ta.dema(c, 100)
    tema5 = ta.tema(c, 5); tema10 = ta.tema(c, 10); tema20 = ta.tema(c, 20)
    kama5 = ta.kama(c, 5); kama10 = ta.kama(c, 10); kama20 = ta.kama(c, 20)
    
    # Oscillators & Momentum
    macd12 = ta.macd(c, 12, 26, 9); macd10 = ta.macd(c, 10, 20, 5)
    stoch_rsi = ta.stochrsi(c, length=14); stoch_rsi21 = ta.stochrsi(c, length=21)
    mfi5 = ta.mfi(h, l, c, v, length=5); mfi7 = ta.mfi(h, l, c, v, length=7)
    cci5 = ta.cci(h, l, c, 5); cci20 = ta.cci(h, l, c, 20); cci40 = ta.cci(h, l, c, 40)
    aroon = ta.aroon(h, l, 10); chop = ta.chop(h, l, c, 14)
    adx9 = ta.adx(h, l, c, 9); adx21 = ta.adx(h, l, c, 21)
    sar = ta.sar(h, l)

    # --- 2. BULLISH SIGNALS (Op basis van jouw lijst) ---
    bull_rules = [
        ("MACD (12,26,9) Cross", 95.0, macd12['MACD_12_26_9'].iloc[-1] > macd12['MACDs_12_26_9'].iloc[-1]),
        ("KAMA (20) Crossover", 93.3, c.iloc[-1] > kama20.iloc[-1]),
        ("Stoch RSI (9) D > 50", 90.5, stoch_rsi['STOCHRSId_14_14_3_3'].iloc[-1] > 50),
        ("DEMA (20) Crossover", 89.3, c.iloc[-1] > dema20.iloc[-1]),
        ("Aroon Osc (10) > 0", 88.5, aroon['AROONOSC_10'].iloc[-1] > 0),
        ("BB (50,2,2) Breakout", 88.2, c.iloc[-1] > bb50['BBU_50_2.0'].iloc[-1]),
        ("WMA (5) Crossover", 86.0, c.iloc[-1] > wma5.iloc[-1]),
        ("SMA (5) Crossover", 82.9, c.iloc[-1] > sma5.iloc[-1]),
        ("DEMA (10) Crossover", 82.9, c.iloc[-1] > dema10.iloc[-1]),
        ("ADX (9) > 20", 80.4, adx9['ADX_9'].iloc[-1] > 20),
        ("SAR Bullish", 82.6, sar.iloc[-1] < c.iloc[-1]),
        ("SMA (10) Crossover", 78.0, c.iloc[-1] > sma10.iloc[-1]),
        ("EMA (10) Crossover", 76.7, c.iloc[-1] > ema10.iloc[-1]),
        ("WMA (50) Crossover", 75.0, c.iloc[-1] > wma50.iloc[-1]),
        ("SMA (50) Crossover", 70.0, c.iloc[-1] > sma50.iloc[-1])
    ]

    # --- 3. BEARISH SIGNALS ---
    bear_rules = [
        ("MFI (5) 70/30 Cross Back", 82.8, mfi5.iloc[-1] < 70 and mfi5.iloc[-2] > 70),
        ("CCI (40) 100/-100 Cross", 81.0, cci40.iloc[-1] < -100),
        ("MFI (5) 80/20 Cross", 81.0, mfi5.iloc[-1] > 80),
        ("BB (50,2,2) Reversal", 88.2, c.iloc[-1] < bb50['BBL_50_2.0'].iloc[-1]),
        ("ADX (21) Crosses", 79.4, adx21['ADX_21'].iloc[-1] > 25),
        ("Stoch RSI (21) K > 80", 72.2, stoch_rsi21['STOCHRSIk_21_21_3_3'].iloc[-1] > 80),
        ("CCI (20) 100/-100", 70.4, cci20.iloc[-1] > 100),
        ("BB (20,1,1) Reversal", 76.5, c.iloc[-1] < bb20['BBL_20_1.0'].iloc[-1])
    ]

    # Verwerk in tabel
    for name, weight, cond in bull_rules: s.append({"Indicator": name, "Type": "Bullish", "Weight": weight, "Active": cond})
    for name, weight, cond in bear_rules: s.append({"Indicator": name, "Type": "Bearish", "Weight": weight, "Active": cond})
    
    return pd.DataFrame(s)

# --- UI DISPLAY ---
st.title("📈 Pro-Analysis Dashboard")
ticker = st.sidebar.text_input("Aandelen Ticker:", "PHM").upper()

if ticker:
    df = get_clean_data(ticker)
    if not df.empty:
        results = calculate_all_signals(df)
        
        # Scores berekenen
        col1, col2 = st.columns(2)
        for i, t in enumerate(["Bullish", "Bearish"]):
            sub = results[results['Type'] == t]
            score = (sub[sub['Active']]['Weight'].sum() / sub['Weight'].sum()) * 100
            with [col1, col2][i]:
                st.metric(f"{t} Score", f"{score:.1f}%")
                st.progress(score / 100)

        st.subheader("Gedetailleerde Analyse")
        st.dataframe(results.sort_values(by="Weight", ascending=False), use_container_width=True)


