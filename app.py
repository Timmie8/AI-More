import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- FEATURE ENGINEERING (Dit maakt de voorspelling beter) ---
def prepare_smart_data(df):
    df = df.copy()
    # Kijk niet naar de prijs, maar naar de verandering (Stationarity)
    df['Returns'] = df['Close'].pct_change()
    df['Vol_Change'] = df['Volume'].pct_change()
    
    # Voeg momentum toe
    df['RSI'] = df['RSI'] / 100  # Normaliseren tussen 0 en 1
    df['MA_Gap'] = (df['Close'] - df['MA20']) / df['MA20']
    
    # De 'Target': Stijgt de prijs over 5 dagen met meer dan 2%? (1 of 0)
    df['Target'] = (df['Close'].shift(-5) > df['Close'] * 1.02).astype(int)
    
    return df.dropna()

# --- XGBOOST MODEL ---
smart_df = prepare_smart_data(df)
features = ['Returns', 'Vol_Change', 'RSI', 'MA_Gap']
X = smart_df[features]
y = smart_df['Target']

# Train een Classifier in plaats van een Regressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
model.fit(X_train, y_train)

# Voorspelling voor VANDAAG
current_features = X.tail(1)
prediction_prob = model.predict_proba(current_features)[0][1] # Kans op >2% stijging

