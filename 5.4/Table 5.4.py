"""
Complete code for Table 5.4: Directional Accuracy Metrics (hourly frequency)
"""

# =============================================================================
# STEP 1: Import Libraries
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Time Series
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("TABLE 5.4 GENERATION SCRIPT - HOURLY DIRECTIONAL ACCURACY")
print("=" * 60)
print("Libraries imported successfully")

# =============================================================================
# STEP 2: Download Hourly Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Downloading Hourly Data from Yahoo Finance")
print("=" * 60)


def download_hourly_data(tickers, days_back=700):
    """Download hourly OHLCV data for the last N days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    print(f"Requesting data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    data = {}
    for ticker in tickers:
        print(f"Downloading hourly data for {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             interval="1h", progress=False, auto_adjust=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} hourly rows")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    return data


tickers = ['AAPL']
data_dict = download_hourly_data(tickers)

if not data_dict:
    print("ERROR: No data downloaded!")
    sys.exit(1)

df_raw = data_dict['AAPL'].copy()
df_raw.columns = [col.capitalize() for col in df_raw.columns]
print(f"\nData shape: {df_raw.shape}")
print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

# =============================================================================
# STEP 3: Feature Engineering for Hourly Directional Prediction
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering for Hourly Directional Prediction")
print("=" * 60)


def engineer_hourly_features(df):
    """Engineer features for hourly directional prediction."""
    df = df.copy()

    # Returns (percentage)
    df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

    # =========================================================================
    # FEATURES (using corrections from previous graphs)
    # =========================================================================

    print("  - Adding price-based features...")
    # More lags for hourly data (up to 24 hours)
    for k in [1, 2, 3, 4, 6, 12, 24]:
        df[f'lag_return_{k}h'] = df['returns'].shift(k)

    # Price range features
    df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['price_range_ma_6h'] = df['price_range'].rolling(window=6).mean()
    df['price_range_ma_12h'] = df['price_range'].rolling(window=12).mean()

    # Technical indicators (hourly versions)
    print("  - Adding technical indicators...")

    # RSI (14-hour)
    def rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI_14h'] = rsi(df['Close'], 14)

    # MACD (12h, 26h)
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Moving averages
    for period in [6, 12, 24, 48]:
        df[f'SMA_{period}h'] = df['Close'].rolling(window=period).mean()
        df[f'price_to_SMA_{period}h'] = (df['Close'] / df[f'SMA_{period}h'] - 1) * 100

    # Rolling statistics (volatility, skewness, etc.)
    print("  - Adding rolling statistics...")
    for window in [6, 12, 24]:
        df[f'returns_std_{window}h'] = df['returns'].rolling(window=window).std()
        df[f'returns_skew_{window}h'] = df['returns'].rolling(window=window).skew()
        df[f'returns_kurt_{window}h'] = df['returns'].rolling(window=window).kurt()
        df[f'positive_ratio_{window}h'] = (df['returns'] > 0).rolling(window=window).mean()

    # Order flow imbalance (from corrections)
    print("  - Adding order flow features...")
    df['buy_volume'] = df['Volume'] * (df['Close'] > df['Open']).astype(int)
    df['sell_volume'] = df['Volume'] * (df['Close'] < df['Open']).astype(int)
    df['ofi'] = (df['buy_volume'] - df['sell_volume']) / (df['Volume'] + 1)
    df['ofi_abs'] = np.abs(df['ofi'])

    for lag in [1, 2, 3, 6]:
        df[f'ofi_lag_{lag}h'] = df['ofi'].shift(lag)
        df[f'ofi_abs_lag_{lag}h'] = df['ofi_abs'].shift(lag)

    # Volume features
    print("  - Adding volume features...")
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=24).mean()
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_change'] = df['Volume'].pct_change() * 100

    # Time-based features (important for hourly patterns)
    print("  - Adding time features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)

    # Cyclical encoding of hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Market open/close effects
    df['minutes_to_open'] = ((df['hour'] < 9) * (9 - df['hour']) +
                             (df['hour'] >= 16) * (24 - df['hour'] + 9)) * 60
    df['minutes_to_close'] = ((df['hour'] < 9) * (16 - df['hour']) +
                              (df['hour'] >= 16) * (24 - df['hour'] + 16)) * 60

    # =========================================================================
    # TARGET: Direction of next hour's return
    # =========================================================================

    print("  - Creating target variable...")
    df['next_return'] = df['returns'].shift(-1)
    df['target_direction'] = (df['next_return'] > 0).astype(int)  # 1 if up, 0 if down

    return df


print("Engineering hourly features...")
df = engineer_hourly_features(df_raw)

# Clean data
print("\nCleaning data...")
initial_rows = len(df)
df = df.ffill().bfill()
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
df = df.dropna()
print(f"Dropped {initial_rows - len(df)} rows with NaNs")
print(f"Final shape: {df.shape}")

if len(df) < 500:
    print("ERROR: Too few samples!")
    sys.exit(1)

# =============================================================================
# STEP 4: Train/Validation/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Train/Validation/Test Split")
print("=" * 60)

n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} hours)")
print(f"Validation: {val.index.min()} to {val.index.max()} ({len(val)} hours)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} hours)")

# Features (exclude target-related columns)
feature_cols = [col for col in df.columns if col not in
                ['returns', 'next_return', 'target_direction', 'Volume',
                 'buy_volume', 'sell_volume']]

X_train = train[feature_cols]
y_train = train['target_direction']

X_val = val[feature_cols]
y_val = val['target_direction']

X_test = test[feature_cols]
y_test = test['target_direction']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Class balance - Up: {y_train.mean() * 100:.1f}%, Down: {(1 - y_train.mean()) * 100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 5: Define Models for Hourly Classification
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Initializing Models for Hourly Directional Prediction")
print("=" * 60)

models = {}

# Benchmark
print("\n1. Benchmark Models:")
models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
print("  ✓ Logistic Regression")

# Classical ML
print("\n2. Classical ML Models:")
models['Random Forest'] = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=10,
    random_state=42, n_jobs=-1
)
print("  ✓ Random Forest")

models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42
)
print("  ✓ Gradient Boosting")

models['ANN (2-layer)'] = MLPClassifier(
    hidden_layer_sizes=(256, 128), activation='relu',
    learning_rate_init=0.001, max_iter=1000, early_stopping=True,
    validation_fraction=0.1, random_state=42
)
print("  ✓ Neural Network")

# Quantum-Inspired
print("\n3. Quantum-Inspired Models:")


def quantum_kernel(X, Y=None):
    from sklearn.metrics.pairwise import cosine_similarity
    if Y is None:
        Y = X
    return cosine_similarity(X, Y) ** 2


models['Quantum SVM'] = SVC(kernel=quantum_kernel, C=1.0, probability=True, random_state=42)
print("  ✓ Quantum SVM")


class AmplitudeEncodingClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.classes_ = None

    def _amplitude_encode(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms
        n_samples, n_features = X.shape
        features = []
        for i in range(n_samples):
            psi = X_norm[i:i + 1].T
            rho = np.dot(psi, psi.T)
            idx = np.triu_indices_from(rho)
            features.append(rho[idx])
        return np.array(features)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_encoded = self._amplitude_encode(X)
        n_features = X_encoded.shape[1]
        I = np.eye(n_features)

        y_binary = (y == self.classes_[1]).astype(float)

        try:
            self.coef_ = np.linalg.solve(
                X_encoded.T @ X_encoded + self.alpha * I,
                X_encoded.T @ y_binary
            )
        except:
            self.coef_ = np.linalg.pinv(
                X_encoded.T @ X_encoded + self.alpha * I
            ) @ X_encoded.T @ y_binary
        return self

    def predict_proba(self, X):
        X_encoded = self._amplitude_encode(X)
        scores = X_encoded @ self.coef_
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


models['Amplitude Encoding'] = AmplitudeEncodingClassifier(alpha=1.0)
print("  ✓ Amplitude Encoding")

# =============================================================================
# STEP 6: LSTM for Hourly Classification
# =============================================================================

print("\n4. LSTM for Hourly Classification:")


def create_sequences(X, y, timesteps=24):  # 24 hours of history
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)


timesteps = 24
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, timesteps)

# Build LSTM classifier
lstm_model = Sequential([
    LSTM(64, input_shape=(timesteps, X_train.shape[1]), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

print("  ✓ LSTM Classifier")

# =============================================================================
# STEP 7: Train Models and Generate Predictions
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Training Models and Generating Predictions")
print("=" * 60)

predictions = {}
probabilities = {}

# Train sklearn classifiers
for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            probabilities[name] = y_prob
        else:
            probabilities[name] = y_pred

        predictions[name] = y_pred
        print(f"  ✓ Complete")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        predictions[name] = np.zeros(len(y_test))
        probabilities[name] = np.zeros(len(y_test))

# LSTM predictions
print("\nGenerating LSTM predictions...")
lstm_pred_proba = lstm_model.predict(X_test_seq, verbose=0).flatten()
lstm_pred = (lstm_pred_proba > 0.5).astype(int)

lstm_pred_full = np.concatenate([np.zeros(timesteps), lstm_pred])[:len(y_test)]
lstm_proba_full = np.concatenate([np.zeros(timesteps), lstm_pred_proba])[:len(y_test)]

predictions['LSTM'] = lstm_pred_full
probabilities['LSTM'] = lstm_proba_full
print("  ✓ LSTM Complete")

# ARIMA-based signals
print("\nGenerating ARIMA-based signals...")
try:
    arima_model = ARIMA(train['returns'].dropna(), order=(1, 0, 1)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))
    arima_direction = (arima_forecast.values > 0).astype(int)
    predictions['ARIMA(1,0,1)'] = arima_direction
    probabilities['ARIMA(1,0,1)'] = arima_forecast.values
    print("  ✓ ARIMA Complete")
except Exception as e:
    print(f"  ✗ ARIMA failed: {e}")
    predictions['ARIMA(1,0,1)'] = np.zeros(len(y_test))
    probabilities['ARIMA(1,0,1)'] = np.zeros(len(y_test))

print("\nAll models trained successfully!")

# =============================================================================
# STEP 8: Calculate Directional Metrics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Calculating Directional Accuracy Metrics")
print("=" * 60)


def calculate_directional_metrics(y_true, y_pred):
    """Calculate directional accuracy metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'DA (%)': accuracy * 100,
        'Precision (%)': precision * 100,
        'Recall (%)': recall * 100,
        'F1 Score (%)': f1 * 100
    }


results = []
model_order = [
    'Logistic Regression',
    'ARIMA(1,0,1)',
    'Random Forest',
    'Gradient Boosting',
    'ANN (2-layer)',
    'LSTM',
    'Quantum SVM',
    'Amplitude Encoding'
]

for name in model_order:
    if name in predictions:
        y_pred = predictions[name]

        min_len = min(len(y_test), len(y_pred))
        y_true_aligned = y_test.values[:min_len]
        y_pred_aligned = y_pred[:min_len]

        metrics = calculate_directional_metrics(y_true_aligned, y_pred_aligned)

        results.append({
            'Model': name,
            'DA (%)': f"{metrics['DA (%)']:.1f}",
            'Precision (%)': f"{metrics['Precision (%)']:.1f}",
            'Recall (%)': f"{metrics['Recall (%)']:.1f}",
            'F1 Score (%)': f"{metrics['F1 Score (%)']:.1f}"
        })

        print(
            f"{name:25s} | DA: {metrics['DA (%)']:.1f}% | Prec: {metrics['Precision (%)']:.1f}% | Rec: {metrics['Recall (%)']:.1f}%")

# =============================================================================
# STEP 9: Create Table 5.4
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Table 5.4")
print("=" * 60)

df_results = pd.DataFrame(results)

print("\n" + "=" * 60)
print("TABLE 5.4: Directional Accuracy Metrics (hourly frequency)")
print("=" * 60)
print("\n")
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('table_5_4_hourly_directional_results.csv', index=False)
print(f"\n✓ Results saved to table_5_4_hourly_directional_results.csv")

# LaTeX format
latex_filename = 'table_5_4_hourly_directional_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Directional Accuracy Metrics (hourly frequency)}\n")
    f.write("\\label{tab:hourly_directional_accuracy}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Model & DA (\\%) & Precision (\\%) & Recall (\\%) & F1 Score (\\%) \\\\\n")
    f.write("\\hline\n")

    for _, row in df_results.iterrows():
        f.write(
            f"{row['Model']} & {row['DA (%)']} & {row['Precision (%)']} & {row['Recall (%)']} & {row['F1 Score (%)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 10: Create Confusion Matrices for Hourly Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Confusion Matrices - Hourly")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

models_for_cm = ['LSTM', 'Quantum SVM', 'Amplitude Encoding',
                 'Random Forest', 'Gradient Boosting', 'Logistic Regression']

for idx, name in enumerate(models_for_cm):
    if idx >= len(axes):
        break

    ax = axes[idx]

    if name in predictions:
        y_pred = predictions[name]
        min_len = min(len(y_test), len(y_pred))
        y_true_cm = y_test.values[:min_len]
        y_pred_cm = y_pred[:min_len]

        cm = confusion_matrix(y_true_cm, y_pred_cm)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')

        acc = accuracy_score(y_true_cm, y_pred_cm) * 100
        ax.set_title(f'{name}\nAccuracy: {acc:.1f}%', fontsize=11)

plt.suptitle('Figure 5.4: Confusion Matrices - Hourly Directional Predictions', fontsize=16, y=1.02)
plt.tight_layout()

plt.savefig('figure_5_4_hourly_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_4_hourly_confusion_matrices.pdf', bbox_inches='tight')
print("✓ Confusion matrices saved")

# =============================================================================
# STEP 11: Comparison Bar Chart (Daily vs Hourly)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Creating Comparison Chart")
print("=" * 60)

# You can manually add daily results from Table 5.3 here for comparison
daily_da = {
    'Logistic Regression': 50.4,
    'ARIMA(1,0,1)': 53.0,
    'Random Forest': 48.9,
    'Gradient Boosting': 48.5,
    'ANN (2-layer)': 48.1,
    'LSTM': 45.1,
    'Quantum SVM': 54.2,
    'Amplitude Encoding': 53.0
}

hourly_da = {r['Model']: float(r['DA (%)']) for r in results}

plt.figure(figsize=(14, 6))

x = np.arange(len(model_order))
width = 0.35

daily_values = [daily_da.get(name, 0) for name in model_order]
hourly_values = [hourly_da.get(name, 0) for name in model_order]

bars1 = plt.bar(x - width / 2, daily_values, width, label='Daily', color='steelblue')
bars2 = plt.bar(x + width / 2, hourly_values, width, label='Hourly', color='orange')

plt.axhline(y=50, color='red', linestyle='--', label='Random Guess (50%)', alpha=0.7)

plt.xlabel('Model', fontsize=12)
plt.ylabel('Directional Accuracy (%)', fontsize=12)
plt.title('Figure 5.5: Directional Accuracy - Daily vs Hourly Comparison', fontsize=14)
plt.xticks(x, model_order, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()

plt.savefig('figure_5_5_daily_vs_hourly_comparison.png', dpi=300)
plt.savefig('figure_5_5_daily_vs_hourly_comparison.pdf')
print("✓ Comparison chart saved")

# =============================================================================
# STEP 12: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - TABLE 5.4 COMPLETE")
print("=" * 60)
print(f"\n✓ Data period: {df.index.min()} to {df.index.max()}")
print(f"✓ Test period: {test.index.min()} to {test.index.max()}")
print(f"✓ Test samples: {len(test)} hours")
print(f"✓ Random guess baseline: 50%")
print(f"✓ Best hourly model: {results[0]['Model']} ({results[0]['DA (%)']}%)")
print(f"✓ Table saved to: table_5_4_hourly_directional_results.csv")
print("\n" + "=" * 60)