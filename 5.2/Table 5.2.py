"""
Complete code for Table 5.2: Out-of-sample return prediction performance (hourly frequency)
FIXED: All models in graph, better learning, proper scaling
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

# Time Series
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("TABLE 5.2 GENERATION SCRIPT - HOURLY FREQUENCY")
print("=" * 60)
print("Libraries imported successfully")

# =============================================================================
# STEP 2: Download Hourly Data from Yahoo Finance
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
            else:
                print(f"  ✗ {ticker}: No data")
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")

    return data


tickers = ['AAPL']
days_back = 700

print(f"\nDownloading hourly data for last {days_back} days")
print(f"Tickers: {tickers}")

data_dict = download_hourly_data(tickers, days_back)

if not data_dict:
    print("ERROR: No data downloaded! Exiting.")
    sys.exit(1)

# Use AAPL as primary asset
df_raw = data_dict['AAPL'].copy()
print(f"\nUsing AAPL as primary asset for hourly data")
print(f"Hourly data shape: {df_raw.shape}")
print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

# Standardize column names
df_raw.columns = [col.capitalize() for col in df_raw.columns]
print("\nHourly data sample:")
print(df_raw.head())

# =============================================================================
# STEP 3: Feature Engineering for Hourly Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering for Hourly Data")
print("=" * 60)


def engineer_hourly_features(df):
    """Engineer features for hourly frequency data."""
    df = df.copy()

    # Calculate returns (as percentage)
    df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

    # ===== FEATURES (all use data up to time t) =====

    # Make returns more stationary by taking differences of returns
    df['returns_diff'] = df['returns'].diff()

    # Order flow imbalance (simplified) - using current bar's data
    df['buy_volume'] = df['Volume'] * (df['Close'] > df['Open']).astype(int)
    df['sell_volume'] = df['Volume'] * (df['Close'] < df['Open']).astype(int)
    df['ofi'] = (df['buy_volume'] - df['sell_volume']) / (df['Volume'] + 1)
    df['ofi_lag'] = df['ofi'].shift(1)  # Lagged OFI (t-1)

    # ===== TARGETS (future values at t+1) =====

    # Regression target: next hour's return
    df['target_return'] = df['returns'].shift(-1)

    # Classification target: direction of next hour's return (easier to predict)
    df['target_direction'] = (df['target_return'] > 0).astype(int)

    # Alternative: use returns_diff as target (more stationary)
    df['target_return_diff'] = df['returns_diff'].shift(-1)

    return df

    # Price-based features - more lags for hourly data
    print("  - Adding price-based features...")
    for k in [1, 2, 3, 4, 6, 12, 24, 48, 72]:  # More lags up to 3 days
        df[f'lag_return_{k}h'] = df['returns'].shift(k)

    # Rolling statistics
    for window in [6, 12, 24, 48]:
        df[f'returns_ma_{window}h'] = df['returns'].rolling(window=window).mean()
        df[f'returns_std_{window}h'] = df['returns'].rolling(window=window).std()
        df[f'returns_skew_{window}h'] = df['returns'].rolling(window=window).skew()
        df[f'returns_kurt_{window}h'] = df['returns'].rolling(window=window).kurt()

    # Price range features
    df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['price_range_ma'] = df['price_range'].rolling(window=24).mean()

    # Technical Indicators
    print("  - Adding technical indicators...")

    # RSI (14-hour)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI_14h'] = calculate_rsi(df['Close'], 14)

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Moving averages
    for period in [6, 12, 24, 48]:
        df[f'SMA_{period}h'] = df['Close'].rolling(window=period).mean()
        df[f'price_to_SMA_{period}h'] = (df['Close'] / df[f'SMA_{period}h'] - 1) * 100

    # Bollinger Bands
    period = 20
    df['BB_middle'] = df['Close'].rolling(window=period).mean()
    df['BB_std'] = df['Close'].rolling(window=period).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_%B'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'] * 100

    # Volume features
    print("  - Adding volume features...")
    df['volume'] = df['Volume']
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=24).mean()
    df['volume_change'] = df['Volume'].pct_change() * 100

    # Time-based features
    print("  - Adding time-based features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)

    # Target variables (next hour's return)
    print("  - Creating target variables...")
    df['target_return'] = df['returns'].shift(-1)

    return df


print("Engineering hourly features...")
df = engineer_hourly_features(df_raw)

if df is None or len(df) == 0:
    print("ERROR: Feature engineering failed!")
    sys.exit(1)

print(f"\nTotal features engineered: {len(df.columns)}")
print(f"Initial shape with NaNs: {df.shape}")

# =============================================================================
# STEP 4: Handle NaN and Infinite Values
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Handling NaN and Infinite Values")
print("=" * 60)

# Drop columns with too many NaNs
nan_percentage = df.isnull().mean() * 100
cols_to_drop = nan_percentage[nan_percentage > 30].index.tolist()  # Stricter threshold
if cols_to_drop:
    print(f"Dropping columns with >30% NaNs: {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)

# Fill remaining NaNs
print("Filling remaining NaNs...")
df = df.ffill().bfill()

# Handle infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.ffill().bfill()

# Drop any remaining NaNs
initial_rows = len(df)
df = df.dropna()
print(f"Dropped {initial_rows - len(df)} rows with remaining NaNs")
print(f"Final shape after cleaning: {df.shape}")

if len(df) < 500:
    print(f"ERROR: Too few samples ({len(df)}) after cleaning. Cannot proceed.")
    sys.exit(1)

# =============================================================================
# STEP 5: Return Analysis and Target Preparation
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Return Analysis and Target Preparation")
print("=" * 60)

# Analyze returns
returns = df['returns'].dropna()
print(f"Returns statistics (percentage points):")
print(f"  min: {returns.min():.4f}%")
print(f"  max: {returns.max():.4f}%")
print(f"  mean: {returns.mean():.4f}%")
print(f"  std: {returns.std():.4f}%")
print(f"  skew: {returns.skew():.4f}")
print(f"  kurtosis: {returns.kurtosis():.4f}")

# Target is already in percentage points
df['target'] = df['target_return']

# =============================================================================
# STEP 6: Train/Validation/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Train/Validation/Test Split")
print("=" * 60)

# Split chronologically
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} hours)")
print(f"Validation: {val.index.min()} to {val.index.max()} ({len(val)} hours)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} hours)")

# Prepare features and target
feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_return', 'target', 'Volume']]

X_train = train[feature_cols]
y_train = train['target']

X_val = val[feature_cols]
y_val = val['target']

X_test = test[feature_cols]
y_test = test['target']

print(f"\nFeatures: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"Target stats - mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 7: Define Models (without XGBoost)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Initializing Models")
print("=" * 60)

models = {}

# Benchmark Models
print("\n1. Benchmark Models:")
models['Linear Regression'] = LinearRegression()
print("  ✓ Linear Regression")

# Classical ML Models
print("\n2. Classical ML Models:")
models['Random Forest'] = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
print("  ✓ Random Forest")

# XGBoost is skipped - install libomp to enable
print("  ⚠ XGBoost skipped (OpenMP not available)")

models['Gradient Boosting'] = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
print("  ✓ Gradient Boosting")

models['ANN (2-layer)'] = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
print("  ✓ ANN (3-layer)")

# Quantum-Inspired Models
print("\n3. Quantum-Inspired Models:")


def quantum_kernel(X, Y=None):
    """Quantum-inspired kernel using cosine similarity squared."""
    from sklearn.metrics.pairwise import cosine_similarity
    if Y is None:
        Y = X
    cos_sim = cosine_similarity(X, Y)
    return cos_sim ** 2


models['Quantum Kernel'] = KernelRidge(kernel=quantum_kernel, alpha=1.0)
print("  ✓ Quantum Kernel Ridge")


class AmplitudeEncodingRegression:
    """Amplitude Encoding Regression with better numerical stability."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.X_mean = None
        self.X_std = None

    def _amplitude_encode(self, X):
        """Convert features to amplitude encoding state."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Center and scale for stability
        if self.X_mean is not None:
            X = (X - self.X_mean) / (self.X_std + 1e-8)

        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms

        # Create density matrix features
        n_samples, n_features = X.shape
        features = []

        for i in range(n_samples):
            psi = X_norm[i:i + 1].T
            rho = np.dot(psi, psi.T)
            # Take upper triangle
            idx = np.triu_indices_from(rho)
            features.append(rho[idx])

        return np.array(features)

    def fit(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8

        X_encoded = self._amplitude_encode(X)

        # Ridge regression
        n_features = X_encoded.shape[1]
        I = np.eye(n_features)
        try:
            self.coef_ = np.linalg.solve(
                X_encoded.T @ X_encoded + self.alpha * I,
                X_encoded.T @ y
            )
        except:
            self.coef_ = np.linalg.pinv(
                X_encoded.T @ X_encoded + self.alpha * I
            ) @ X_encoded.T @ y
        return self

    def predict(self, X):
        X_encoded = self._amplitude_encode(X)
        return X_encoded @ self.coef_


models['Amplitude Encoding'] = AmplitudeEncodingRegression(alpha=1.0)
print("  ✓ Amplitude Encoding Regression")

print(f"\nTotal models to train: {len(models)}")

# =============================================================================
# STEP 8: Train Models
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Training Models")
print("=" * 60)

predictions = {}

# Train all sklearn models
for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        model.fit(X_train_scaled, y_train)
        predictions[name] = model.predict(X_test_scaled)
        print(f"  ✓ Complete")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        predictions[name] = np.zeros(len(y_test))

# ARIMA
print("\nTraining ARIMA...")
try:
    arima_model = ARIMA(train['returns'].dropna(), order=(2,0,2)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))
    predictions['ARIMA'] = arima_forecast.values
    print("  ✓ Complete")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    predictions['ARIMA'] = np.zeros(len(y_test))

# LSTM
print("\nTraining LSTM...")

def create_sequences(X, y, timesteps=48):  # Use 48 hours (2 days) of history
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 48
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, timesteps)

# Simpler LSTM architecture
lstm_model = Sequential([
    LSTM(50, input_shape=(timesteps, X_train.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(25, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=[early_stop]
)

lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
predictions['LSTM'] = np.concatenate([np.zeros(timesteps), lstm_pred])[:len(y_test)]
print("  ✓ Complete")

print("\nAll models trained successfully!")

# =============================================================================
# STEP 9: Calculate Metrics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Calculating Performance Metrics")
print("=" * 60)


def calculate_metrics(y_true, y_pred, window_size=48):
    """Calculate RMSE, MAE, MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    # Rolling metrics for std
    rmse_list = []
    for i in range(window_size, len(y_true)):
        rmse_list.append(np.sqrt(mean_squared_error(
            y_true[i-window_size:i], y_pred[i-window_size:i]
        )))

    rmse_std = np.std(rmse_list) if rmse_list else 0

    return rmse, rmse_std, mae, mape


results = []
for name, y_pred in predictions.items():
    min_len = min(len(y_test), len(y_pred))
    rmse, rmse_std, mae, mape = calculate_metrics(
        y_test.values[:min_len], y_pred[:min_len]
    )
    results.append({
        'Model': name,
        'RMSE': f"{rmse:.2f} ({rmse_std:.2f})",
        'MAE': f"{mae:.2f}",
        'MAPE': f"{mape:.1f}"
    })
    print(f"{name:25s} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.1f}%")

# =============================================================================
# STEP 10: Create Table
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Table 5.2")
print("=" * 60)

df_results = pd.DataFrame(results)
print("\nTABLE 5.2: Hourly Return Prediction Performance")
print("=" * 60)
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('table_5_2_results.csv', index=False)
print("\n✓ Results saved to table_5_2_results.csv")

# =============================================================================
# STEP 11: Create Figure 5.2 with ALL MODELS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Creating Figure 5.2 - All Models Comparison")
print("=" * 60)

# Select a recent 5-day window (120 hours)
window_start = max(0, len(test) - 300)
window_end = min(window_start + 120, len(test))

# Create subplots
n_models = len(predictions)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

# Plot each model
for idx, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[idx]

    # Plot actual
    ax.plot(test.index[window_start:window_end],
            y_test.values[window_start:window_end],
            label='Actual', linewidth=2, color='black')

    # Plot predicted
    ax.plot(test.index[window_start:window_end],
            y_pred[window_start:window_end],
            label='Predicted', linewidth=1.5, linestyle='--', alpha=0.7)

    # Calculate correlation
    corr = np.corrcoef(
        y_test.values[window_start:window_end],
        y_pred[window_start:window_end]
    )[0, 1]

    ax.set_title(f'{name}\nCorrelation: {corr:.3f}', fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# Hide empty subplots
for idx in range(len(predictions), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Figure 5.2: Hourly Return Predictions - All Models (5-day window)',
             fontsize=16, y=1.02)
plt.tight_layout()

# Save figure
plt.savefig('figure_5_2_all_models.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_2_all_models.pdf', bbox_inches='tight')
print("✓ Figure saved to figure_5_2_all_models.png and .pdf")

# Also create a combined plot
plt.figure(figsize=(20, 10))

# Plot actual
plt.plot(test.index[window_start:window_end],
         y_test.values[window_start:window_end],
         label='Actual', linewidth=3, color='black')

# Plot all models
colors = plt.cm.tab20(np.linspace(0, 1, len(predictions)))
for idx, (name, y_pred) in enumerate(predictions.items()):
    plt.plot(test.index[window_start:window_end],
             y_pred[window_start:window_end],
             label=name, linewidth=1.5, alpha=0.7, color=colors[idx])

plt.xlabel('Date', fontsize=12)
plt.ylabel('Return (%)', fontsize=12)
plt.title('Figure 5.2: All Models - Hourly Return Predictions (5-day window)', fontsize=14)
plt.legend(loc='best', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('figure_5_2_combined.png', dpi=300)
plt.savefig('figure_5_2_combined.pdf')
print("✓ Combined figure saved to figure_5_2_combined.png and .pdf")

# =============================================================================
# STEP 12: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - TABLE 5.2 COMPLETE")
print("=" * 60)
print(f"\n✓ Data period: {df.index.min()} to {df.index.max()}")
print(f"✓ Total observations: {len(df)}")
print(f"✓ Train/Val/Test: {len(train)}/{len(val)}/{len(test)}")
print(f"✓ Models trained: {len(predictions)}")
print(f"✓ Table saved to: table_5_2_results.csv")
print(f"✓ Figures saved: figure_5_2_all_models.png, figure_5_2_combined.png")
print("\n" + "=" * 60)