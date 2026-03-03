"""
Complete code for Section 5.3.2: Hourly Frequency - Performance by Volatility Regime
Using Parkinson volatility for regime classification (test-period median)
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.kernel_ridge import KernelRidge

# Time Series
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("SECTION 5.3.2: HOURLY FREQUENCY - PERFORMANCE BY VOLATILITY REGIME")
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
# STEP 3: Feature Engineering with Parkinson Volatility
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering with Parkinson Volatility")
print("=" * 60)


def engineer_hourly_features(df):
    """Engineer features for hourly frequency with Parkinson volatility."""
    df = df.copy()

    # Returns (percentage)
    df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

    # =========================================================================
    # PARKINSON VOLATILITY (for regime classification)
    # Formula: σ_park = sqrt( (1/(4 ln 2)) * (ln(High/Low))^2 )
    # =========================================================================

    print("  - Computing Parkinson volatility...")
    df['parkinson_raw'] = (1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2)
    df['parkinson_vol'] = np.sqrt(df['parkinson_raw']) * 100  # Convert to percentage

    # Also compute rolling volatility for features
    for window in [6, 12, 24, 48]:
        df[f'parkinson_ma_{window}h'] = df['parkinson_vol'].rolling(window=window).mean()
        df[f'returns_std_{window}h'] = df['returns'].rolling(window=window).std()

    # =========================================================================
    # PRICE-BASED FEATURES
    # =========================================================================

    print("  - Adding price features...")
    for k in [1, 2, 3, 6, 12, 24]:
        df[f'lag_return_{k}h'] = df['returns'].shift(k)
        df[f'lag_parkinson_{k}h'] = df['parkinson_vol'].shift(k)

    df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['opening_gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100

    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================

    print("  - Adding technical indicators...")

    # RSI
    def rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI_14h'] = rsi(df['Close'], 14)

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================

    print("  - Adding volume features...")
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=24).mean()
    df['log_volume'] = np.log(df['Volume'] + 1)

    # =========================================================================
    # TIME-BASED FEATURES
    # =========================================================================

    print("  - Adding time features...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 16)).astype(int)

    # Cyclical encoding of hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # =========================================================================
    # TARGET VARIABLES
    # =========================================================================

    print("  - Creating target variables...")
    df['target_return'] = df['returns'].shift(-1)
    df['target_direction'] = (df['target_return'] > 0).astype(int)

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

print(f"\nParkinson volatility statistics:")
print(f"  Mean: {df['parkinson_vol'].mean():.4f}%")
print(f"  Std: {df['parkinson_vol'].std():.4f}%")
print(f"  Min: {df['parkinson_vol'].min():.4f}%")
print(f"  Max: {df['parkinson_vol'].max():.4f}%")

# =============================================================================
# STEP 4: Train/Validation/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Train/Validation/Test Split")
print("=" * 60)

# Split chronologically (70/15/15)
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} hours)")
print(f"Validation: {val.index.min()} to {val.index.max()} ({len(val)} hours)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} hours)")

# =============================================================================
# STEP 5: Define Volatility Regimes Using Test-Period Median
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Defining Volatility Regimes (Test-Period Median)")
print("=" * 60)

# Calculate Parkinson volatility median on TEST period only
test_parkinson_median = test['parkinson_vol'].median()
test_parkinson_mean = test['parkinson_vol'].mean()

print(f"Test Period Parkinson Volatility:")
print(f"  Median: {test_parkinson_median:.4f}%")
print(f"  Mean: {test_parkinson_mean:.4f}%")
print(f"  Min: {test['parkinson_vol'].min():.4f}%")
print(f"  Max: {test['parkinson_vol'].max():.4f}%")

# Define regimes based on test-period median
test['regime'] = (test['parkinson_vol'] > test_parkinson_median).astype(int)
test['regime_label'] = test['regime'].map({0: 'Low Volatility', 1: 'High Volatility'})

# Count hours in each regime
regime_counts = test['regime_label'].value_counts()
print(f"\nTest Set Regime Distribution:")
for regime in ['Low Volatility', 'High Volatility']:
    count = regime_counts.get(regime, 0)
    pct = count / len(test) * 100
    print(f"  {regime}: {count} hours ({pct:.1f}%)")

# Ensure we have both regimes
if len(regime_counts) < 2:
    print("\n⚠ WARNING: Test set missing a regime! Using 40th/60th percentiles...")
    parkinson_40th = test['parkinson_vol'].quantile(0.4)
    parkinson_60th = test['parkinson_vol'].quantile(0.6)

    # Use middle 20% as buffer
    test['regime'] = 0  # Low
    test.loc[test['parkinson_vol'] > parkinson_60th, 'regime'] = 1  # High
    test.loc[(test['parkinson_vol'] > parkinson_40th) & (
                test['parkinson_vol'] <= parkinson_60th), 'regime'] = -1  # Medium (exclude)

    test['regime_label'] = test['regime'].map({0: 'Low Volatility', 1: 'High Volatility', -1: 'Medium'})

    # Filter to only Low and High
    test = test[test['regime'] != -1].copy()

    print(f"  Using thresholds: <{parkinson_40th:.4f}% (Low), >{parkinson_60th:.4f}% (High)")
    print(f"  Low: {sum(test['regime'] == 0)} hours, High: {sum(test['regime'] == 1)} hours")

# =============================================================================
# STEP 6: Prepare Features
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Preparing Features")
print("=" * 60)

# Feature columns (exclude target and regime-related)
feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_return', 'target_direction', 'Volume',
                 'parkinson_raw', 'parkinson_vol', 'regime', 'regime_label']]

# Get train features from original train set (not modified)
X_train = train[feature_cols]
y_train = train['target_return']
y_dir_train = train['target_direction']

# Test features from test set (with regimes)
X_test = test[feature_cols]
y_test = test['target_return']
y_dir_test = test['target_direction']
test_regimes = test['regime']

print(f"Features: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 7: Define Models (simplified set for hourly)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Initializing Models for Hourly Prediction")
print("=" * 60)

predictions = {}

# 1. Linear Regression
print("\n1. Linear Regression")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
predictions['Linear Regression'] = lr.predict(X_test_scaled)
print("  ✓ Complete")

# 2. Random Forest
print("\n2. Random Forest")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
predictions['Random Forest'] = rf.predict(X_test_scaled)
print("  ✓ Complete")

# 3. Gradient Boosting
print("\n3. Gradient Boosting")
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
predictions['Gradient Boosting'] = gb.predict(X_test_scaled)
print("  ✓ Complete")

# 4. Neural Network
print("\n4. Neural Network")
ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
ann.fit(X_train_scaled, y_train)
predictions['ANN (2-layer)'] = ann.predict(X_test_scaled)
print("  ✓ Complete")

# 5. Quantum Kernel (simplified)
print("\n5. Quantum Kernel")


class QuantumKernelRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def _kernel(self, X1, X2):
        # Normalize
        norm1 = np.linalg.norm(X1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(X2, axis=1, keepdims=True) + 1e-8
        X1_norm = X1 / norm1
        X2_norm = X2 / norm2

        # Cosine similarity squared
        sim = np.dot(X1_norm, X2_norm.T)
        return sim ** 2

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        K = self._kernel(X, self.X_train)
        weights = K / (K.sum(axis=1, keepdims=True) + 1e-8)
        return np.sum(weights * self.y_train, axis=1)


qk = QuantumKernelRidge(alpha=1.0)
qk.fit(X_train_scaled[:500], y_train.values[:500])
predictions['Quantum Kernel'] = qk.predict(X_test_scaled)
print("  ✓ Complete")

# 6. Amplitude Encoding (simplified)
print("\n6. Amplitude Encoding")


class AmplitudeEncoding:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None

    def _amplitude_encode(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / norms

    def fit(self, X, y):
        X_enc = self._amplitude_encode(X)
        n_features = X_enc.shape[1]
        I = np.eye(n_features)
        try:
            self.coef_ = np.linalg.solve(
                X_enc.T @ X_enc + self.alpha * I,
                X_enc.T @ y
            )
        except:
            self.coef_ = np.linalg.pinv(
                X_enc.T @ X_enc + self.alpha * I
            ) @ X_enc.T @ y
        return self

    def predict(self, X):
        X_enc = self._amplitude_encode(X)
        return X_enc @ self.coef_


ae = AmplitudeEncoding(alpha=1.0)
ae.fit(X_train_scaled[:500], y_train.values[:500])
predictions['Amplitude Encoding'] = ae.predict(X_test_scaled)
print("  ✓ Complete")

# =============================================================================
# STEP 8: Calculate Performance by Volatility Regime
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Calculating Performance by Volatility Regime")
print("=" * 60)

# Create masks for Low and High regimes
low_mask = test_regimes == 0
high_mask = test_regimes == 1

print(f"Test samples by regime:")
print(f"  Low Volatility: {sum(low_mask)} hours ({sum(low_mask) / len(test) * 100:.1f}%)")
print(f"  High Volatility: {sum(high_mask)} hours ({sum(high_mask) / len(test) * 100:.1f}%)")

results = []

for name, y_pred in predictions.items():
    # Ensure same length
    min_len = min(len(y_test), len(y_pred))
    y_true_aligned = y_test.values[:min_len]
    y_pred_aligned = y_pred[:min_len]

    # Align masks
    low_mask_aligned = low_mask[:min_len].values if hasattr(low_mask, 'values') else low_mask[:min_len]
    high_mask_aligned = high_mask[:min_len].values if hasattr(high_mask, 'values') else high_mask[:min_len]

    # Calculate metrics for Low regime
    if sum(low_mask_aligned) >= 10:  # Need at least 10 samples for hourly
        rmse_low = np.sqrt(mean_squared_error(
            y_true_aligned[low_mask_aligned],
            y_pred_aligned[low_mask_aligned]
        )) * 100  # Convert to basis points

        # Directional accuracy
        y_dir_true = (y_true_aligned > 0).astype(int)
        y_dir_pred = (y_pred_aligned > 0).astype(int)
        da_low = accuracy_score(
            y_dir_true[low_mask_aligned],
            y_dir_pred[low_mask_aligned]
        ) * 100
    else:
        rmse_low = np.nan
        da_low = np.nan

    # Calculate metrics for High regime
    if sum(high_mask_aligned) >= 10:
        rmse_high = np.sqrt(mean_squared_error(
            y_true_aligned[high_mask_aligned],
            y_pred_aligned[high_mask_aligned]
        )) * 100

        y_dir_true = (y_true_aligned > 0).astype(int)
        y_dir_pred = (y_pred_aligned > 0).astype(int)
        da_high = accuracy_score(
            y_dir_true[high_mask_aligned],
            y_dir_pred[high_mask_aligned]
        ) * 100
    else:
        rmse_high = np.nan
        da_high = np.nan

    # Calculate deltas (if both regimes have values)
    if not np.isnan(rmse_low) and not np.isnan(rmse_high):
        delta_rmse = abs(rmse_high - rmse_low)
        delta_da = abs(da_high - da_low)
    else:
        delta_rmse = np.nan
        delta_da = np.nan

    results.append({
        'Model': name,
        'RMSE_High (bp)': f"{rmse_high:.2f}" if not np.isnan(rmse_high) else "N/A",
        'RMSE_Low (bp)': f"{rmse_low:.2f}" if not np.isnan(rmse_low) else "N/A",
        'Δ_RMSE (bp)': f"{delta_rmse:.2f}" if not np.isnan(delta_rmse) else "N/A",
        'DA_High (%)': f"{da_high:.1f}" if not np.isnan(da_high) else "N/A",
        'DA_Low (%)': f"{da_low:.1f}" if not np.isnan(da_low) else "N/A",
        'Δ_DA (pp)': f"{delta_da:.1f}" if not np.isnan(delta_da) else "N/A"
    })

# =============================================================================
# STEP 9: Create Table 5.7
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Table 5.7 - Hourly Performance by Volatility Regime")
print("=" * 60)

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

print("\n" + "=" * 60)
print("TABLE 5.7: Hourly Performance by Volatility Regime")
print("Using Parkinson volatility (test-period median)")
print("=" * 60)
print("\n")
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('table_5_7_hourly_regime_performance.csv', index=False)
print(f"\n✓ Results saved to table_5_7_hourly_regime_performance.csv")

# LaTeX format
latex_filename = 'table_5_7_hourly_regime_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Hourly Performance by Volatility Regime}\n")
    f.write("\\label{tab:hourly_regime_performance}\n")
    f.write("\\begin{tabular}{lcccccc}\n")
    f.write("\\hline\n")
    f.write(
        "Model & RMSE High (bp) & RMSE Low (bp) & $\\Delta$ RMSE (bp) & DA High (\\%) & DA Low (\\%) & $\\Delta$ DA (pp) \\\\\n")
    f.write("\\hline\n")

    for _, row in df_results.iterrows():
        f.write(
            f"{row['Model']} & {row['RMSE_High (bp)']} & {row['RMSE_Low (bp)']} & {row['Δ_RMSE (bp)']} & {row['DA_High (%)']} & {row['DA_Low (%)']} & {row['Δ_DA (pp)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 10: Create Visualization
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Visualization")
print("=" * 60)

# Filter to models with both regimes
valid_results = [r for r in results if r['Δ_RMSE (bp)'] != 'N/A']

if valid_results:
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Δ RMSE
    models_plot = [r['Model'] for r in valid_results]
    delta_rmse = [float(r['Δ_RMSE (bp)']) for r in valid_results]
    delta_da = [float(r['Δ_DA (pp)']) for r in valid_results]

    # Colors for quantum vs classical
    colors = ['purple' if ('Quantum' in m or 'Amplitude' in m) else 'steelblue' for m in models_plot]

    # Δ RMSE bar chart
    bars1 = axes[0].bar(models_plot, delta_rmse, color=colors, alpha=0.8)
    axes[0].set_xlabel('Model', fontsize=11)
    axes[0].set_ylabel('Δ RMSE (basis points)', fontsize=11)
    axes[0].set_title('Instability Measure - RMSE (Lower is More Robust)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, delta_rmse):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Δ DA bar chart
    bars2 = axes[1].bar(models_plot, delta_da, color=colors, alpha=0.8)
    axes[1].axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5 pp threshold')
    axes[1].set_xlabel('Model', fontsize=11)
    axes[1].set_ylabel('Δ DA (percentage points)', fontsize=11)
    axes[1].set_title('Instability Measure - Directional Accuracy (Lower is More Robust)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend()

    # Add value labels
    for bar, val in zip(bars2, delta_da):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Figure 5.3.2: Hourly Performance Stability Across Volatility Regimes', fontsize=14, y=1.05)
    plt.tight_layout()

    # Save figure
    plt.savefig('figure_5_3_2_hourly_regime_stability.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_5_3_2_hourly_regime_stability.pdf', bbox_inches='tight')
    print("✓ Figure saved: figure_5_3_2_hourly_regime_stability.png")

    # Also create a combined bar chart for Δ RMSE comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models_plot, delta_rmse, color=colors, alpha=0.8)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Δ RMSE (basis points)', fontsize=12)
    plt.title('Figure 5.3.2b: Hourly Instability Measure (Δ RMSE) - Lower is More Robust', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add average line
    avg_delta = np.mean(delta_rmse)
    plt.axhline(y=avg_delta, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_delta:.1f} bp')
    plt.legend()

    # Add value labels
    for bar, val in zip(bars, delta_rmse):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figure_5_3_2b_hourly_delta_rmse.png', dpi=300)
    plt.savefig('figure_5_3_2b_hourly_delta_rmse.pdf')
    print("✓ Additional figure saved: figure_5_3_2b_hourly_delta_rmse.png")

# =============================================================================
# STEP 11: Statistical Test for H1a (Hourly)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Statistical Test for H1a (Hourly)")
print("=" * 60)

# Extract delta values for quantum vs classical
quantum_deltas = []
classical_deltas = []

for r in results:
    if r['Δ_RMSE (bp)'] != 'N/A':
        delta = float(r['Δ_RMSE (bp)'])
        if 'Quantum' in r['Model'] or 'Amplitude' in r['Model']:
            quantum_deltas.append(delta)
        else:
            classical_deltas.append(delta)

if len(quantum_deltas) > 0 and len(classical_deltas) > 0:
    t_stat, p_value = stats.ttest_ind(quantum_deltas, classical_deltas)

    print(f"\nQuantum models mean Δ: {np.mean(quantum_deltas):.2f} bp")
    print(f"Classical models mean Δ: {np.mean(classical_deltas):.2f} bp")
    print(f"Difference: {np.mean(classical_deltas) - np.mean(quantum_deltas):.2f} bp")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("✓ H1a SUPPORTED (hourly): Quantum models significantly more robust")
    else:
        print("✗ H1a NOT supported (hourly): No significant difference")
else:
    print("Insufficient data for statistical test")

# =============================================================================
# STEP 12: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - SECTION 5.3.2 COMPLETE")
print("=" * 60)
print(f"\n✓ Test period: {test.index.min()} to {test.index.max()}")
print(f"✓ Total test hours: {len(test)}")
print(f"✓ Low volatility hours: {sum(low_mask)} ({sum(low_mask) / len(test) * 100:.1f}%)")
print(f"✓ High volatility hours: {sum(high_mask)} ({sum(high_mask) / len(test) * 100:.1f}%)")
print(f"✓ Models evaluated: {len(predictions)}")
print(f"✓ Table saved to: table_5_7_hourly_regime_performance.csv")
print(f"✓ Figures saved: figure_5_3_2_hourly_regime_stability.png")
print("\n" + "=" * 60)