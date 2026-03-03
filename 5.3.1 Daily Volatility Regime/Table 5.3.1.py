"""
Fixed code for Section 5.3.1: Daily Frequency - Performance by Volatility Regime
Properly handles multi-index columns from yfinance
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
print("SECTION 5.3.1: DAILY FREQUENCY - PERFORMANCE BY VOLATILITY REGIME (FIXED)")
print("=" * 60)

# =============================================================================
# STEP 2: Download Data with Proper Column Handling
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Downloading Daily Data")
print("=" * 60)

# Download AAPL
aapl = yf.download('AAPL', start='2019-01-01', end='2025-12-31', progress=False)

# Fix multi-index columns if present
if isinstance(aapl.columns, pd.MultiIndex):
    print("Detected multi-index columns. Flattening...")
    aapl.columns = aapl.columns.get_level_values(0)
else:
    aapl.columns = [col.capitalize() for col in aapl.columns]

print(f"AAPL columns: {aapl.columns.tolist()}")
print(f"AAPL shape: {aapl.shape}")

# Download VIX
vix = yf.download('^VIX', start='2019-01-01', end='2025-12-31', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix = vix['Close'].rename('vix')

print(f"VIX shape: {vix.shape}")

# =============================================================================
# STEP 3: Feature Engineering
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

df = aapl.copy()

# Calculate returns (percentage)
df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

# Add VIX (align indices)
df['vix'] = vix

# Create lagged features
for lag in [1, 2, 3, 5]:
    df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

# Create rolling statistics
for window in [5, 10, 20]:
    df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(252)

# Target variables
df['target_return'] = df['returns'].shift(-1)
df['target_direction'] = (df['target_return'] > 0).astype(int)

# Clean data - drop rows with NaN
initial_rows = len(df)
df = df.dropna()
print(f"Dropped {initial_rows - len(df)} rows with NaNs")
print(f"Final shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# =============================================================================
# STEP 4: Define Volatility Regimes Using Percentiles
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Defining Volatility Regimes")
print("=" * 60)

# Use percentile-based thresholds to ensure balanced regimes
viz_33rd = df['vix'].quantile(0.33)
vix_67th = df['vix'].quantile(0.67)

print(f"VIX Statistics:")
print(f"  Min: {df['vix'].min():.2f}")
print(f"  33rd percentile: {viz_33rd:.2f}")
print(f"  Median: {df['vix'].median():.2f}")
print(f"  67th percentile: {vix_67th:.2f}")
print(f"  Max: {df['vix'].max():.2f}")

# Define three regimes
df['regime'] = pd.cut(df['vix'],
                       bins=[-np.inf, viz_33rd, vix_67th, np.inf],
                       labels=['Low', 'Medium', 'High'])

# Count days in each regime
regime_counts = df['regime'].value_counts()
print(f"\nFull Dataset Regime Distribution:")
for regime in ['Low', 'Medium', 'High']:
    count = regime_counts.get(regime, 0)
    pct = count / len(df) * 100
    print(f"  {regime} Volatility: {count} days ({pct:.1f}%)")

# =============================================================================
# STEP 5: Train/Test Split (80/20 chronological)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Train/Test Split")
print("=" * 60)

# Use 80/20 split
split_idx = int(0.8 * len(df))
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Check regime distribution in test set
test_regime_counts = test['regime'].value_counts()
print(f"\nTest Set Regime Distribution:")
for regime in ['Low', 'Medium', 'High']:
    count = test_regime_counts.get(regime, 0)
    pct = count / len(test) * 100
    print(f"  {regime} Volatility: {count} days ({pct:.1f}%)")

# Ensure we have both Low and High regimes in test set
if 'Low' not in test_regime_counts or 'High' not in test_regime_counts:
    print("\n⚠ Test set missing a regime! Using test-specific percentiles...")
    # Use test set's own percentiles
    test_viz_33rd = test['vix'].quantile(0.33)
    test_vix_67th = test['vix'].quantile(0.67)

    test['regime_test'] = pd.cut(test['vix'],
                                  bins=[-np.inf, test_viz_33rd, test_vix_67th, np.inf],
                                  labels=['Low', 'Medium', 'High'])

    test_regime_test_counts = test['regime_test'].value_counts()
    print(f"  Test-specific thresholds: <{test_viz_33rd:.2f} (Low), >{test_vix_67th:.2f} (High)")
    for regime in ['Low', 'Medium', 'High']:
        count = test_regime_test_counts.get(regime, 0)
        pct = count / len(test) * 100
        print(f"    {regime}: {count} days ({pct:.1f}%)")

# =============================================================================
# STEP 6: Prepare Features
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Preparing Features")
print("=" * 60)

# Select feature columns
feature_cols = ['returns', 'vix'] + [f'returns_lag_{lag}' for lag in [1, 2, 3, 5]] + \
               [f'volatility_{window}d' for window in [5, 10, 20]]

print(f"Features: {feature_cols}")

# Split features and target
X_train = train[feature_cols]
y_train = train['target_return']
y_dir_train = train['target_direction']

X_test = test[feature_cols]
y_test = test['target_return']
y_dir_test = test['target_direction']

# Get regimes for test set (use test-specific if available)
if 'regime_test' in test.columns:
    test_regimes = test['regime_test']
else:
    test_regimes = test['regime']

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 7: Train Models
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Training Models")
print("=" * 60)

predictions = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
predictions['Linear Regression'] = lr.predict(X_test_scaled)
print("✓ Linear Regression")

# 2. Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
predictions['Random Forest'] = rf.predict(X_test_scaled)
print("✓ Random Forest")

# 3. Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
predictions['Gradient Boosting'] = gb.predict(X_test_scaled)
print("✓ Gradient Boosting")

# 4. Neural Network
ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
ann.fit(X_train_scaled, y_train)
predictions['ANN (2-layer)'] = ann.predict(X_test_scaled)
print("✓ ANN")

# 5. Quantum Kernel (simplified)
class QuantumKernelRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def _kernel(self, X1, X2):
        """Quantum-inspired kernel: cosine similarity squared"""
        # Normalize
        norm1 = np.linalg.norm(X1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(X2, axis=1, keepdims=True) + 1e-8
        X1_norm = X1 / norm1
        X2_norm = X2 / norm2

        # Cosine similarity
        sim = np.dot(X1_norm, X2_norm.T)

        # Square it (quantum-inspired)
        return sim ** 2

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        K = self._kernel(X, self.X_train)
        # Simple weighted average based on kernel similarity
        weights = K / (K.sum(axis=1, keepdims=True) + 1e-8)
        return np.sum(weights * self.y_train, axis=1)

qk = QuantumKernelRidge(alpha=1.0)
qk.fit(X_train_scaled[:500], y_train.values[:500])  # Use subset for speed
predictions['Quantum Kernel'] = qk.predict(X_test_scaled)
print("✓ Quantum Kernel")

# 6. Amplitude Encoding (simplified)
class AmplitudeEncoding:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None

    def _amplitude_encode(self, X):
        """Simple amplitude-inspired encoding"""
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / norms

    def fit(self, X, y):
        X_enc = self._amplitude_encode(X)
        # Ridge regression solution
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
ae.fit(X_train_scaled[:500], y_train.values[:500])  # Use subset
predictions['Amplitude Encoding'] = ae.predict(X_test_scaled)
print("✓ Amplitude Encoding")

# =============================================================================
# STEP 8: Calculate Performance by Regime
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Calculating Performance by Volatility Regime")
print("=" * 60)

# Create masks for Low and High regimes
low_mask = test_regimes == 'Low'
high_mask = test_regimes == 'High'

print(f"Test samples by regime:")
print(f"  Low Volatility: {sum(low_mask)} samples")
print(f"  High Volatility: {sum(high_mask)} samples")
print(f"  Medium Volatility: {sum(~(low_mask | high_mask))} samples")

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
    if sum(low_mask_aligned) >= 5:  # Need at least 5 samples
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
    if sum(high_mask_aligned) >= 5:
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
# STEP 9: Display Table 5.6
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Table 5.6")
print("=" * 60)

# Convert results to DataFrame
df_display = pd.DataFrame(results)

# Clean display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 20)

print("\n" + "=" * 60)
print("TABLE 5.6: Performance by Volatility Regime (Daily Frequency)")
print("=" * 60)
print("\n")
print(df_display.to_string(index=False))

# Save results
df_display.to_csv('table_5_6_regime_performance.csv', index=False)
print(f"\n✓ Results saved to table_5_6_regime_performance.csv")

# LaTeX format for thesis
latex_filename = 'table_5_6_regime_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Performance by Volatility Regime (Daily Frequency)}\n")
    f.write("\\label{tab:regime_performance}\n")
    f.write("\\begin{tabular}{lcccccc}\n")
    f.write("\\hline\n")
    f.write(
        "Model & RMSE High (bp) & RMSE Low (bp) & $\\Delta$ RMSE (bp) & DA High (\\%) & DA Low (\\%) & $\\Delta$ DA (pp) \\\\\n")
    f.write("\\hline\n")

    for _, row in df_display.iterrows():
        f.write(
            f"{row['Model']} & {row['RMSE_High (bp)']} & {row['RMSE_Low (bp)']} & {row['Δ_RMSE (bp)']} & {row['DA_High (%)']} & {row['DA_Low (%)']} & {row['Δ_DA (pp)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")
# =============================================================================
# STEP 10: Visualization
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Visualization")
print("=" * 60)

# Filter to models with both regimes
valid_results = [r for r in results if r['Δ_RMSE (bp)'] != 'N/A']

if valid_results:
    models_plot = [r['Model'] for r in valid_results]
    delta_rmse = [float(r['Δ_RMSE (bp)']) for r in valid_results]

    plt.figure(figsize=(12, 6))
    colors = ['purple' if ('Quantum' in m or 'Amplitude' in m) else 'steelblue' for m in models_plot]
    bars = plt.bar(models_plot, delta_rmse, color=colors, alpha=0.8)

    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Δ RMSE (basis points)', fontsize=12)
    plt.title('Figure 5.3.1: Instability Measure - Lower is More Robust', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, delta_rmse):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('figure_5_3_1_delta_rmse_fixed.png', dpi=300)
    plt.savefig('figure_5_3_1_delta_rmse_fixed.pdf')
    print("✓ Figure saved: figure_5_3_1_delta_rmse_fixed.png")
else:
    print("No valid results for visualization")

# =============================================================================
# STEP 11: Statistical Test
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Statistical Test for H1a")
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
        print("✓ H1a SUPPORTED: Quantum models significantly more robust")
    else:
        print("✗ H1a NOT supported: No significant difference")
else:
    print("Insufficient data for statistical test")

# =============================================================================
# STEP 12: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - SECTION 5.3.1 COMPLETE")
print("=" * 60)
print(f"\n✓ Test period: {test.index.min()} to {test.index.max()}")
print(f"✓ Low volatility samples: {sum(low_mask)}")
print(f"✓ High volatility samples: {sum(high_mask)}")
print(f"✓ Models with complete data: {len(valid_results)}/{len(results)}")
print(f"✓ Table saved to: table_5_6_regime_performance_fixed.csv")
print("\n" + "=" * 60)