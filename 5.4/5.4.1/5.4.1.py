"""
Complete code for Section 5.4.1: Performance Across Feature Subsets
Table 5.9: Performance across feature subsets and sensitivity metric S^m
Figure 5.5: Relative RMSE increase by feature subset
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

#statistics
from scipy import stats
import scipy.stats as scipy_stats

print("=" * 60)
print("SECTION 5.4.1: PERFORMANCE ACROSS FEATURE SUBSETS")
print("Table 5.9: Feature subset sensitivity analysis")
print("Figure 5.5: Relative RMSE increase visualization")
print("=" * 60)

# =============================================================================
# STEP 2: Download Daily Data
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Downloading Daily Data")
print("=" * 60)

# Download AAPL
aapl = yf.download('AAPL', start='2019-01-01', end='2025-12-31', progress=False)
if isinstance(aapl.columns, pd.MultiIndex):
    aapl.columns = aapl.columns.get_level_values(0)
aapl.columns = [col.capitalize() for col in aapl.columns]

print(f"AAPL shape: {aapl.shape}")
print(f"Date range: {aapl.index.min()} to {aapl.index.max()}")

# =============================================================================
# STEP 3: Create Full Feature Set (Section 4.3.1)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Creating Full Feature Set")
print("=" * 60)

df = aapl.copy()

# Calculate returns (percentage)
df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

# =============================================================================
# PRICE-BASED FEATURES
# =============================================================================
print("  - Adding price-based features...")
for k in [1, 2, 3, 4, 5]:
    df[f'lag_return_{k}d'] = df['returns'].shift(k)

df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
df['opening_gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================
print("  - Adding technical indicators...")


# RSI (14-day)
def rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


df['RSI_14'] = rsi(df['Close'], 14)

# MACD
exp1 = df['Close'].ewm(span=12).mean()
exp2 = df['Close'].ewm(span=26).mean()
df['MACD'] = exp1 - exp2
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# Simple Moving Averages
for period in [5, 10, 20]:
    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    df[f'price_to_SMA_{period}'] = (df['Close'] / df[f'SMA_{period}'] - 1) * 100

# Bollinger Bands
period = 20
df['BB_middle'] = df['Close'].rolling(window=period).mean()
df['BB_std'] = df['Close'].rolling(window=period).std()
df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
df['BB_%B'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# Average True Range (ATR)
df['tr1'] = df['High'] - df['Low']
df['tr2'] = abs(df['High'] - df['Close'].shift(1))
df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR_14'] = df['TR'].rolling(window=14).mean()
df = df.drop(['tr1', 'tr2', 'tr3', 'TR'], axis=1)

# =============================================================================
# VOLATILITY MEASURES
# =============================================================================
print("  - Adding volatility measures...")

# Historical Volatility
for window in [5, 10, 20]:
    df[f'hist_vol_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(252)

# Parkinson Volatility
df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) *
                              (np.log(df['High'] / df['Low']) ** 2)) * 100

# =============================================================================
# VOLUME FEATURES
# =============================================================================
print("  - Adding volume features...")
df['log_volume'] = np.log(df['Volume'] + 1)
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
df['volume_change'] = df['Volume'].pct_change() * 100

# =============================================================================
# MARKET CONTEXT
# =============================================================================
print("  - Adding market context...")
try:
    spy = yf.download('SPY', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_returns = (spy['Close'] / spy['Close'].shift(1) - 1) * 100
    df['spy_return'] = spy_returns.reindex(df.index)
    df['spy_volume_ratio'] = (spy['Volume'] / spy['Volume'].rolling(window=5).mean()).reindex(df.index)
except:
    df['spy_return'] = 0
    df['spy_volume_ratio'] = 1

# VIX
try:
    vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df['vix'] = vix['Close'].reindex(df.index)
    df['vix_change'] = vix['Close'].pct_change().reindex(df.index) * 100
except:
    df['vix'] = 20
    df['vix_change'] = 0

# =============================================================================
# TARGET
# =============================================================================
print("  - Creating target variable...")
df['target_return'] = df['returns'].shift(-1)

# Clean data
df = df.dropna()
print(f"\nFinal full dataset shape: {df.shape}")
print(f"Total features: {len(df.columns)}")

# =============================================================================
# STEP 4: Define Feature Subsets
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Defining Feature Subsets")
print("=" * 60)

# Full feature set (all features except target)
full_features = [col for col in df.columns if col not in
                 ['returns', 'target_return', 'Volume']]
print(f"Full features count: {len(full_features)}")

# Technical indicators only
technical_features = ['RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
                      'price_to_SMA_5', 'price_to_SMA_10', 'price_to_SMA_20',
                      'BB_%B', 'ATR_14']
technical_features = [f for f in technical_features if f in df.columns]
print(f"Technical features count: {len(technical_features)}")

# Price/Volume only
price_volume_features = ['lag_return_1d', 'lag_return_2d', 'lag_return_3d',
                         'lag_return_4d', 'lag_return_5d', 'price_range',
                         'opening_gap', 'log_volume', 'volume_ratio', 'volume_change']
price_volume_features = [f for f in price_volume_features if f in df.columns]
print(f"Price/Volume features count: {len(price_volume_features)}")

# =============================================================================
# STEP 5: Train/Test Split
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

# Target
y_train = train['target_return']
y_test = test['target_return']

# =============================================================================
# STEP 6: Define Models for Comparison
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Initializing Models")
print("=" * 60)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    'ANN': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
    'Linear Regression': LinearRegression()
}


# Quantum-inspired models
class QuantumKernelRidge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def _kernel(self, X1, X2):
        norm1 = np.linalg.norm(X1, axis=1, keepdims=True) + 1e-8
        norm2 = np.linalg.norm(X2, axis=1, keepdims=True) + 1e-8
        X1_norm = X1 / norm1
        X2_norm = X2 / norm2
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


models['Quantum Kernel'] = QuantumKernelRidge(alpha=1.0)


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


models['Amplitude Encoding'] = AmplitudeEncoding(alpha=1.0)

print(f"Total models: {len(models)}")

# =============================================================================
# STEP 7: Evaluate Models on Each Feature Subset
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Evaluating Models on Feature Subsets")
print("=" * 60)

subsets = {
    'Full': full_features,
    'Technical Only': technical_features,
    'Price/Volume Only': price_volume_features
}

results = []

for subset_name, subset_features in subsets.items():
    print(f"\n{'-' * 40}")
    print(f"Evaluating on {subset_name} features ({len(subset_features)} features)")
    print(f"{'-' * 40}")

    # Prepare data for this subset
    X_train_subset = train[subset_features]
    X_test_subset = test[subset_features]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    X_test_scaled = scaler.transform(X_test_subset)

    for model_name, model in models.items():
        print(f"  Training {model_name}...", end=' ')

        try:
            if model_name in ['Quantum Kernel', 'Amplitude Encoding']:
                # These models might need subset of data for speed
                if len(X_train_scaled) > 500:
                    model.fit(X_train_scaled[:500], y_train.values[:500])
                else:
                    model.fit(X_train_scaled, y_train.values)
            else:
                model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 1000  # Convert to ×10⁻³

            results.append({
                'Model': model_name,
                'Subset': subset_name,
                'RMSE': rmse
            })
            print(f"RMSE: {rmse:.2f}")
        except Exception as e:
            print(f"Failed: {e}")
            results.append({
                'Model': model_name,
                'Subset': subset_name,
                'RMSE': np.nan
            })

# =============================================================================
# STEP 8: Create Pivot Table and Calculate Sensitivity Metric S^m
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Calculating Sensitivity Metric S^m")
print("=" * 60)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Pivot to get RMSE for each model and subset
pivot_df = results_df.pivot(index='Model', columns='Subset', values='RMSE').reset_index()

# Get baseline (Full) RMSE
baseline = pivot_df[['Model', 'Full']].copy()
baseline.columns = ['Model', 'Baseline_RMSE']

# Calculate relative increase for each subset
pivot_df = pivot_df.merge(baseline, on='Model')

for subset in ['Technical Only', 'Price/Volume Only']:
    pivot_df[f'{subset}_Increase'] = (pivot_df[subset] - pivot_df['Baseline_RMSE']) / pivot_df['Baseline_RMSE'] * 100

# Calculate S^m (average relative increase across subsets)
pivot_df['S^m'] = pivot_df[['Technical Only_Increase', 'Price/Volume Only_Increase']].mean(axis=1)

# Format for display
display_df = pivot_df[['Model', 'Full', 'Technical Only', 'Price/Volume Only', 'S^m']].copy()
display_df.columns = ['Model', 'Full (Baseline)', 'Technical Only', 'Price/Volume Only', 'S^m']

# Round values
for col in ['Full (Baseline)', 'Technical Only', 'Price/Volume Only']:
    display_df[col] = display_df[col].round(2)
display_df['S^m'] = display_df['S^m'].round(3)

# Sort by S^m (lower is better)
display_df = display_df.sort_values('S^m')

print("\n" + "=" * 60)
print("TABLE 5.9: Performance across feature subsets and sensitivity metric S^m")
print("(daily frequency). Lower S^m indicates lower sensitivity to feature selection.")
print("=" * 60)
print("\n")
print(display_df.to_string(index=False))

# Save results
display_df.to_csv('table_5_9_feature_sensitivity.csv', index=False)
print(f"\n✓ Results saved to table_5_9_feature_sensitivity.csv")

# LaTeX format
latex_filename = 'table_5_9_feature_sensitivity_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Performance across feature subsets and sensitivity metric $S^m$ (daily frequency)}\n")
    f.write("\\label{tab:feature_sensitivity}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Model & Full (Baseline) & Technical Only & Price/Volume Only & $S^m$ \\\\\n")
    f.write("\\hline\n")

    for _, row in display_df.iterrows():
        f.write(
            f"{row['Model']} & {row['Full (Baseline)']:.2f} & {row['Technical Only']:.2f} & {row['Price/Volume Only']:.2f} & {row['S^m']:.3f} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 9: Create Figure 5.5 - Relative RMSE Increase
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Figure 5.5")
print("=" * 60)

# Prepare data for plotting
plot_data = []
for _, row in pivot_df.iterrows():
    plot_data.append({
        'Model': row['Model'],
        'Subset': 'Technical Only',
        'Relative Increase (%)': row['Technical Only_Increase']
    })
    plot_data.append({
        'Model': row['Model'],
        'Subset': 'Price/Volume Only',
        'Relative Increase (%)': row['Price/Volume Only_Increase']
    })

plot_df = pd.DataFrame(plot_data)

# Identify quantum models
quantum_models = ['Quantum Kernel', 'Amplitude Encoding']
plot_df['Type'] = plot_df['Model'].apply(
    lambda x: 'Quantum-Inspired' if x in quantum_models else 'Classical'
)

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Plot bars
sns.barplot(data=plot_df, x='Model', y='Relative Increase (%)',
            hue='Subset', palette={'Technical Only': 'steelblue', 'Price/Volume Only': 'orange'},
            alpha=0.8, ax=ax)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Highlight quantum models
for i, model in enumerate(plot_df['Model'].unique()):
    if model in quantum_models:
        ax.get_xticklabels()[i].set_color('purple')
        ax.get_xticklabels()[i].set_fontweight('bold')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Relative RMSE Increase (%)', fontsize=12)
ax.set_title(
    'Figure 5.5: Relative RMSE Increase by Feature Subset\n(Lower is better - indicates less sensitivity to feature selection)',
    fontsize=14)
ax.legend(title='Feature Subset', loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8, rotation=0)

plt.tight_layout()
plt.savefig('figure_5_5_feature_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_5_feature_sensitivity.pdf', bbox_inches='tight')
print("✓ Figure 5.5 saved to figure_5_5_feature_sensitivity.png and .pdf")

# =============================================================================
# STEP 10: Create Additional Visualization - S^m Comparison
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating S^m Comparison Chart")
print("=" * 60)

plt.figure(figsize=(12, 6))

# Sort by S^m
sm_data = display_df[['Model', 'S^m']].copy()
sm_data = sm_data.sort_values('S^m')

# Colors
colors = ['purple' if m in quantum_models else 'steelblue' for m in sm_data['Model']]

bars = plt.bar(sm_data['Model'], sm_data['S^m'], color=colors, alpha=0.8)

# Add horizontal line at average
avg_sm = sm_data['S^m'].mean()
plt.axhline(y=avg_sm, color='red', linestyle='--', alpha=0.7, label=f'Average S^m: {avg_sm:.3f}')

plt.xlabel('Model', fontsize=12)
plt.ylabel('S^m (Sensitivity Metric)', fontsize=12)
plt.title('Figure 5.5b: Feature Sensitivity Metric S^m - Lower is Better', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, sm_data['S^m']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure_5_5b_sm_comparison.png', dpi=300)
plt.savefig('figure_5_5b_sm_comparison.pdf')
print("✓ Figure 5.5b saved to figure_5_5b_sm_comparison.png and .pdf")

# =============================================================================
# STEP 11: Statistical Comparison of S^m
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Statistical Comparison of S^m")
print("=" * 60)

# Import stats if not already imported
from scipy import stats

quantum_models = ['Quantum Kernel', 'Amplitude Encoding']
quantum_sm = display_df[display_df['Model'].isin(quantum_models)]['S^m'].values
classical_sm = display_df[~display_df['Model'].isin(quantum_models)]['S^m'].values

print(f"\nQuantum models S^m: {quantum_sm}")
print(f"  Mean: {np.mean(quantum_sm):.4f}")
print(f"  Std: {np.std(quantum_sm):.4f}")

print(f"\nClassical models S^m: {classical_sm}")
print(f"  Mean: {np.mean(classical_sm):.4f}")
print(f"  Std: {np.std(classical_sm):.4f}")

# Perform t-test
t_stat, p_value = stats.ttest_ind(quantum_sm, classical_sm)

print(f"\nt-test results:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.05:
    print("  ✓ Significant difference: Quantum models are less sensitive to feature selection")
    print(f"    Quantum models have {np.mean(classical_sm) - np.mean(quantum_sm):.4f} lower S^m on average")
else:
    print("  ✗ No significant difference between quantum and classical models")

# Also calculate Cohen's d for effect size
pooled_std = np.sqrt((np.std(quantum_sm)**2 + np.std(classical_sm)**2) / 2)
cohens_d = (np.mean(classical_sm) - np.mean(quantum_sm)) / pooled_std

print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("  Negligible effect")
elif abs(cohens_d) < 0.5:
    print("  Small effect")
elif abs(cohens_d) < 0.8:
    print("  Medium effect")
else:
    print("  Large effect")
# =============================================================================
# STEP 12: Summary Statistics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 12: Summary Statistics")
print("=" * 60)

# Calculate average degradation
print("\nAverage RMSE increase by subset:")
for subset in ['Technical Only', 'Price/Volume Only']:
    avg_increase = pivot_df[f'{subset}_Increase'].mean()
    print(f"  {subset}: {avg_increase:.2f}%")

print(f"\nAverage S^m by model type:")
print(f"  Quantum-inspired: {np.mean(quantum_sm):.4f}")
print(f"  Classical: {np.mean(classical_sm):.4f}")
print(f"  Difference: {np.mean(classical_sm) - np.mean(quantum_sm):.4f}")

# Find best and worst
best_model = display_df.loc[display_df['S^m'].idxmin()]
worst_model = display_df.loc[display_df['S^m'].idxmax()]

print(f"\nMost robust model (lowest S^m): {best_model['Model']} (S^m = {best_model['S^m']:.4f})")
print(f"Most sensitive model (highest S^m): {worst_model['Model']} (S^m = {worst_model['S^m']:.4f})")

# =============================================================================
# STEP 13: Final Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - SECTION 5.4.1 COMPLETE")
print("=" * 60)
print(
    f"\n✓ Feature subsets evaluated: Full ({len(full_features)}), Technical ({len(technical_features)}), Price/Volume ({len(price_volume_features)})")
print(f"✓ Models evaluated: {len(models)}")
print(f"✓ Test period: {test.index.min()} to {test.index.max()} ({len(test)} days)")
print(f"\n✓ Key Findings:")
print(f"  - Quantum models average S^m: {np.mean(quantum_sm):.4f}")
print(f"  - Classical models average S^m: {np.mean(classical_sm):.4f}")
print(f"  - Improvement: {np.mean(classical_sm) - np.mean(quantum_sm):.4f} ({p_value:.4f} p-value)")
print(f"\n✓ Table 5.9 saved to: table_5_9_feature_sensitivity.csv")
print(f"✓ Figure 5.5 saved to: figure_5_5_feature_sensitivity.png")
print("\n" + "=" * 60)