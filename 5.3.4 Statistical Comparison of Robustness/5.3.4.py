"""
Complete code for Section 5.3.4: Statistical Comparison of Robustness
Table 5.8: Paired t-test results for robustness differences (daily frequency)
Bootstrap sampling with n = 1000 iterations
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
import random

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

# Statistics
from scipy import stats
import scipy.stats as scipy_stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print("=" * 60)
print("SECTION 5.3.4: STATISTICAL COMPARISON OF ROBUSTNESS")
print("Table 5.8: Paired t-test results with bootstrap sampling (n=1000)")
print("=" * 60)

# =============================================================================
# STEP 2: Download Daily Data with VIX
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Downloading Daily Data")
print("=" * 60)

# Download AAPL
aapl = yf.download('AAPL', start='2019-01-01', end='2025-12-31', progress=False)
if isinstance(aapl.columns, pd.MultiIndex):
    aapl.columns = aapl.columns.get_level_values(0)
aapl.columns = [col.capitalize() for col in aapl.columns]

# Download VIX for regime classification
vix = yf.download('^VIX', start='2019-01-01', end='2025-12-31', progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix = vix['Close'].rename('vix')

print(f"AAPL shape: {aapl.shape}")
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

# Add VIX
df['vix'] = vix

# Create features
for lag in [1, 2, 3, 5, 10]:
    df[f'lag_return_{lag}d'] = df['returns'].shift(lag)

# Rolling statistics
for window in [5, 10, 20]:
    df[f'returns_std_{window}d'] = df['returns'].rolling(window=window).std()
    df[f'returns_mean_{window}d'] = df['returns'].rolling(window=window).mean()

# Technical indicators (simplified)
df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close'] * 100
df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open'] * 100

# Target
df['target_return'] = df['returns'].shift(-1)
df['target_direction'] = (df['target_return'] > 0).astype(int)

# Clean data
df = df.dropna()
print(f"Final data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# =============================================================================
# STEP 4: Train/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Train/Test Split")
print("=" * 60)

# Use 70/30 split to have enough test data for bootstrap
split_idx = int(0.7 * len(df))
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Features
feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_return', 'target_direction', 'vix', 'Volume']]

X_train = train[feature_cols]
y_train = train['target_return']
X_test = test[feature_cols]
y_test = test['target_return']

print(f"Features: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 5: Define Models
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Training Models")
print("=" * 60)

models = {}
predictions = {}

# 1. Linear Regression (classical baseline)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
predictions['Linear Regression'] = lr.predict(X_test_scaled)
print("✓ Linear Regression")

# 2. Random Forest (classical)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
predictions['Random Forest'] = rf.predict(X_test_scaled)
print("✓ Random Forest")

# 3. Gradient Boosting (classical)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
predictions['Gradient Boosting'] = gb.predict(X_test_scaled)
print("✓ Gradient Boosting")

# 4. Neural Network (classical)
ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
ann.fit(X_train_scaled, y_train)
predictions['ANN'] = ann.predict(X_test_scaled)
print("✓ Neural Network")


# 5. Quantum Kernel (quantum-inspired)
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

        # Cosine similarity squared (quantum-inspired)
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
qk.fit(X_train_scaled, y_train.values)
predictions['Quantum Kernel'] = qk.predict(X_test_scaled)
print("✓ Quantum Kernel")


# 6. Amplitude Encoding (quantum-inspired)
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
ae.fit(X_train_scaled, y_train.values)
predictions['Amplitude Encoding'] = ae.predict(X_test_scaled)
print("✓ Amplitude Encoding")

print(f"\nTotal models: {len(predictions)}")

# =============================================================================
# STEP 6: Define Volatility Regimes
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Defining Volatility Regimes")
print("=" * 60)

# Use VIX percentile-based thresholds for balanced regimes
vix_33rd = test['vix'].quantile(0.33)
vix_67th = test['vix'].quantile(0.67)

print(f"Test period VIX thresholds:")
print(f"  33rd percentile: {vix_33rd:.2f}")
print(f"  67th percentile: {vix_67th:.2f}")

# Define regimes (Low, Medium, High)
test_regimes = pd.cut(test['vix'],
                      bins=[-np.inf, vix_33rd, vix_67th, np.inf],
                      labels=['Low', 'Medium', 'High'])

# For robustness comparison, we'll use Low and High (exclude Medium)
low_mask = test_regimes == 'Low'
high_mask = test_regimes == 'High'

print(f"\nTest samples by regime:")
print(f"  Low Volatility: {sum(low_mask)} days ({sum(low_mask) / len(test) * 100:.1f}%)")
print(f"  Medium Volatility: {sum(test_regimes == 'Medium')} days")
print(f"  High Volatility: {sum(high_mask)} days ({sum(high_mask) / len(test) * 100:.1f}%)")

# =============================================================================
# STEP 7: Calculate Instability Measure (Δ_RMSE) for Each Model
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Calculating Instability Measures")
print("=" * 60)


def calculate_delta_rmse(y_true, y_pred, low_mask, high_mask):
    """Calculate Δ_RMSE = |RMSE_high - RMSE_low|"""

    # RMSE for low volatility regime
    rmse_low = np.sqrt(mean_squared_error(
        y_true[low_mask], y_pred[low_mask]
    )) * 100  # Convert to basis points

    # RMSE for high volatility regime
    rmse_high = np.sqrt(mean_squared_error(
        y_true[high_mask], y_pred[high_mask]
    )) * 100

    # Instability measure
    delta_rmse = abs(rmse_high - rmse_low)

    return delta_rmse, rmse_low, rmse_high


# Calculate delta for each model
model_deltas = {}
model_rmse_low = {}
model_rmse_high = {}

for name, y_pred in predictions.items():
    delta, rmse_low, rmse_high = calculate_delta_rmse(
        y_test.values, y_pred, low_mask.values, high_mask.values
    )
    model_deltas[name] = delta
    model_rmse_low[name] = rmse_low
    model_rmse_high[name] = rmse_high

    print(f"{name:20s} | Δ_RMSE: {delta:.2f} bp | Low: {rmse_low:.2f} bp | High: {rmse_high:.2f} bp")

# =============================================================================
# STEP 8: Bootstrap Sampling for Statistical Testing
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Bootstrap Sampling (n = 1000 iterations)")
print("=" * 60)

n_bootstrap = 1000
n_test = len(y_test)

# Define model groups
quantum_models = ['Quantum Kernel', 'Amplitude Encoding']
classical_models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'ANN']

# Store bootstrap results
bootstrap_results = []

for b in range(n_bootstrap):
    if (b + 1) % 100 == 0:
        print(f"  Bootstrap iteration {b + 1}/{n_bootstrap}")

    # Sample with replacement from test indices
    bootstrap_idx = np.random.choice(n_test, n_test, replace=True)

    # Get bootstrap sample
    y_bootstrap = y_test.values[bootstrap_idx]
    low_mask_bootstrap = low_mask.values[bootstrap_idx]
    high_mask_bootstrap = high_mask.values[bootstrap_idx]

    # Skip if not enough samples in each regime
    if sum(low_mask_bootstrap) < 5 or sum(high_mask_bootstrap) < 5:
        continue

    # Calculate delta for each model on this bootstrap sample
    bootstrap_deltas = {}

    for name, y_pred in predictions.items():
        y_pred_bootstrap = y_pred[bootstrap_idx]
        delta, _, _ = calculate_delta_rmse(
            y_bootstrap, y_pred_bootstrap,
            low_mask_bootstrap, high_mask_bootstrap
        )
        bootstrap_deltas[name] = delta

    # Calculate mean delta for quantum and classical groups
    quantum_deltas = [bootstrap_deltas[m] for m in quantum_models if m in bootstrap_deltas]
    classical_deltas = [bootstrap_deltas[m] for m in classical_models if m in bootstrap_deltas]

    if quantum_deltas and classical_deltas:
        bootstrap_results.append({
            'iteration': b + 1,
            'quantum_mean': np.mean(quantum_deltas),
            'classical_mean': np.mean(classical_deltas),
            'difference': np.mean(classical_deltas) - np.mean(quantum_deltas)
        })

# Convert to DataFrame
bootstrap_df = pd.DataFrame(bootstrap_results)
print(f"\nCompleted {len(bootstrap_df)} valid bootstrap iterations")

# =============================================================================
# STEP 9: Paired t-test Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Paired t-test Results")
print("=" * 60)

# Perform paired t-test on bootstrap differences
t_stat, p_value = stats.ttest_1samp(bootstrap_df['difference'], 0)

# Calculate confidence intervals
ci_lower = np.percentile(bootstrap_df['difference'], 2.5)
ci_upper = np.percentile(bootstrap_df['difference'], 97.5)

print(f"\nBootstrap Results (n = {len(bootstrap_df)} iterations):")
print(f"  Mean difference (Classical - Quantum): {bootstrap_df['difference'].mean():.2f} bp")
print(f"  Std deviation: {bootstrap_df['difference'].std():.2f} bp")
print(f"  95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] bp")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.001:
    significance = "p < 0.001"
elif p_value < 0.01:
    significance = "p < 0.01"
elif p_value < 0.05:
    significance = "p < 0.05"
else:
    significance = "not significant"

print(f"\n  Significance: {significance}")

# =============================================================================
# STEP 10: Create Table 5.8 - Paired t-test Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Table 5.8")
print("=" * 60)

# Create comparison table for individual models
comparison_results = []

for q_model in quantum_models:
    for c_model in classical_models:
        if q_model in model_deltas and c_model in model_deltas:
            diff = model_deltas[c_model] - model_deltas[q_model]
            comparison_results.append({
                'Quantum Model': q_model,
                'Classical Model': c_model,
                'Δ_RMSE Quantum (bp)': f"{model_deltas[q_model]:.2f}",
                'Δ_RMSE Classical (bp)': f"{model_deltas[c_model]:.2f}",
                'Difference (bp)': f"{diff:.2f}",
                'More Robust': 'Quantum' if diff > 0 else 'Classical'
            })

comparison_df = pd.DataFrame(comparison_results)

# Create Table 5.8 (summary table)
table_5_8 = pd.DataFrame({
    'Comparison': ['Quantum vs Classical (all models)'],
    'Mean Δ Difference (bp)': [f"{bootstrap_df['difference'].mean():.2f}"],
    'Std Dev (bp)': [f"{bootstrap_df['difference'].std():.2f}"],
    't-statistic': [f"{t_stat:.3f}"],
    'p-value': [f"{p_value:.4f}"],
    'Significant (α = 0.05)?': ['Yes' if p_value < 0.05 else 'No']
})

print("\n" + "=" * 60)
print("TABLE 5.8: Paired t-test results for robustness differences (daily frequency)")
print("=" * 60)
print("\n")
print(table_5_8.to_string(index=False))

print("\n\nDetailed Model Comparisons:")
print("-" * 60)
print(comparison_df.to_string(index=False))

# Save results
table_5_8.to_csv('table_5_8_ttest_results.csv', index=False)
comparison_df.to_csv('table_5_8_detailed_comparisons.csv', index=False)
print(f"\n✓ Results saved to table_5_8_ttest_results.csv")

# LaTeX format for thesis
latex_filename = 'table_5_8_ttest_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Paired t-test results for robustness differences (daily frequency)}\n")
    f.write("\\label{tab:robustness_ttest}\n")
    f.write("\\begin{tabular}{lccccc}\n")
    f.write("\\hline\n")
    f.write("Comparison & Mean $\\Delta$ Difference (bp) & Std Dev (bp) & t-statistic & p-value & Significant? \\\\\n")
    f.write("\\hline\n")

    for _, row in table_5_8.iterrows():
        f.write(
            f"{row['Comparison']} & {row['Mean Δ Difference (bp)']} & {row['Std Dev (bp)']} & {row['t-statistic']} & {row['p-value']} & {row['Significant (α = 0.05)?']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 11: Create Visualization of Bootstrap Distribution
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Creating Visualization of Bootstrap Distribution")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Histogram of bootstrap differences
ax1 = axes[0]
ax1.hist(bootstrap_df['difference'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
ax1.axvline(x=bootstrap_df['difference'].mean(), color='green', linestyle='-', linewidth=2,
            label=f"Mean: {bootstrap_df['difference'].mean():.2f} bp")
ax1.axvline(x=ci_lower, color='orange', linestyle=':', linewidth=1.5, label='95% CI')
ax1.axvline(x=ci_upper, color='orange', linestyle=':', linewidth=1.5)

ax1.set_xlabel('Difference in Δ_RMSE (Classical - Quantum) [bp]', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Bootstrap Distribution of Robustness Differences', fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Bar chart of model deltas
ax2 = axes[1]
models_list = list(model_deltas.keys())
deltas_list = [model_deltas[m] for m in models_list]
colors = ['purple' if m in quantum_models else 'steelblue' for m in models_list]

bars = ax2.bar(models_list, deltas_list, color=colors, alpha=0.8)
ax2.set_xlabel('Model', fontsize=11)
ax2.set_ylabel('Δ_RMSE (basis points)', fontsize=11)
ax2.set_title('Instability Measure by Model (Lower is More Robust)', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, deltas_list):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure_5_3_4_bootstrap_ttest.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_3_4_bootstrap_ttest.pdf', bbox_inches='tight')
print("✓ Figure saved to figure_5_3_4_bootstrap_ttest.png and .pdf")

# =============================================================================
# STEP 12: Summary Statistics by Model Group
# =============================================================================

print("\n" + "=" * 60)
print("STEP 12: Summary Statistics by Model Group")
print("=" * 60)

quantum_deltas_list = [model_deltas[m] for m in quantum_models if m in model_deltas]
classical_deltas_list = [model_deltas[m] for m in classical_models if m in model_deltas]

print(f"\nQuantum-Inspired Models:")
print(f"  Count: {len(quantum_deltas_list)}")
print(f"  Mean Δ_RMSE: {np.mean(quantum_deltas_list):.2f} bp")
print(f"  Std Δ_RMSE: {np.std(quantum_deltas_list):.2f} bp")
print(f"  Min Δ_RMSE: {np.min(quantum_deltas_list):.2f} bp")
print(f"  Max Δ_RMSE: {np.max(quantum_deltas_list):.2f} bp")

print(f"\nClassical Models:")
print(f"  Count: {len(classical_deltas_list)}")
print(f"  Mean Δ_RMSE: {np.mean(classical_deltas_list):.2f} bp")
print(f"  Std Δ_RMSE: {np.std(classical_deltas_list):.2f} bp")
print(f"  Min Δ_RMSE: {np.min(classical_deltas_list):.2f} bp")
print(f"  Max Δ_RMSE: {np.max(classical_deltas_list):.2f} bp")

# Effect size (Cohen's d)
pooled_std = np.sqrt((np.std(quantum_deltas_list) ** 2 + np.std(classical_deltas_list) ** 2) / 2)
cohens_d = (np.mean(classical_deltas_list) - np.mean(quantum_deltas_list)) / pooled_std

print(f"\nEffect Size (Cohen's d): {cohens_d:.3f}")
if abs(cohens_d) < 0.2:
    effect = "negligible"
elif abs(cohens_d) < 0.5:
    effect = "small"
elif abs(cohens_d) < 0.8:
    effect = "medium"
else:
    effect = "large"
print(f"  Effect magnitude: {effect}")

# =============================================================================
# STEP 13: Final Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - SECTION 5.3.4 COMPLETE")
print("=" * 60)
print(f"\n✓ Bootstrap iterations: {len(bootstrap_df)}")
print(f"✓ Test period: {test.index.min()} to {test.index.max()}")
print(f"✓ Total test samples: {len(test)} days")
print(f"✓ Quantum models: {', '.join(quantum_models)}")
print(f"✓ Classical models: {', '.join(classical_models)}")
print(f"\n✓ Key Findings:")
print(f"  - Mean difference (Classical - Quantum): {bootstrap_df['difference'].mean():.2f} bp")
print(f"  - p-value: {p_value:.6f} ({significance})")
print(f"  - Effect size: {cohens_d:.3f} ({effect})")
print(f"\n✓ Table 5.8 saved to: table_5_8_ttest_results.csv")
print(f"✓ Figure saved to: figure_5_3_4_bootstrap_ttest.png")
print("\n" + "=" * 60)