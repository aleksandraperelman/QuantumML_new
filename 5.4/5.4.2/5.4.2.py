"""
Complete code for Section 5.4.2: Performance under Feature Noise Injection
Figure 5.6: RMSE degradation curves as noise intensity increases
Noise levels: δ = 0.0 to 1.0 in increments of 0.2
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

# Statistics
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("SECTION 5.4.2: PERFORMANCE UNDER FEATURE NOISE INJECTION")
print("Figure 5.6: RMSE degradation curves with increasing noise")
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
# STEP 3: Create Feature Set (simplified from Section 4.3.1)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Creating Feature Set")
print("=" * 60)

df = aapl.copy()

# Calculate returns (percentage)
df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

# Price-based features
print("  - Adding price-based features...")
for k in [1, 2, 3, 5]:
    df[f'lag_return_{k}d'] = df['returns'].shift(k)

df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
df['opening_gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100

# Technical indicators (simplified)
print("  - Adding technical indicators...")


# RSI
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

# Moving averages
for period in [5, 10, 20]:
    df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
    df[f'price_to_SMA_{period}'] = (df['Close'] / df[f'SMA_{period}'] - 1) * 100

# Volatility measures
print("  - Adding volatility measures...")
df['volatility_10d'] = df['returns'].rolling(window=10).std() * np.sqrt(252)

# Volume features
print("  - Adding volume features...")
df['log_volume'] = np.log(df['Volume'] + 1)
df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()

# Target
df['target_return'] = df['returns'].shift(-1)

# Clean data
df = df.dropna()
print(f"Final dataset shape: {df.shape}")
print(f"Total features: {len(df.columns)}")

# =============================================================================
# STEP 4: Train/Test Split
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Train/Test Split")
print("=" * 60)

# Use 80/20 split
split_idx = int(0.8 * len(df))
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Features and target
feature_cols = [col for col in df.columns if col not in
                ['returns', 'target_return', 'Volume']]

X_train = train[feature_cols]
X_test = test[feature_cols]
y_train = train['target_return']
y_test = test['target_return']

print(f"Features: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features (clean, no noise)
scaler = StandardScaler()
X_train_scaled_clean = scaler.fit_transform(X_train)
X_test_scaled_clean = scaler.transform(X_test)

# Get feature standard deviations for noise scaling
feature_stds = X_train.std().values

# =============================================================================
# STEP 5: Define Models for Noise Injection Test
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Initializing Models")
print("=" * 60)

models = {}

# Classical models
models['Linear Regression'] = LinearRegression()
models['Random Forest'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
models['Gradient Boosting'] = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                                        random_state=42)
models['ANN'] = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)


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
# STEP 6: Train Models on Clean Data (Baseline)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Training Models on Clean Data (δ = 0.0)")
print("=" * 60)

baseline_rmse = {}

for name, model in models.items():
    print(f"  Training {name}...", end=' ')
    try:
        if name in ['Quantum Kernel', 'Amplitude Encoding']:
            model.fit(X_train_scaled_clean[:500], y_train.values[:500])
        else:
            model.fit(X_train_scaled_clean, y_train)

        y_pred = model.predict(X_test_scaled_clean)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 1000  # to ×10⁻³
        baseline_rmse[name] = rmse
        print(f"RMSE: {rmse:.2f}")
    except Exception as e:
        print(f"Failed: {e}")
        baseline_rmse[name] = np.nan

# =============================================================================
# STEP 7: Noise Injection Experiment
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Noise Injection Experiment")
print("Noise levels: δ = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0")
print("=" * 60)

# Noise levels to test
noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Store results
results = []

for delta in noise_levels:
    print(f"\n{'-' * 40}")
    print(f"Testing noise level δ = {delta:.1f}")
    print(f"{'-' * 40}")

    # Add Gaussian noise to test features
    # Noise = N(0, δ * σ_x) where σ_x is feature std
    noise = np.random.normal(0, delta, X_test_scaled_clean.shape) * feature_stds.reshape(1, -1)
    X_test_noisy = X_test_scaled_clean + noise

    for name, model in models.items():
        try:
            # Make predictions on noisy data
            y_pred = model.predict(X_test_noisy)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 1000

            # Calculate relative increase
            rel_increase = (rmse - baseline_rmse[name]) / baseline_rmse[name] * 100 if baseline_rmse[
                                                                                           name] > 0 else np.nan

            results.append({
                'Model': name,
                'Noise Level': delta,
                'RMSE': rmse,
                'Rel_Increase': rel_increase
            })

            print(f"  {name:20s} | RMSE: {rmse:.2f} | Increase: {rel_increase:6.2f}%")
        except Exception as e:
            print(f"  {name:20s} | Failed: {e}")
            results.append({
                'Model': name,
                'Noise Level': delta,
                'RMSE': np.nan,
                'Rel_Increase': np.nan
            })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# =============================================================================
# STEP 8: Create Pivot Table for Easy Viewing
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Noise Injection Results Summary")
print("=" * 60)

# Pivot for RMSE
rmse_pivot = results_df.pivot(index='Model', columns='Noise Level', values='RMSE')
rmse_pivot.columns = [f'δ={col:.1f}' for col in rmse_pivot.columns]

# Pivot for Relative Increase
inc_pivot = results_df.pivot(index='Model', columns='Noise Level', values='Rel_Increase')
inc_pivot.columns = [f'δ={col:.1f} (%)' for col in inc_pivot.columns]

# Combine
summary_df = pd.concat([rmse_pivot, inc_pivot], axis=1)

# Add baseline RMSE for reference
summary_df['Baseline'] = summary_df['δ=0.0']

# Sort by degradation at max noise
summary_df['Degradation_δ1.0'] = summary_df['δ=1.0'] - summary_df['Baseline']
summary_df = summary_df.sort_values('Degradation_δ1.0')

print("\nRMSE at different noise levels (×10⁻³):")
print("-" * 80)
print(rmse_pivot.round(2).to_string())

print("\nRelative Increase (%) at different noise levels:")
print("-" * 80)
print(inc_pivot.round(2).to_string())

# Save results
summary_df.to_csv('table_5_10_noise_injection_results.csv')
print(f"\n✓ Results saved to table_5_10_noise_injection_results.csv")

# =============================================================================
# STEP 9: Create Figure 5.6 - RMSE Degradation Curves
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Figure 5.6 - RMSE Degradation Curves")
print("=" * 60)

# Select models for the figure (highlight quantum vs classical)
selected_models = ['Linear Regression', 'Random Forest', 'Gradient Boosting',
                   'ANN', 'Quantum Kernel', 'Amplitude Encoding']

# Colors and styles
model_styles = {
    'Linear Regression': {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
    'Random Forest': {'color': 'green', 'linestyle': '-', 'marker': 's'},
    'Gradient Boosting': {'color': 'orange', 'linestyle': '-', 'marker': '^'},
    'ANN': {'color': 'brown', 'linestyle': '-', 'marker': 'd'},
    'Quantum Kernel': {'color': 'purple', 'linestyle': '--', 'marker': '*', 'linewidth': 2.5},
    'Amplitude Encoding': {'color': 'magenta', 'linestyle': '--', 'marker': 'X', 'linewidth': 2.5}
}

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Absolute RMSE
ax1 = axes[0]
for model in selected_models:
    model_data = results_df[results_df['Model'] == model]
    if not model_data.empty:
        style = model_styles.get(model, {'color': 'gray', 'linestyle': '-', 'marker': 'o'})
        ax1.plot(model_data['Noise Level'], model_data['RMSE'],
                 label=model,
                 color=style['color'],
                 linestyle=style['linestyle'],
                 marker=style['marker'],
                 linewidth=style.get('linewidth', 1.5),
                 markersize=6)

ax1.set_xlabel('Noise Level (δ)', fontsize=12)
ax1.set_ylabel('RMSE (×10⁻³)', fontsize=12)
ax1.set_title('Figure 5.6a: Absolute RMSE under Feature Noise Injection', fontsize=14)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(noise_levels)

# Right plot: Relative Increase
ax2 = axes[1]
for model in selected_models:
    model_data = results_df[results_df['Model'] == model]
    if not model_data.empty:
        style = model_styles.get(model, {'color': 'gray', 'linestyle': '-', 'marker': 'o'})
        ax2.plot(model_data['Noise Level'], model_data['Rel_Increase'],
                 label=model,
                 color=style['color'],
                 linestyle=style['linestyle'],
                 marker=style['marker'],
                 linewidth=style.get('linewidth', 1.5),
                 markersize=6)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax2.set_xlabel('Noise Level (δ)', fontsize=12)
ax2.set_ylabel('Relative RMSE Increase (%)', fontsize=12)
ax2.set_title('Figure 5.6b: Relative Degradation under Feature Noise Injection', fontsize=14)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(noise_levels)

plt.tight_layout()
plt.savefig('figure_5_6_noise_degradation.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_6_noise_degradation.pdf', bbox_inches='tight')
print("✓ Figure 5.6 saved to figure_5_6_noise_degradation.png and .pdf")

# =============================================================================
# STEP 10: Create Additional Visualization - Degradation at δ=1.0
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Degradation Comparison Bar Chart")
print("=" * 60)

# Get degradation at max noise
max_noise_data = results_df[results_df['Noise Level'] == 1.0].copy()
max_noise_data = max_noise_data.sort_values('Rel_Increase')

# Colors for quantum vs classical
quantum_models = ['Quantum Kernel', 'Amplitude Encoding']
colors = ['purple' if m in quantum_models else 'steelblue' for m in max_noise_data['Model']]

plt.figure(figsize=(12, 6))
bars = plt.bar(max_noise_data['Model'], max_noise_data['Rel_Increase'], color=colors, alpha=0.8)

plt.xlabel('Model', fontsize=12)
plt.ylabel('Relative RMSE Increase at δ=1.0 (%)', fontsize=12)
plt.title('Figure 5.6c: Model Degradation at Maximum Noise Level (δ=1.0)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, max_noise_data['Rel_Increase']):
    if not np.isnan(val):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure_5_6c_degradation_comparison.png', dpi=300)
plt.savefig('figure_5_6c_degradation_comparison.pdf')
print("✓ Figure 5.6c saved to figure_5_6c_degradation_comparison.png and .pdf")

# =============================================================================
# STEP 11: Statistical Comparison at Max Noise
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Statistical Comparison at δ=1.0")
print("=" * 60)

# Get degradation at max noise
quantum_degradation = max_noise_data[max_noise_data['Model'].isin(quantum_models)]['Rel_Increase'].values
classical_degradation = max_noise_data[~max_noise_data['Model'].isin(quantum_models)]['Rel_Increase'].values

print(f"\nQuantum models degradation at δ=1.0: {quantum_degradation}")
print(f"  Mean: {np.nanmean(quantum_degradation):.2f}%")
print(f"  Std: {np.nanstd(quantum_degradation):.2f}%")

print(f"\nClassical models degradation at δ=1.0: {classical_degradation}")
print(f"  Mean: {np.nanmean(classical_degradation):.2f}%")
print(f"  Std: {np.nanstd(classical_degradation):.2f}%")

# Perform t-test
t_stat, p_value = stats.ttest_ind(quantum_degradation, classical_degradation, nan_policy='omit')

print(f"\nt-test results:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.05:
    print(
        f"  ✓ Significant difference: Quantum models degrade {np.nanmean(classical_degradation) - np.nanmean(quantum_degradation):.2f}% less")
else:
    print("  ✗ No significant difference")

# =============================================================================
# STEP 12: Create Table 5.10 - Noise Injection Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 12: Creating Table 5.10 - Noise Injection Summary")
print("=" * 60)

# Create formatted table for thesis
table_data = []
for model in selected_models:
    model_data = results_df[results_df['Model'] == model]
    if not model_data.empty:
        row = {'Model': model}
        for delta in noise_levels:
            val = model_data[model_data['Noise Level'] == delta]['RMSE'].values
            if len(val) > 0 and not np.isnan(val[0]):
                row[f'δ={delta:.1f}'] = f"{val[0]:.2f}"
            else:
                row[f'δ={delta:.1f}'] = "N/A"

        # Add degradation at δ=1.0
        deg_val = model_data[model_data['Noise Level'] == 1.0]['Rel_Increase'].values
        if len(deg_val) > 0 and not np.isnan(deg_val[0]):
            row['Degradation (%)'] = f"{deg_val[0]:.1f}"
        else:
            row['Degradation (%)'] = "N/A"

        table_data.append(row)

table_df = pd.DataFrame(table_data)

print("\n" + "=" * 60)
print("TABLE 5.10: RMSE under Feature Noise Injection (×10⁻³)")
print("=" * 60)
print("\n")
print(table_df.to_string(index=False))

# Save table
table_df.to_csv('table_5_10_noise_injection.csv', index=False)
print(f"\n✓ Table 5.10 saved to table_5_10_noise_injection.csv")

# LaTeX format
latex_filename = 'table_5_10_noise_injection_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{RMSE under Feature Noise Injection ($\\times 10^{-3}$)}\n")
    f.write("\\label{tab:noise_injection}\n")
    f.write("\\begin{tabular}{lccccccc}\n")
    f.write("\\hline\n")
    f.write(
        "Model & $\\delta=0.0$ & $\\delta=0.2$ & $\\delta=0.4$ & $\\delta=0.6$ & $\\delta=0.8$ & $\\delta=1.0$ & Degradation (\\%) \\\\\n")
    f.write("\\hline\n")

    for _, row in table_df.iterrows():
        f.write(
            f"{row['Model']} & {row['δ=0.0']} & {row['δ=0.2']} & {row['δ=0.4']} & {row['δ=0.6']} & {row['δ=0.8']} & {row['δ=1.0']} & {row['Degradation (%)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 13: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - SECTION 5.4.2 COMPLETE")
print("=" * 60)
print(f"\n✓ Noise levels tested: {noise_levels}")
print(f"✓ Models evaluated: {len(models)}")
print(f"✓ Test period: {test.index.min()} to {test.index.max()} ({len(test)} days)")

print(f"\n✓ Key Findings:")
if not np.isnan(np.nanmean(quantum_degradation)):
    print(f"  - Quantum models average degradation at δ=1.0: {np.nanmean(quantum_degradation):.2f}%")
    print(f"  - Classical models average degradation at δ=1.0: {np.nanmean(classical_degradation):.2f}%")
    print(f"  - Difference: {np.nanmean(classical_degradation) - np.nanmean(quantum_degradation):.2f}%")

print(f"\n✓ Figure 5.6 saved to: figure_5_6_noise_degradation.png")
print(f"✓ Table 5.10 saved to: table_5_10_noise_injection.csv")
print("\n" + "=" * 60)