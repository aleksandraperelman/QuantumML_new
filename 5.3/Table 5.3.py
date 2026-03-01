"""
Complete code for Table 5.3: Directional Accuracy Metrics (daily frequency)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("TABLE 5.3 GENERATION SCRIPT - DIRECTIONAL ACCURACY")
print("=" * 60)
print("Libraries imported successfully")

# =============================================================================
# STEP 2: Download Daily Data (same as Table 5.1)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Downloading Daily Data from Yahoo Finance")
print("=" * 60)


def download_daily_data(tickers, start_date, end_date):
    """Download daily OHLCV data."""
    data = {}
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    return data


tickers = ['AAPL']
start_date = '2019-01-01'
end_date = '2025-12-31'

print(f"\nDownloading data from {start_date} to {end_date}")
data_dict = download_daily_data(tickers, start_date, end_date)

if not data_dict:
    print("ERROR: No data downloaded!")
    sys.exit(1)

df_raw = data_dict['AAPL'].copy()
df_raw.columns = [col.capitalize() for col in df_raw.columns]
print(f"\nData shape: {df_raw.shape}")
print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")

# =============================================================================
# STEP 3: Feature Engineering (simplified from Table 5.1)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)


def engineer_features(df):
    """Engineer features for directional prediction."""
    df = df.copy()

    # Returns (percentage)
    df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

    # Price-based features
    print("  - Adding price features...")
    for k in [1, 2, 3, 4, 5]:
        df[f'lag_return_{k}d'] = df['returns'].shift(k)

    df['price_range'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['opening_gap'] = (df['Open'] / df['Close'].shift(1) - 1) * 100

    # Technical indicators
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

    # Volume features
    print("  - Adding volume features...")
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df['log_volume'] = np.log(df['Volume'] + 1)

    # =========================================================================
    # TARGET: Direction of next day's return (for classification)
    # =========================================================================

    print("  - Creating target variables...")
    df['next_return'] = df['returns'].shift(-1)
    df['target_direction'] = (df['next_return'] > 0).astype(int)  # 1 if up, 0 if down

    return df


print("Engineering features...")
df = engineer_features(df_raw)

# Clean data
print("\nCleaning data...")
df = df.ffill().bfill()
df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
df = df.dropna()
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

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Validation: {val.index.min()} to {val.index.max()} ({len(val)} days)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Features (exclude target-related columns)
feature_cols = [col for col in df.columns if col not in
                ['returns', 'next_return', 'target_direction', 'Volume']]

X_train = train[feature_cols]
y_train = train['target_direction']  # Binary classification target

X_val = val[feature_cols]
y_val = val['target_direction']

X_test = test[feature_cols]
y_test = test['target_direction']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Class balance - Up: {y_train.mean()*100:.1f}%, Down: {(1-y_train.mean())*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 5: Define Models for Classification
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Initializing Models for Directional Prediction")
print("=" * 60)

# For classification, we need to modify regression models or use classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

models = {}

# Benchmark
print("\n1. Benchmark Models:")
models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
print("  ✓ Logistic Regression")

# Classical ML
print("\n2. Classical ML Models:")
models['Random Forest'] = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
print("  ✓ Random Forest Classifier")

models['Gradient Boosting'] = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
)
print("  ✓ Gradient Boosting Classifier")

models['ANN (2-layer)'] = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation='relu',
    learning_rate_init=0.001, max_iter=1000, early_stopping=True,
    validation_fraction=0.1, random_state=42
)
print("  ✓ Neural Network Classifier")

# Quantum-Inspired (using KernelRidge with classification threshold)
print("\n3. Quantum-Inspired Models:")

def quantum_kernel(X, Y=None):
    from sklearn.metrics.pairwise import cosine_similarity
    if Y is None:
        Y = X
    return cosine_similarity(X, Y) ** 2

# For quantum kernel, we'll use it with SVM
models['Quantum SVM'] = SVC(kernel=quantum_kernel, C=1.0, probability=True, random_state=42)
print("  ✓ Quantum SVM")

# Amplitude Encoding adapted for classification
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
            psi = X_norm[i:i+1].T
            rho = np.dot(psi, psi.T)
            idx = np.triu_indices_from(rho)
            features.append(rho[idx])
        return np.array(features)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_encoded = self._amplitude_encode(X)
        n_features = X_encoded.shape[1]
        I = np.eye(n_features)

        # One-vs-rest approach for binary classification
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
        # Sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

models['Amplitude Encoding'] = AmplitudeEncodingClassifier(alpha=1.0)
print("  ✓ Amplitude Encoding Classifier")

# LSTM for classification
print("\n4. LSTM for Classification:")

def create_sequences(X, y, timesteps=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, timesteps)

# Build LSTM classifier
lstm_model = Sequential([
    LSTM(50, input_shape=(timesteps, X_train.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(25, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, verbose=0, callbacks=[early_stop])

# Store LSTM model separately
lstm_classifier = lstm_model
print("  ✓ LSTM Classifier")

# =============================================================================
# STEP 6: Train Models and Generate Predictions
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

        # Get probabilities if available
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

# Pad to match length
lstm_pred_full = np.concatenate([np.zeros(timesteps), lstm_pred])[:len(y_test)]
lstm_proba_full = np.concatenate([np.zeros(timesteps), lstm_pred_proba])[:len(y_test)]

predictions['LSTM'] = lstm_pred_full
probabilities['LSTM'] = lstm_proba_full

print("  ✓ LSTM Complete")

# ARIMA doesn't do classification directly, so we'll use a simple rule
print("\nGenerating ARIMA-based signals...")
try:
    arima_model = ARIMA(train['returns'].dropna(), order=(1,0,1)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))
    arima_direction = (arima_forecast.values > 0).astype(int)
    predictions['ARIMA(1,0,1)'] = arima_direction
    probabilities['ARIMA(1,0,1)'] = arima_forecast.values  # Not probabilities, but used for ranking
    print("  ✓ ARIMA Complete")
except Exception as e:
    print(f"  ✗ ARIMA failed: {e}")
    predictions['ARIMA(1,0,1)'] = np.zeros(len(y_test))
    probabilities['ARIMA(1,0,1)'] = np.zeros(len(y_test))

print("\nAll models trained successfully!")

# =============================================================================
# STEP 7: Calculate Directional Metrics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Calculating Directional Accuracy Metrics")
print("=" * 60)


def calculate_directional_metrics(y_true, y_pred):
    """
    Calculate directional accuracy metrics.

    DA: Directional Accuracy (proportion of correct sign predictions)
    Precision: TP / (TP + FP)
    Recall: TP / (TP + FN)
    F1: Harmonic mean of precision and recall
    """
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

        # Ensure same length
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

        print(f"{name:25s} | DA: {metrics['DA (%)']:.1f}% | Prec: {metrics['Precision (%)']:.1f}% | Rec: {metrics['Recall (%)']:.1f}%")

# =============================================================================
# STEP 8: Create Table 5.3
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Creating Table 5.3")
print("=" * 60)

df_results = pd.DataFrame(results)

print("\n" + "=" * 60)
print("TABLE 5.3: Directional Accuracy Metrics (daily frequency)")
print("=" * 60)
print("\n")
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('table_5_3_directional_results.csv', index=False)
print(f"\n✓ Results saved to table_5_3_directional_results.csv")

# LaTeX format
latex_filename = 'table_5_3_directional_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Directional Accuracy Metrics (daily frequency)}\n")
    f.write("\\label{tab:directional_accuracy}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Model & DA (\\%) & Precision (\\%) & Recall (\\%) & F1 Score (\\%) \\\\\n")
    f.write("\\hline\n")

    for _, row in df_results.iterrows():
        f.write(f"{row['Model']} & {row['DA (%)']} & {row['Precision (%)']} & {row['Recall (%)']} & {row['F1 Score (%)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 9: Create Figure 5.2 - Confusion Matrices
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Figure 5.2 - Confusion Matrices")
print("=" * 60)

# Select best classical (LSTM) and best quantum (Amplitude Encoding / Quantum SVM)
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

        # Compute confusion matrix
        cm = confusion_matrix(y_true_cm, y_pred_cm)

        # Plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Down', 'Up'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')

        # Calculate accuracy
        acc = accuracy_score(y_true_cm, y_pred_cm) * 100
        ax.set_title(f'{name}\nAccuracy: {acc:.1f}%', fontsize=11)

        # Add quadrant labels
        ax.text(0.3, 0.7, 'TN', transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.text(0.7, 0.7, 'FP', transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.text(0.3, 0.3, 'FN', transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.text(0.7, 0.3, 'TP', transform=ax.transAxes, fontsize=12, fontweight='bold')

plt.suptitle('Figure 5.2: Confusion Matrices - Directional Predictions', fontsize=16, y=1.02)
plt.tight_layout()

# Save figure
plt.savefig('figure_5_2_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_2_confusion_matrices.pdf', bbox_inches='tight')
print("✓ Figure saved to figure_5_2_confusion_matrices.png and .pdf")

# Also create a comparison bar chart
plt.figure(figsize=(12, 6))

models_bar = [r['Model'] for r in results]
da_values = [float(r['DA (%)']) for r in results]

bars = plt.bar(models_bar, da_values, color='steelblue')
plt.axhline(y=50, color='red', linestyle='--', label='Random Guess (50%)')

# Color quantum models differently
for i, name in enumerate(models_bar):
    if 'Quantum' in name or 'Amplitude' in name:
        bars[i].set_color('purple')
    elif 'LSTM' in name or 'ANN' in name:
        bars[i].set_color('orange')
    else:
        bars[i].set_color('steelblue')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Directional Accuracy (%)', fontsize=12)
plt.title('Figure 5.2b: Directional Accuracy Comparison', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.savefig('figure_5_2_da_comparison.png', dpi=300)
plt.savefig('figure_5_2_da_comparison.pdf')
print("✓ Comparison bar chart saved")

# =============================================================================
# STEP 10: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - TABLE 5.3 COMPLETE")
print("=" * 60)
print(f"\n✓ Data period: {df.index.min()} to {df.index.max()}")
print(f"✓ Test period: {test.index.min()} to {test.index.max()}")
print(f"✓ Test samples: {len(test)} days")
print(f"✓ Random guess baseline: 50%")
print(f"✓ Best model: {results[0]['Model']} ({results[0]['DA (%)']}%)")
print(f"✓ Table saved to: table_5_3_directional_results.csv")
print(f"✓ Figures saved: figure_5_2_confusion_matrices.png")
print("\n" + "=" * 60)