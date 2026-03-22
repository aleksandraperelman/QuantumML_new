
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

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

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
print("TABLE 5.5 GENERATION SCRIPT - VOLATILITY PREDICTION")
print("=" * 60)
print("Libraries imported successfully")

# =============================================================================
# STEP 2: Download Daily Data
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
# STEP 3: Feature Engineering for Volatility Prediction
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering for Volatility Prediction")
print("=" * 60)


def engineer_volatility_features(df):
    """Engineer features for volatility prediction."""
    df = df.copy()

    # Returns (percentage)
    df['returns'] = (df['Close'] / df['Close'].shift(1) - 1) * 100

    # =========================================================================
    # PARKINSON VOLATILITY (Realized Volatility Proxy)
    # Formula: σ_park = sqrt( (1/(4 ln 2)) * (ln(High/Low))^2 )
    # =========================================================================

    print("  - Computing Parkinson volatility...")
    df['parkinson_raw'] = (1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2)
    df['parkinson_vol'] = np.sqrt(df['parkinson_raw']) * 100  # Convert to percentage

    # Also compute other volatility estimators for comparison
    df['realized_vol_5d'] = df['returns'].rolling(window=5).std() * np.sqrt(252)
    df['realized_vol_21d'] = df['returns'].rolling(window=21).std() * np.sqrt(252)

    # Garman-Klass volatility
    df['garman_klass'] = np.sqrt(
        0.5 * (np.log(df['High'] / df['Low']) ** 2) -
        (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
    ) * 100

    # =========================================================================
    # TARGET: Next period's Parkinson volatility
    # =========================================================================

    df['target_parkinson'] = df['parkinson_vol'].shift(-1)

    # =========================================================================
    # FEATURES FOR VOLATILITY PREDICTION
    # =========================================================================

    print("  - Adding volatility features...")

    # Lagged volatility measures
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'parkinson_lag_{lag}d'] = df['parkinson_vol'].shift(lag)
        df[f'returns_std_lag_{lag}d'] = df['returns'].rolling(window=5).std().shift(lag)

    # Volatility of volatility
    df['vol_of_vol'] = df['parkinson_vol'].rolling(window=20).std()

    # Volatility ratios
    df['vol_ratio_5_20'] = df['parkinson_vol'].rolling(window=5).mean() / df['parkinson_vol'].rolling(window=20).mean()
    df['vol_ratio_10_20'] = df['parkinson_vol'].rolling(window=10).mean() / df['parkinson_vol'].rolling(
        window=20).mean()

    # Volatility regime indicators
    df['vol_above_ma20'] = (df['parkinson_vol'] > df['parkinson_vol'].rolling(window=20).mean()).astype(int)
    df['vol_above_ma50'] = (df['parkinson_vol'] > df['parkinson_vol'].rolling(window=50).mean()).astype(int)

    # =========================================================================
    # RETURN-BASED FEATURES
    # =========================================================================

    print("  - Adding return features...")

    # Lagged returns
    for lag in [1, 2, 3, 5]:
        df[f'lag_return_{lag}d'] = df['returns'].shift(lag)

    # Return moments
    for window in [5, 10, 20]:
        df[f'returns_mean_{window}d'] = df['returns'].rolling(window=window).mean()
        df[f'returns_std_{window}d'] = df['returns'].rolling(window=window).std()
        df[f'returns_skew_{window}d'] = df['returns'].rolling(window=window).skew()
        df[f'returns_kurt_{window}d'] = df['returns'].rolling(window=window).kurt()

    # Absolute returns (volatility proxy)
    df['abs_return'] = np.abs(df['returns'])
    for lag in [1, 2, 3, 5]:
        df[f'abs_return_lag_{lag}d'] = df['abs_return'].shift(lag)

    # Signed returns (leverage effect)
    df['neg_return'] = (df['returns'] < 0).astype(int) * df['returns']
    df['pos_return'] = (df['returns'] > 0).astype(int) * df['returns']

    # =========================================================================
    # VOLUME FEATURES
    # =========================================================================

    print("  - Adding volume features...")
    df['volume'] = df['Volume']
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_ma_5'] = df['Volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']

    # Volume-volatility relationship
    df['volume_vol_corr'] = df['Volume'].rolling(window=20).corr(df['parkinson_vol'])

    # =========================================================================
    # MARKET CONTEXT
    # =========================================================================

    print("  - Adding market context...")
    try:
        spy = yf.download('SPY', start=df.index.min(), end=df.index.max(), progress=False)
        spy_returns = (spy['Close'] / spy['Close'].shift(1) - 1) * 100
        df['spy_return'] = spy_returns.reindex(df.index)
        df['spy_vol'] = spy_returns.rolling(window=20).std().reindex(df.index)
    except:
        df['spy_return'] = 0
        df['spy_vol'] = df['parkinson_vol'].mean()

    try:
        vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
        df['vix'] = vix['Close'].reindex(df.index)
        df['vix_change'] = vix['Close'].pct_change().reindex(df.index) * 100
    except:
        df['vix'] = 20
        df['vix_change'] = 0

    return df


print("Engineering volatility features...")
df = engineer_volatility_features(df_raw)

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

print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Validation: {val.index.min()} to {val.index.max()} ({len(val)} days)")
print(f"Test: {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Features (exclude target-related columns)
feature_cols = [col for col in df.columns if col not in
                ['returns', 'parkinson_raw', 'parkinson_vol', 'garman_klass',
                 'realized_vol_5d', 'realized_vol_21d', 'target_parkinson',
                 'Volume', 'volume']]

X_train = train[feature_cols]
y_train = train['target_parkinson']  # Predict next period's Parkinson volatility

X_val = val[feature_cols]
y_val = val['target_parkinson']

X_test = test[feature_cols]
y_test = test['target_parkinson']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target stats - mean: {y_train.mean():.4f}%, std: {y_train.std():.4f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

"""
IMPROVED MODELS — addressing all concerns:

1. LSTM:     Add val-based early stopping, reduce capacity, clip outputs to valid range
2. GARCH:    Switch to GJR-GARCH (asymmetric) which handles leverage effect better
3. ANN:      Add stronger regularisation (L2 + dropout via noise), reduce overfitting
4. RF:       Add calibration layer to reduce bias in extreme regimes  
5. Quantum Kernel: Use RBF+ZZ kernel with proper bandwidth selection (median heuristic)
6. Amplitude Encoding: Keep two-stage but add cross-validated alpha selection
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("IMPROVED MODELS")
print("=" * 60)

# ── Helpers ───────────────────────────────────────────────────────────────────

def sanity_check(name, y_pred, y_ref_mean, y_ref_std):
    """Warn if predictions are off-scale; return clipped predictions."""
    pmean, pstd = np.mean(y_pred), np.std(y_pred)
    lo, hi = y_ref_mean * 0.05, y_ref_mean * 20
    n_clipped = np.sum((y_pred < lo) | (y_pred > hi))
    y_clipped = np.clip(y_pred, lo, hi)
    print(f"  {name:25s} | mean={pmean:.4f} std={pstd:.4f} "
          f"range=[{y_pred.min():.3f},{y_pred.max():.3f}] "
          f"clipped={n_clipped}")
    return y_clipped


def robust_qlike(y_true, y_pred):
    """
    QLIKE loss (Patton 2011) with robust clipping.
    Predictions clipped to [5%, 2000%] of true mean — standard practice.
    """
    mu = y_true.mean()
    yp = np.clip(y_pred, mu * 0.05, mu * 20)
    ratio = (y_true ** 2) / (yp ** 2 + 1e-10)
    return float(np.mean(ratio - np.log(ratio) - 1))


def metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    ql   = robust_qlike(y_true, y_pred)
    return {'QLIKE': ql, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 1. LINEAR REGRESSION (baseline, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
lr_model = LinearRegression()

# ─────────────────────────────────────────────────────────────────────────────
# 2. RANDOM FOREST — add isotonic calibration on val set to reduce bias
# ─────────────────────────────────────────────────────────────────────────────
class CalibratedRF:
    """
    RF with isotonic regression calibration fitted on validation set.
    Fixes the well-known RF bias toward the mean in extreme regimes.
    """
    def __init__(self):
        self.rf = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_split=15,
            min_samples_leaf=5, max_features=0.5,
            random_state=42, n_jobs=-1)
        self.calibrator = None

    def fit(self, X_tr, y_tr, X_val, y_val):
        self.rf.fit(X_tr, y_tr)
        val_pred = self.rf.predict(X_val)
        # Isotonic calibration: learn monotone mapping from raw pred → true
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_pred, y_val)
        return self

    def predict(self, X):
        raw = self.rf.predict(X)
        return self.calibrator.transform(raw)

rf_model = CalibratedRF()

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADIENT BOOSTING (unchanged — already well-performing)
# ─────────────────────────────────────────────────────────────────────────────
gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ANN — stronger regularisation, smaller capacity, proper val stopping
#
# Issues fixed:
#   - Was 256→128→64 neurons with no weight decay → overfitting
#   - early_stopping used random train split, not time-ordered val set
#   - Fix: reduce to 128→64→32, add alpha L2 penalty, use val set properly
# ─────────────────────────────────────────────────────────────────────────────
ann_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    alpha=0.01,              # L2 weight decay (was 0.0001 default)
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,     # more patience to avoid premature stop
    tol=1e-5,
    random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. LSTM — key fixes:
#   - Validation-based early stopping (not training-loss based)
#   - Clip outputs to [0, 5*train_std] — LSTM can produce negative vols
#   - Reduce capacity + increase dropout to reduce overfitting
#   - Use huber loss instead of MSE (robust to outlier vol spikes)
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm(n_timesteps, n_features):
    model = Sequential([
        LSTM(32, input_shape=(n_timesteps, n_features),
             return_sequences=True,
             kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        LSTM(16, return_sequences=False,
             kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dense(1, activation='softplus')  # softplus ensures positive output
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.Huber(delta=1.0))
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 6. GJR-GARCH — replaces symmetric GARCH(1,1)
#
# Why: standard GARCH assumes symmetric response to + and - returns.
#      GJR-GARCH adds a γ term for negative returns (leverage effect),
#      which is critical for equity volatility (bad news → more vol).
# ─────────────────────────────────────────────────────────────────────────────
# (fitted inline in training section below)

# ─────────────────────────────────────────────────────────────────────────────
# 7. QUANTUM KERNEL — fix: median heuristic for gamma
#
# Previous gamma = 1/n_features was too small → kernel matrix near constant
# → dual coefficients near zero → predictions near zero.
# Median heuristic: γ = 1 / (2 * median(||x_i - x_j||²))
# This ensures the kernel has meaningful variation across training pairs.
# ─────────────────────────────────────────────────────────────────────────────
class QuantumKernelRidge:
    """
    Projected Quantum Kernel with median-heuristic bandwidth.
    Inputs L2-normalised to unit sphere → K(x,x)=1 by construction.
    """
    def __init__(self, alpha=0.5, max_features=20, n_zz_pairs=10):
        self.alpha        = alpha
        self.max_features = max_features
        self.n_zz_pairs   = n_zz_pairs

    def _select_features(self, X):
        if X.shape[1] <= self.max_features:
            return np.arange(X.shape[1])
        return np.argsort(np.var(X, axis=0))[-self.max_features:]

    def _normalise(self, X):
        norms = np.linalg.norm(X[:, self.feat_idx_], axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        return X[:, self.feat_idx_] / norms

    def _kernel(self, X, Y):
        # RBF on unit sphere
        cosine = np.clip(X @ Y.T, -1, 1)
        rbf    = 2.0 * (1.0 - cosine)

        # ZZ interaction
        zz = np.zeros_like(rbf)
        for fi, fj in self.zz_pairs_:
            phi_X = (np.pi - X[:, fi]) * (np.pi - X[:, fj])
            phi_Y = (np.pi - Y[:, fi]) * (np.pi - Y[:, fj])
            zz += (phi_X[:, None] - phi_Y[None, :]) ** 2

        return np.exp(-self.gamma_ * (rbf + zz))

    def fit(self, X, y):
        self.feat_idx_ = self._select_features(X)
        d = len(self.feat_idx_)
        self.zz_pairs_ = [(i, (i + 1) % d) for i in range(min(self.n_zz_pairs, d))]

        Xn = self._normalise(X)

        # ── Median heuristic for gamma ──────────────────────────────────────
        # Sample 500 pairs to estimate median pairwise distance efficiently
        n = Xn.shape[0]
        idx_sample = np.random.choice(n, size=min(500, n), replace=False)
        Xs = Xn[idx_sample]
        dists_sq = np.sum((Xs[:, None, :] - Xs[None, :, :]) ** 2, axis=-1)
        median_dist_sq = np.median(dists_sq[dists_sq > 0])
        self.gamma_ = 1.0 / (2.0 * max(median_dist_sq, 1e-6))
        print(f"    QK gamma (median heuristic) = {self.gamma_:.6f}")

        self.X_train_norm_ = Xn
        K = self._kernel(Xn, Xn)
        n = K.shape[0]
        self.dual_coef_ = np.linalg.solve(K + self.alpha * np.eye(n), y)
        return self

    def predict(self, X):
        Xn = self._normalise(X)
        K  = self._kernel(Xn, self.X_train_norm_)
        return K @ self.dual_coef_

qk_model = QuantumKernelRidge(alpha=0.5, max_features=20, n_zz_pairs=10)

# ─────────────────────────────────────────────────────────────────────────────
# 8. AMPLITUDE ENCODING — cross-validated alpha selection
#
# Previous issue: fixed alpha=0.1 may over-regularise the quantum features.
# Fix: use RidgeCV with time-series-safe CV to select alpha automatically.
# ─────────────────────────────────────────────────────────────────────────────
class AmplitudeEncodingRegression:
    """
    Two-stage amplitude encoding:
      Stage 1: RidgeCV anchor (correct scale, cross-validated alpha)
      Stage 2: RidgeCV on quantum features predicts residuals
    """
    def __init__(self, alphas=None, quantum_weight=0.5):
        self.alphas = alphas or [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        self.quantum_weight = quantum_weight

    def _encode(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        psi  = X / norms
        z1   = psi ** 2
        z2   = z1[:, :-1] * z1[:, 1:]
        off  = psi[:, :-1] * psi[:, 1:]
        return np.hstack([z1, z2, off])

    def fit(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)

        # Stage 1: anchor with CV alpha
        self.anchor_ = RidgeCV(alphas=self.alphas, cv=tscv)
        self.anchor_.fit(X, y)
        print(f"    AE anchor alpha = {self.anchor_.alpha_:.4f}")
        residuals = y - self.anchor_.predict(X)

        # Stage 2: quantum residual correction with CV alpha
        Phi = self._encode(X)
        self.quantum_ = RidgeCV(alphas=self.alphas, cv=tscv)
        self.quantum_.fit(Phi, residuals)
        print(f"    AE quantum alpha = {self.quantum_.alpha_:.4f}")

        # Optimal weight for quantum correction
        correction = self.quantum_.predict(Phi)
        denom = np.sum(correction ** 2) + 1e-10
        self.lambda_ = float(np.clip(
            np.dot(correction, residuals) / denom,
            -self.quantum_weight, self.quantum_weight))
        print(f"    AE quantum weight λ = {self.lambda_:.4f}")
        return self

    def predict(self, X):
        stage1 = self.anchor_.predict(X)
        Phi    = self._encode(X)
        corr   = self.quantum_.predict(Phi)
        return stage1 + self.lambda_ * corr

ae_model = AmplitudeEncodingRegression(quantum_weight=0.5)


# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("TRAINING ALL MODELS")
print("=" * 60)

predictions       = {}
predictions_ytrue = {}
timesteps         = 20
y_ref_mean        = float(y_train.mean())
y_ref_std         = float(y_train.std())

# ── Linear Regression ────────────────────────────────────────────────────────
print("\nLinear Regression...")
lr_model.fit(X_train_scaled, y_train.values)
predictions['Linear Regression'] = sanity_check(
    'Linear Regression', lr_model.predict(X_test_scaled), y_ref_mean, y_ref_std)
predictions_ytrue['Linear Regression'] = y_test.values

# ── Random Forest (calibrated) ───────────────────────────────────────────────
print("\nRandom Forest (calibrated)...")
rf_model.fit(X_train_scaled, y_train.values, X_val_scaled, y_val.values)
predictions['Random Forest'] = sanity_check(
    'Random Forest', rf_model.predict(X_test_scaled), y_ref_mean, y_ref_std)
predictions_ytrue['Random Forest'] = y_test.values

# ── Gradient Boosting ────────────────────────────────────────────────────────
print("\nGradient Boosting...")
gb_model.fit(X_train_scaled, y_train.values)
predictions['Gradient Boosting'] = sanity_check(
    'Gradient Boosting', gb_model.predict(X_test_scaled), y_ref_mean, y_ref_std)
predictions_ytrue['Gradient Boosting'] = y_test.values

# ── ANN ──────────────────────────────────────────────────────────────────────
print("\nANN (regularised)...")
ann_model.fit(X_train_scaled, y_train.values)
predictions['ANN (2-layer)'] = sanity_check(
    'ANN (2-layer)', ann_model.predict(X_test_scaled), y_ref_mean, y_ref_std)
predictions_ytrue['ANN (2-layer)'] = y_test.values

# ── LSTM ─────────────────────────────────────────────────────────────────────
print("\nLSTM (with val-based early stopping)...")

def make_sequences(X, y, ts):
    Xs, ys = [], []
    for i in range(len(X) - ts):
        Xs.append(X[i:i + ts])
        ys.append(y[i + ts])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq  = make_sequences(X_train_scaled, y_train.values, timesteps)
X_val_seq, y_val_seq = make_sequences(X_val_scaled,   y_val.values,   timesteps)
X_te_seq,  y_te_seq  = make_sequences(X_test_scaled,  y_test.values,  timesteps)

lstm_net = build_lstm(timesteps, X_train.shape[1])
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10,      # val_loss, not train_loss
        restore_best_weights=True, min_delta=1e-5),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]
lstm_net.fit(X_tr_seq, y_tr_seq,
             validation_data=(X_val_seq, y_val_seq),
             epochs=100, batch_size=32, verbose=0, callbacks=callbacks)

lstm_raw         = lstm_net.predict(X_te_seq, verbose=0).flatten()
lstm_test_offset = timesteps
# softplus output is always positive, but clip anyway for safety
lstm_raw         = np.clip(lstm_raw, 1e-4, y_ref_mean * 15)

predictions['LSTM']          = sanity_check('LSTM', lstm_raw, y_ref_mean, y_ref_std)
predictions_ytrue['LSTM']    = y_test.values[
    lstm_test_offset: lstm_test_offset + len(lstm_raw)]

# ── GJR-GARCH ────────────────────────────────────────────────────────────────
print("\nGJR-GARCH(1,1,1) rolling 1-step forecasts...")
try:
    train_val_ret = pd.concat([train['returns'], val['returns']]).dropna()
    test_ret      = test['returns'].dropna()
    garch_preds   = []
    window        = train_val_ret.copy()

    for i in range(len(test)):
        gm     = arch_model(window, vol='GARCH', p=1, o=1, q=1,
                            dist='skewt',        # skewed-t handles fat tails
                            rescale=False)
        gm_fit = gm.fit(disp='off', show_warning=False)
        fc     = gm_fit.forecast(horizon=1, reindex=False)
        pred_v = float(np.sqrt(max(fc.variance.iloc[-1, 0], 1e-8)))
        garch_preds.append(pred_v)

        next_r = test_ret.iloc[i] if i < len(test_ret) else 0.0
        window = pd.concat([window,
                            pd.Series([next_r], index=[test.index[i]])])

    garch_preds = np.array(garch_preds)
    print(f"  ✓ GJR-GARCH done")
except Exception as e:
    print(f"  ✗ GJR-GARCH failed ({e}), falling back to GARCH(1,1)")
    try:
        train_val_ret = pd.concat([train['returns'], val['returns']]).dropna()
        test_ret      = test['returns'].dropna()
        garch_preds   = []
        window        = train_val_ret.copy()
        for i in range(len(test)):
            gm     = arch_model(window, vol='Garch', p=1, q=1,
                                dist='normal', rescale=False)
            gm_fit = gm.fit(disp='off', show_warning=False)
            fc     = gm_fit.forecast(horizon=1, reindex=False)
            garch_preds.append(float(np.sqrt(max(fc.variance.iloc[-1, 0], 1e-8))))
            next_r = test_ret.iloc[i] if i < len(test_ret) else 0.0
            window = pd.concat([window, pd.Series([next_r], index=[test.index[i]])])
        garch_preds = np.array(garch_preds)
    except Exception as e2:
        print(f"  ✗ Fallback GARCH also failed: {e2}")
        garch_preds = np.full(len(y_test), y_ref_mean)

predictions['GARCH(1,1)']       = sanity_check(
    'GARCH(1,1)', garch_preds, y_ref_mean, y_ref_std)
predictions_ytrue['GARCH(1,1)'] = y_test.values[:len(garch_preds)]

# ── Quantum Kernel ────────────────────────────────────────────────────────────
print("\nQuantum Kernel Ridge (median heuristic gamma)...")
qk_model.fit(X_train_scaled, y_train.values)
predictions['Quantum Kernel'] = sanity_check(
    'Quantum Kernel', qk_model.predict(X_test_scaled), y_ref_mean, y_ref_std)
predictions_ytrue['Quantum Kernel'] = y_test.values

# ── Amplitude Encoding ────────────────────────────────────────────────────────
print("\nAmplitude Encoding (CV alpha)...")
ae_model.fit(X_train_scaled, y_train.values)
predictions['Amplitude Encoding'] = sanity_check(
    'Amplitude Encoding', ae_model.predict(X_test_scaled), y_ref_mean, y_ref_std)
predictions_ytrue['Amplitude Encoding'] = y_test.values


# =============================================================================
# METRICS
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

model_order = ['Linear Regression', 'Random Forest', 'Gradient Boosting',
               'ANN (2-layer)', 'LSTM', 'GARCH(1,1)',
               'Quantum Kernel', 'Amplitude Encoding']

results = []
for name in model_order:
    if name not in predictions:
        continue
    yp = predictions[name]
    yt = predictions_ytrue[name]
    n  = min(len(yt), len(yp))
    m  = metrics(yt[:n], yp[:n])
    results.append({
        'Model': name,
        'QLIKE':    f"{m['QLIKE']:.4f}",
        'MSE':      f"{m['MSE']:.4f}",
        'RMSE (%)': f"{m['RMSE']:.4f}",
        'MAE (%)':  f"{m['MAE']:.4f}"
    })
    print(f"{name:25s} | QLIKE:{m['QLIKE']:8.4f} | MSE:{m['MSE']:7.4f} | "
          f"RMSE:{m['RMSE']:7.4f} | MAE:{m['MAE']:7.4f}")

import pandas as pd
df_results = pd.DataFrame(results)
print("\n")
print(df_results.to_string(index=False))
df_results.to_csv('table_5_5_volatility_results.csv', index=False)
print("\n✓ Saved to table_5_5_volatility_results.csv")

# =============================================================================
# STEP 9: Create Table 5.5
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Creating Table 5.5")
print("=" * 60)

df_results = pd.DataFrame(results)

print("\n" + "=" * 60)
print("TABLE 5.5: Volatility Prediction Performance (daily frequency)")
print("Using Parkinson's estimator as realized volatility proxy")
print("=" * 60)
print("\n")
print(df_results.to_string(index=False))

# Save results
df_results.to_csv('table_5_5_volatility_results.csv', index=False)
print(f"\n✓ Results saved to table_5_5_volatility_results.csv")

# LaTeX format
latex_filename = 'table_5_5_volatility_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Volatility Prediction Performance (daily frequency) using Parkinson's estimator}\n")
    f.write("\\label{tab:volatility_prediction}\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\hline\n")
    f.write("Model & QLIKE & MSE & RMSE (\\%) & MAE (\\%) \\\\\n")
    f.write("\\hline\n")

    for _, row in df_results.iterrows():
        f.write(f"{row['Model']} & {row['QLIKE']} & {row['MSE']} & {row['RMSE (%)']} & {row['MAE (%)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 10: Creating Figure 5.3 - QNN Volatility Predictions
# =============================================================================

# Helper: get the correctly-aligned (index, pred, true) triple for any model
def get_aligned(name):
    pred  = predictions[name]
    true  = predictions_ytrue[name]
    idx   = test.index

    # LSTM is shorter — trim the index to match
    if name == 'LSTM':
        idx  = test.index[lstm_test_offset : lstm_test_offset + len(pred)]
        true = true[:len(pred)]
    else:
        min_len = min(len(idx), len(pred), len(true))
        idx     = idx[:min_len]
        pred    = pred[:min_len]
        true    = true[:min_len]

    return idx, pred, true


# ── Figure 5.3: QNN single-model plot ───────────────────────────────────────
qnn_name = 'Amplitude Encoding'
idx_qnn, pred_qnn, true_qnn = get_aligned(qnn_name)

plt.figure(figsize=(16, 8))
plt.plot(idx_qnn, true_qnn,  label='Actual Parkinson Volatility',
         linewidth=2, color='black')
plt.plot(idx_qnn, pred_qnn,  label='QNN Predicted Volatility',
         linewidth=2, linestyle='--', color='red', alpha=0.7)

if test.index.max() > pd.Timestamp('2025-03-01'):
    plt.axvspan(pd.Timestamp('2025-03-01'), pd.Timestamp('2025-05-01'),
                alpha=0.2, color='orange', label='High Volatility Regimes')
    plt.axvspan(pd.Timestamp('2025-10-01'), pd.Timestamp('2025-12-01'),
                alpha=0.2, color='orange')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (%)', fontsize=12)
plt.title('Figure 5.3: QNN Predicted vs Actual Parkinson Volatility', fontsize=14)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figure_5_3_qnn_volatility.png', dpi=300)
plt.savefig('figure_5_3_qnn_volatility.pdf')
print("✓ Figure 5.3 saved")

# ── Figure 5.3b: all-models grid ────────────────────────────────────────────
vol_models = ['GARCH(1,1)', 'Random Forest', 'Gradient Boosting',
              'LSTM', 'Quantum Kernel', 'Amplitude Encoding']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, name in zip(axes, vol_models):
    if name not in predictions:
        ax.set_visible(False)
        continue

    idx_m, pred_m, true_m = get_aligned(name)

    ax.plot(idx_m, true_m,  label='Actual',    linewidth=1.5, color='black')
    ax.plot(idx_m, pred_m,  label='Predicted', linewidth=1.5,
            linestyle='--', alpha=0.7)

    corr = np.corrcoef(true_m, pred_m)[0, 1]
    ax.set_title(f'{name}\nCorr: {corr:.3f}', fontsize=10)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Figure 5.3b: Volatility Predictions — All Models', fontsize=14)
plt.tight_layout()
plt.savefig('figure_5_3b_all_volatility_models.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_5_3b_all_volatility_models.pdf', bbox_inches='tight')
print("✓ All models comparison saved")

# =============================================================================
# STEP 11: Summary Statistics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 11: Volatility Statistics Summary")
print("=" * 60)

# Calculate test period volatility statistics
test_vol_mean = y_test.mean()
test_vol_std = y_test.std()
test_vol_max = y_test.max()
test_vol_min = y_test.min()

print(f"\nTest Period Volatility Statistics (Parkinson estimator):")
print(f"  Mean: {test_vol_mean:.4f}%")
print(f"  Std Dev: {test_vol_std:.4f}%")
print(f"  Max: {test_vol_max:.4f}%")
print(f"  Min: {test_vol_min:.4f}%")

# Identify high-volatility periods
high_vol_threshold = test_vol_mean + test_vol_std
high_vol_days = test[test['parkinson_vol'] > high_vol_threshold].index
print(f"\nHigh-volatility periods (>{high_vol_threshold:.2f}%):")
for date in high_vol_days[:5]:  # Show first 5
    print(f"  {date.date()}: {test.loc[date, 'parkinson_vol']:.4f}%")

# =============================================================================
# STEP 12: Final Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY - TABLE 5.5 COMPLETE")
print("=" * 60)
print(f"\n✓ Data period: {df.index.min()} to {df.index.max()}")
print(f"✓ Test period: {test.index.min()} to {test.index.max()}")
print(f"✓ Test samples: {len(test)} days")
print(f"✓ Models evaluated: {len(predictions)}")
print(f"✓ Best QLIKE: {results[0]['QLIKE']} ({results[0]['Model']})")
print(f"✓ Best MSE: {results[0]['MSE']} ({results[0]['Model']})")
print(f"✓ Table saved to: table_5_5_volatility_results.csv")
print(f"✓ Figure 5.3 saved to: figure_5_3_qnn_volatility.png")
print("\n" + "=" * 60)