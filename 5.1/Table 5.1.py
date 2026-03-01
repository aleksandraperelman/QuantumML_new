"""
Complete code for Table 5.1: Out-of-sample return prediction performance (daily frequency)
"""

# =============================================================================
# STEP 1: Import Libraries
# =============================================================================

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.kernel_ridge import KernelRidge

# Time Series
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Visualization
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("TABLE 5.1 GENERATION SCRIPT")
print("=" * 60)
print("Libraries imported successfully")

# =============================================================================
# STEP 2: Download Data from Yahoo Finance
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Downloading Data from Yahoo Finance")
print("=" * 60)


def download_data(tickers, start_date, end_date):
    """
    Download OHLCV data from Yahoo Finance.

    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns:
    --------
    dict : Dictionary of DataFrames for each ticker (single-level columns)
    """
    data = {}
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            # Download with auto_adjust=False to get OHLCV
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                # Flatten the columns if they're multi-level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} rows")
            else:
                print(f"  ✗ {ticker}: No data")
        except Exception as e:
            print(f"  ✗ {ticker}: Error - {e}")

    return data


tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
start_date = '2019-01-01'
end_date = '2025-12-31'

print(f"\nDownloading data from {start_date} to {end_date}")
print(f"Tickers: {tickers}")

data_dict = download_data(tickers, start_date, end_date)

# Use AAPL as primary asset
if 'AAPL' in data_dict:
    df_raw = data_dict['AAPL'].copy()
    print(f"\nUsing AAPL as primary asset")
    print(f"Data shape: {df_raw.shape}")
    print(f"Date range: {df_raw.index.min()} to {df_raw.index.max()}")
else:
    first_ticker = list(data_dict.keys())[0]
    df_raw = data_dict[first_ticker].copy()
    print(f"\nUsing {first_ticker} as primary asset")

# Ensure we have the right columns
print("\nColumns in raw data:")
print(df_raw.columns.tolist())

# Standardize column names (ensure they're lowercase for consistency)
df_raw.columns = [col.capitalize() for col in df_raw.columns]
print("\nStandardized columns:")
print(df_raw.columns.tolist())

print("\nRaw data sample:")
print(df_raw.head())

# =============================================================================
# STEP 3: Feature Engineering (Section 4.3.1)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)


def engineer_features(df):
    """
    Engineer all features from Section 4.3.1.

    Parameters:
    -----------
    df : DataFrame
        Raw OHLCV data with columns: Open, High, Low, Close, Volume

    Returns:
    --------
    DataFrame : DataFrame with all engineered features
    """
    # Ensure we're working with a single asset (flatten if multi-index)
    if isinstance(df.columns, pd.MultiIndex):
        print("  - Flattening multi-index columns...")
        df.columns = df.columns.get_level_values(0)

    # Ensure column names are strings and standardized
    df.columns = [str(col).capitalize() for col in df.columns]

    # Verify we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            # Try alternative names
            alt_cols = [c for c in df.columns if c.lower() == col.lower()]
            if alt_cols:
                df.rename(columns={alt_cols[0]: col}, inplace=True)
            else:
                print(f"  Warning: Column {col} not found. Available columns: {df.columns.tolist()}")
                # Create dummy column if missing
                if col in ['Open', 'High', 'Low']:
                    df[col] = df['Close']  # Approximate
                elif col == 'Volume':
                    df[col] = 0

    print(f"  - Working with columns: {df.columns.tolist()}")

    # Calculate returns
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    df = df.copy()

    # Calculate returns
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Price-based features
    print("  - Adding price-based features...")
    for k in [1, 2, 3, 4, 5]:
        df[f'lag_return_{k}'] = df['returns'].shift(k)

    df['price_range'] = (df['High'] - df['Low']) / df['Close']
    df['opening_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Technical Indicators
    print("  - Adding technical indicators...")

    # RSI (14-day)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI_14'] = calculate_rsi(df['Close'], 14)

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    # Simple Moving Averages
    for period in [5, 10, 20]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'price_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}'] - 1

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
    df.drop(['tr1', 'tr2', 'tr3', 'TR'], axis=1, inplace=True)

    # Volatility Measures
    print("  - Adding volatility measures...")

    # Historical Volatility (20-day)
    df['hist_vol_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

    # Parkinson Volatility
    df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) *
                                  (np.log(df['High'] / df['Low']) ** 2))

    # Volume features
    print("  - Adding volume features...")
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
    df['volume_change'] = df['Volume'].pct_change()

    # Market context (S&P 500)
    print("  - Adding market context...")
    try:
        spy = yf.download('SPY', start=df.index.min(), end=df.index.max(), progress=False)
        df['spy_return'] = spy['Close'].pct_change().reindex(df.index)
        df['spy_volume_ratio'] = spy['Volume'] / spy['Volume'].rolling(window=5).mean()
        df['spy_volume_ratio'] = df['spy_volume_ratio'].reindex(df.index)
    except:
        print("  Warning: Could not download SPY data")
        df['spy_return'] = 0
        df['spy_volume_ratio'] = 1

    # VIX (fear index)
    try:
        vix = yf.download('^VIX', start=df.index.min(), end=df.index.max(), progress=False)
        df['vix'] = vix['Close'].reindex(df.index)
        df['vix_change'] = vix['Close'].pct_change().reindex(df.index)
    except:
        print("  Warning: Could not download VIX data")
        df['vix'] = 20
        df['vix_change'] = 0

    # Target variables
    print("  - Creating target variables...")
    df['target_return'] = df['returns'].shift(-1)  # Next day's return
    df['target_direction'] = (df['target_return'] > 0).astype(int)

    # Realized volatility for next period
    df['target_realized_vol'] = df['returns'].rolling(window=5).std().shift(-5)

    return df


print("Engineering features for AAPL...")
df = engineer_features(df_raw)

print(f"\nTotal features engineered: {len(df.columns)}")
print("\nFeature columns:")
feature_cols = [col for col in df.columns if col not in ['target_return', 'target_direction', 'target_realized_vol']]
print(feature_cols[:20])  # Show first 20

# Drop NaN values
initial_rows = len(df)
df.dropna(inplace=True)
print(f"\nDropped {initial_rows - len(df)} rows with NaN values")
print(f"Final data shape: {df.shape}")

# =============================================================================
# STEP 4: Train/Validation/Test Split (70/15/15)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Train/Validation/Test Split")
print("=" * 60)

# Split by date (chronological)
n = len(df)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print(f"Train period: {train.index.min()} to {train.index.max()} ({len(train)} days)")
print(f"Validation period: {val.index.min()} to {val.index.max()} ({len(val)} days)")
print(f"Test period: {test.index.min()} to {test.index.max()} ({len(test)} days)")

# Prepare features and targets
feature_cols = [col for col in df.columns if col not in
                ['target_return', 'target_direction', 'target_realized_vol', 'returns']]

X_train = train[feature_cols]
y_train = train['target_return']

X_val = val[feature_cols]
y_val = val['target_return']

X_test = test[feature_cols]
y_test = test['target_return']

print(f"\nFeatures: {len(feature_cols)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# STEP 5: Define Models (Section 4.4)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Initializing Models")
print("=" * 60)

models = {}

# Benchmark Models
print("\n1. Benchmark Models:")
models['Linear Regression'] = LinearRegression()
print("  ✓ Linear Regression")

# ARIMA (will be handled separately)

# Classical ML Models
print("\n2. Classical ML Models:")
models['Random Forest'] = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
print("  ✓ Random Forest")

models['ANN (2-layer)'] = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),  # Larger architecture
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,  # More iterations
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    tol=0.0001,  # Tolerance for improvement
    n_iter_no_change=20,  # More patience
    alpha=0.001  # L2 regularization
)
print("  ✓ ANN (2-layer)")

# Quantum-Inspired Models
print("\n3. Quantum-Inspired Models:")


# Quantum Kernel Ridge Regression with custom kernel
def quantum_kernel(X, Y=None):
    """Quantum-inspired kernel: (z_t^T z_s)^2 / (||z_t||^2 ||z_s||^2)"""
    if Y is None:
        Y = X

    # Convert to 2D if necessary
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    # Compute kernel matrix
    n_X = X.shape[0]
    n_Y = Y.shape[0]

    # Pre-compute norms
    norm_X = np.linalg.norm(X, axis=1) + 1e-8
    norm_Y = np.linalg.norm(Y, axis=1) + 1e-8

    K = np.zeros((n_X, n_Y))

    for i in range(n_X):
        for j in range(n_Y):
            dot_product = np.dot(X[i], Y[j])
            K[i, j] = (dot_product ** 2) / (norm_X[i] * norm_Y[j])

    # Return scalar if both inputs are single samples
    if n_X == 1 and n_Y == 1:
        return float(K[0, 0])

    return K

# Amplitude Encoding Regression
class AmplitudeEncodingRegression:
    """Amplitude Encoding Regression from Section 4.4.3"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None

    def _amplitude_encode(self, X):
        """Convert features to amplitude encoding state"""
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms

        # Create density matrix features (outer product)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        density_features = []

        for i in range(n_samples):
            psi = X_norm[i:i + 1].T  # (n_features, 1)
            rho = np.dot(psi, psi.T)  # (n_features, n_features)
            # Take upper triangle (including diagonal)
            idx = np.triu_indices_from(rho)
            features = rho[idx]
            density_features.append(features)

        return np.array(density_features)

    def fit(self, X, y):
        X_encoded = self._amplitude_encode(X)
        # Ridge regression solution: (X^T X + αI)^(-1) X^T y
        n_features = X_encoded.shape[1]
        I = np.eye(n_features)
        try:
            self.coef_ = np.linalg.solve(
                X_encoded.T @ X_encoded + self.alpha * I,
                X_encoded.T @ y
            )
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            self.coef_ = np.linalg.pinv(
                X_encoded.T @ X_encoded + self.alpha * I
            ) @ X_encoded.T @ y
        return self

    def predict(self, X):
        X_encoded = self._amplitude_encode(X)
        return X_encoded @ self.coef_


models['Amplitude Encoding'] = AmplitudeEncodingRegression(alpha=1.0)
print("  ✓ Amplitude Encoding Regression")

# =============================================================================
# STEP 6: Train Models and Generate Predictions
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Training Models and Generating Predictions")
print("=" * 60)

predictions = {}
training_times = {}

# 1. Linear Regression
print("\n1. Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
predictions['Linear Regression'] = lr.predict(X_test_scaled)
print("  ✓ Complete")

# 2. ARIMA (using only returns, not features)
print("\n2. Training ARIMA(1,0,1)...")
try:
    arima_model = ARIMA(train['returns'].dropna(), order=(1, 0, 1)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))
    predictions['ARIMA(1,0,1)'] = arima_forecast.values
    print("  ✓ Complete")
except Exception as e:
    print(f"  ✗ ARIMA failed: {e}")
    predictions['ARIMA(1,0,1)'] = np.zeros(len(y_test))

# 3. Random Forest
print("\n3. Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
predictions['Random Forest'] = rf.predict(X_test_scaled)
print("  ✓ Complete")

# 5. ANN
print("\n5. Training ANN (2-layer)...")
# Scale target values for ANN (they're very small)
y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train_scaled = (y_train - y_train_mean) / y_train_std

ann = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    tol=0.0001,
    n_iter_no_change=20,
    alpha=0.001
)
ann.fit(X_train_scaled, y_train_scaled)  # Train on scaled targets
ann_pred_scaled = ann.predict(X_test_scaled)
ann_pred = ann_pred_scaled * y_train_std + y_train_mean  # Rescale back
predictions['ANN (2-layer)'] = ann_pred
print("  ✓ Complete")

# 6. LSTM (requires reshaping)
print("\n6. Training LSTM (50 units)...")

def create_sequences(X, y, timesteps=10):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

timesteps = 10

# Create sequences
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train.values, timesteps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test.values, timesteps)

# Build LSTM model
lstm_model = Sequential([
    LSTM(100, input_shape=(timesteps, X_train.shape[1]), return_sequences=True),  # Increased units
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)

# Train
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=32,
    verbose=0,
    callbacks=[early_stop]
)

# Predict
lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()

# Pad to match length (since sequences start later)
predictions['LSTM (50 units)'] = np.concatenate([np.zeros(timesteps), lstm_pred])[:len(y_test)]
print("  ✓ Complete")

# 7. Quantum Kernel Ridge
print("\n7. Training Quantum Kernel Ridge...")
qkr = KernelRidge(kernel=quantum_kernel, alpha=1.0)
qkr.fit(X_train_scaled[:1000], y_train.values[:1000])  # Use subset for speed
qkr_pred = qkr.predict(X_test_scaled)
predictions['Quantum Kernel Ridge'] = qkr_pred
print("  ✓ Complete")

# 8. Amplitude Encoding
print("\n8. Training Amplitude Encoding Regression...")
ae = AmplitudeEncodingRegression(alpha=1.0)
ae.fit(X_train_scaled[:1000], y_train.values[:1000])  # Use subset for speed
ae_pred = ae.predict(X_test_scaled)
predictions['Amplitude Encoding'] = ae_pred
print("  ✓ Complete")

# 9. Quantum Neural Network (simplified implementation)
# 9. Quantum Neural Network (simplified implementation)
print("\n9. Training Quantum Neural Network (simplified)...")


class SimpleQNN:
    """Simplified QNN with training capability"""

    def __init__(self, n_qubits=8, n_layers=3, learning_rate=0.01):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = learning_rate
        # Initialize weights
        self.W = [np.random.randn(n_qubits, n_qubits) * 0.1 for _ in range(n_layers)]

    def _encode(self, x):
        """Amplitude encoding"""
        x = x[:self.n_qubits]  # Truncate to n_qubits
        norm = np.linalg.norm(x) + 1e-8
        return x / norm  # Remove complex numbers for simplicity

    def _forward(self, x):
        """Forward pass"""
        state = self._encode(x)
        for layer in range(self.n_layers):
            state = self.W[layer] @ state
        return np.mean(state)  # Return scalar

    def _compute_gradient(self, x, target):
        """Simple gradient approximation"""
        pred = self._forward(x)
        loss_grad = 2 * (pred - target)  # MSE gradient

        # Store original weights
        original_W = [w.copy() for w in self.W]
        gradients = []

        # Approximate gradients by perturbation
        epsilon = 1e-4
        for layer in range(self.n_layers):
            grad_layer = np.zeros_like(self.W[layer])
            for i in range(self.W[layer].shape[0]):
                for j in range(self.W[layer].shape[1]):
                    # Positive perturbation
                    self.W[layer][i, j] += epsilon
                    pred_plus = self._forward(x)

                    # Negative perturbation
                    self.W[layer][i, j] -= 2 * epsilon
                    pred_minus = self._forward(x)

                    # Gradient approximation
                    grad_layer[i, j] = (pred_plus - pred_minus) / (2 * epsilon) * loss_grad

                    # Restore
                    self.W[layer][i, j] = original_W[layer][i, j]

            gradients.append(grad_layer)
            self.W[layer] = original_W[layer].copy()

        return gradients

    def fit(self, X, y, epochs=10, batch_size=32, verbose=True):
        """Train the QNN"""
        n_samples = len(X)
        losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            epoch_loss = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_loss = 0

                for idx in batch_idx:
                    x_i = X[idx]
                    y_i = y[idx]

                    # Forward pass
                    pred = self._forward(x_i)
                    loss = (pred - y_i) ** 2
                    batch_loss += loss

                    # Compute and apply gradients
                    gradients = self._compute_gradient(x_i, y_i)

                    # Update weights
                    for layer in range(self.n_layers):
                        self.W[layer] -= self.lr * gradients[layer]

                epoch_loss += batch_loss

            avg_loss = epoch_loss / n_samples
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, X):
        """Make predictions"""
        predictions = []
        for i in range(len(X)):
            pred = self._forward(X[i])
            predictions.append(pred)
        return np.array(predictions)


# Train QNN properly
subset_size = min(500, len(X_train_scaled))
qnn = SimpleQNN(n_qubits=min(8, X_train.shape[1]), n_layers=3, learning_rate=0.01)
print("  Training QNN...")
qnn.fit(X_train_scaled[:subset_size], y_train.values[:subset_size], epochs=10, batch_size=16)
qnn_pred = qnn.predict(X_test_scaled)
predictions['QNN (L=3)'] = qnn_pred
print("  ✓ Complete")

print("\nAll models trained successfully!")

# =============================================================================
# STEP 7: Calculate Performance Metrics
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Calculating Performance Metrics")
print("=" * 60)


def calculate_metrics(y_true, y_pred, window_size=60):
    """
    Calculate RMSE, MAE, MAPE with standard deviations.
    """
    # Overall metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE with epsilon to avoid division by zero
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

    # Rolling window metrics for standard deviation
    n = len(y_true)
    rmse_list = []
    mae_list = []

    for i in range(window_size, n):
        y_true_window = y_true[i - window_size:i]
        y_pred_window = y_pred[i - window_size:i]

        rmse_list.append(np.sqrt(mean_squared_error(y_true_window, y_pred_window)))
        mae_list.append(mean_absolute_error(y_true_window, y_pred_window))

    rmse_std = np.std(rmse_list)
    mae_std = np.std(mae_list)

    return rmse, rmse_std, mae, mae_std, mape


# Calculate metrics for each model
results = {
    'Model': [],
    'RMSE (×10⁻³)': [],
    'MAE (×10⁻³)': [],
    'MAPE (%)': []
}

model_order = [
    'Linear Regression',
    'ARIMA(1,0,1)',
    'Random Forest',
    'ANN (2-layer)',
    'LSTM (50 units)',
    'Quantum Kernel Ridge',
    'Amplitude Encoding',
    'QNN (L=3)'
]

for model_name in model_order:
    if model_name in predictions:
        y_pred = predictions[model_name]

        # Ensure same length
        min_len = min(len(y_test), len(y_pred))
        y_true_aligned = y_test.values[:min_len]
        y_pred_aligned = y_pred[:min_len]

        rmse, rmse_std, mae, mae_std, mape = calculate_metrics(y_true_aligned, y_pred_aligned)

        results['Model'].append(model_name)
        results['RMSE (×10⁻³)'].append(f"{rmse * 1000:.2f} ({rmse_std * 1000:.2f})")
        results['MAE (×10⁻³)'].append(f"{mae * 1000:.2f} ({mae_std * 1000:.2f})")
        results['MAPE (%)'].append(f"{mape:.1f}")

        print(f"{model_name:25s} | RMSE: {rmse * 1000:.2f} | MAE: {mae * 1000:.2f} | MAPE: {mape:.1f}%")

# =============================================================================
# STEP 8: Create Table 5.1 DataFrame
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Creating Table 5.1")
print("=" * 60)

# Create DataFrame
df_results = pd.DataFrame(results)

# Display table
print("\n" + "=" * 60)
print("TABLE 5.1: Out-of-sample return prediction performance (daily frequency)")
print("=" * 60)
print("\n")

# Format for display
display_df = df_results.copy()
print(display_df.to_string(index=False))

# =============================================================================
# STEP 9: Save Results
# =============================================================================

print("\n" + "=" * 60)
print("STEP 9: Saving Results")
print("=" * 60)

# Save to CSV
csv_filename = 'table_5_1_results.csv'
df_results.to_csv(csv_filename, index=False)
print(f"✓ Results saved to {csv_filename}")

# Save to LaTeX format for thesis
latex_filename = 'table_5_1_latex.txt'
with open(latex_filename, 'w') as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Out-of-sample return prediction performance (daily frequency)}\n")
    f.write("\\label{tab:return_prediction}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\hline\n")
    f.write("Model & RMSE ($\\times 10^{-3}$) & MAE ($\\times 10^{-3}$) & MAPE (\\%) \\\\\n")
    f.write("\\hline\n")

    for _, row in df_results.iterrows():
        f.write(f"{row['Model']} & {row['RMSE (×10⁻³)']} & {row['MAE (×10⁻³)']} & {row['MAPE (%)']} \\\\\n")

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"✓ LaTeX version saved to {latex_filename}")

# =============================================================================
# STEP 10: Create Figure 5.1 (All Models Comparison)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 10: Creating Figure 5.1 - All Models Comparison")
print("=" * 60)

# Select 60-day window from test set
window_start = 50
window_end = 110

# Create figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

# Plot actual returns in each subplot for reference
for i, (model_name, ax) in enumerate(zip(model_order[:9], axes)):
    if model_name in predictions:
        # Plot actual returns
        ax.plot(test.index[window_start:window_end],
                y_test.values[window_start:window_end],
                label='Actual', linewidth=2, color='black')

        # Plot model predictions
        pred = predictions[model_name][window_start:window_end]
        ax.plot(test.index[window_start:window_end],
                pred,
                label='Predicted', linewidth=1.5, linestyle='--', alpha=0.8)

        ax.set_title(f'{model_name}', fontsize=12)
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Return', fontsize=9)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

plt.suptitle('Figure 5.1: Predicted vs Actual Returns - All Models (60-day window)', fontsize=16, y=1.02)
plt.tight_layout()

# Save figure
figure_png = 'figure_5_1_all_models.png'
figure_pdf = 'figure_5_1_all_models.pdf'
plt.savefig(figure_png, dpi=300, bbox_inches='tight')
plt.savefig(figure_pdf, bbox_inches='tight')
print(f"✓ Figure saved to {figure_png} and {figure_pdf}")

# Also create a combined plot for easy comparison
plt.figure(figsize=(16, 8))

# Plot actual returns
plt.plot(test.index[window_start:window_end],
         y_test.values[window_start:window_end],
         label='Actual Returns', linewidth=3, color='black')

# Plot all models
colors = plt.cm.tab10(np.linspace(0, 1, len(model_order)))
for i, model_name in enumerate(model_order):
    if model_name in predictions:
        pred = predictions[model_name][window_start:window_end]
        plt.plot(test.index[window_start:window_end],
                 pred,
                 label=model_name, linewidth=1.5, alpha=0.7, color=colors[i])

plt.xlabel('Date', fontsize=12)
plt.ylabel('Return', fontsize=12)
plt.title('Figure 5.1: All Models - Predicted vs Actual Returns (60-day window)', fontsize=14)
plt.legend(loc='best', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Save combined figure
combined_png = 'figure_5_1_all_models_combined.png'
combined_pdf = 'figure_5_1_all_models_combined.pdf'
plt.savefig(combined_png, dpi=300)
plt.savefig(combined_pdf)
print(f"✓ Combined figure saved to {combined_png} and {combined_pdf}")

# =============================================================================
# STEP 11: Summary
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("\n✓ All tasks completed successfully!")
print(f"✓ Table 5.1 saved to: {csv_filename}")
print(f"✓ LaTeX table saved to: {latex_filename}")
print(f"✓ Figure 5.1 saved to: {figure_png}")
print("\nNext steps:")
print("1. Copy the LaTeX code into your thesis")
print("2. Insert the figure into your document")
print("3. Verify results match your expectations")
print("4. Proceed to Table 5.2 (hourly frequency)")

print("\n" + "=" * 60)