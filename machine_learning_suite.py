#%%
# Data & Finance Imports
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

# Machine Learning Imports
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
import joblib
import os
from keras.models import load_model

# BTQuant Imports
import backtrader as bt
import datetime as dt
import time
from dontcommit import connection_string, MSSQLData
from live_strategys.live_functions import *

# Set up date range and parameters
startdate = "2024-04-01"
enddate = "2024-08-08"
timeframe = "1m"
coin_name = "BTC" + "USDT_klines"
start_date = dt.datetime.strptime(startdate, "%Y-%m-%d")
end_date = dt.datetime.strptime(enddate, "%Y-%m-%d")
start_time = time.time()

# Fetch data from the database
df = pd.DataFrame(MSSQLData.get_data_from_db(connection_string, coin_name, timeframe, start_date, end_date))
elapsed_time = time.time() - start_time

# Early exit if no data is returned
if df.empty:
    print("No data returned from the database. Please check your query and date range.")
    exit()  # Stop execution if no data is found
else:
    print(f"Data extraction completed in {elapsed_time:.2f} seconds")
    print(f"Number of rows retrieved: {len(df)}")

#%% Technical Indicator Calculation
def moving_average(data, period, method='simple'):
    if method == 'simple':
        return data.rolling(window=period).mean()
    elif method == 'exponential':
        return data.ewm(span=period, adjust=False).mean()

def macd(data, fast=12, slow=26):
    ema_fast = moving_average(data, fast, 'exponential')
    ema_slow = moving_average(data, slow, 'exponential')
    return ema_fast - ema_slow

# Adding technical indicators to dataframe
df['SMA'] = moving_average(df['Close'], 50)
df['MACD'] = macd(df['Close'])
df['EMA'] = moving_average(df['Close'], 100, 'exponential')
df['ROC'] = df['Close'].pct_change(10)
df['RSI'] = moving_average((df['Close'].diff(1) > 0).astype(int), 14)

# Set binary target
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df.dropna()

# Feature and label split
X = df[['SMA', 'MACD', 'EMA', 'ROC', 'RSI']]
y = df['target']

# Train-test split (80% train, 20% test)
split = int(len(df) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for LSTM
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#%% Model Building Functions
def build_ann_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_lstm.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

#%% Model Save/Load Functionality
model_dir = "saved_models/"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

ann_model_path = model_dir + "ann_model.keras"
lstm_model_path = model_dir + "lstm_model.keras"
nb_model_path = model_dir + "nb_model.pkl"

# Save models
def save_models():
    ann_model.save(ann_model_path)
    lstm_model.save(lstm_model_path)
    joblib.dump(nb_model, nb_model_path)

# Load models
def load_models():
    if os.path.exists(ann_model_path) and os.path.exists(lstm_model_path) and os.path.exists(nb_model_path):
        ann_model = load_model(ann_model_path)
        lstm_model = load_model(lstm_model_path)
        nb_model = joblib.load(nb_model_path)
        return ann_model, lstm_model, nb_model
    else:
        print("Some models are missing. Training from scratch.")
        return None, None, None

# Try loading models
ann_model, lstm_model, nb_model = load_models()

# Train models if not loaded
if ann_model is None or lstm_model is None or nb_model is None:
    ann_model = build_ann_model()
    print("Training ANN model...")
    ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    lstm_model = build_lstm_model()
    print("Training LSTM model...")
    lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)

    nb_model = GaussianNB()
    print("Training Naive Bayes model...")
    nb_model.fit(X_train, y_train)

    save_models()

#%% Predictions
print("Making predictions...")
y_pred_ann = ann_model.predict(X_test)
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_nb = nb_model.predict(X_test)

# Apply binary threshold
y_pred_ann = (y_pred_ann > 0.5).astype(int)
y_pred_lstm = (y_pred_lstm > 0.5).astype(int)

# Classification report
print("ANN Classification Report:\n", classification_report(y_test, y_pred_ann))
print("LSTM Classification Report:\n", classification_report(y_test, y_pred_lstm))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

#%% Cross-validation for Keras models
def evaluate_keras_model(model_fn, X, y, n_splits=5, epochs=20, batch_size=32):
    kf = KFold(n_splits=n_splits)
    scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = model_fn()
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        scores.append(model.evaluate(X_test, y_test, verbose=0)[1])  # Accuracy
        
    return np.array(scores)

# Cross-validation for `scikit-learn` models
tscv = TimeSeriesSplit(n_splits=10)
nb_scores = cross_val_score(nb_model, X_train, y_train, cv=tscv, scoring='accuracy')

# Cross-validation for Keras models
ann_scores = evaluate_keras_model(build_ann_model, X_train, y_train, n_splits=5, epochs=20)
lstm_scores = evaluate_keras_model(build_lstm_model, X_train_lstm, y_train, n_splits=5, epochs=20)

# Plot cross-validation results
results = [ann_scores, lstm_scores, nb_scores]
names = ['ANN', 'LSTM', 'Naive Bayes']
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

#%% Backtrader Setup and Backtesting
class CustomPandasData(bt.feeds.PandasData):
    lines = ('target',)
    params = (('datetime', None), ('target', -1))

class ML_Strategy(bt.Strategy):
    params = (("predictions", None),)

    def __init__(self):
        self.prediction_idx = 0  # Initialize prediction index

    def next(self):
        # Ensure we are within the bounds of predictions array
        if self.prediction_idx < len(self.p.predictions):
            prediction = self.p.predictions[self.prediction_idx]
            if prediction == 1:
                self.buy(size=0.1)
            elif prediction == 0:
                if self.position:
                    self.sell(size=0.1)

        self.prediction_idx += 1  # Move to the next prediction

# Ensure the length of predictions matches the backtest data
model_predictions = y_pred_ann.flatten()  # Example using ANN predictions
if len(model_predictions) < len(df):
    padding_size = len(df) - len(model_predictions)
    model_predictions = np.concatenate((np.zeros(padding_size), model_predictions))
elif len(model_predictions) > len(df):
    model_predictions = model_predictions[:len(df)]

print(f"Model predictions length: {len(model_predictions)}")
print(f"Dataframe length: {len(df)}")

# # Backtest setup

def run_backtest(df, model_predictions, model_name):
    pd.set_option('display.max_columns', None)
    print(df.head())  # Print only the head of the DataFrame for verification

    cerebro = bt.Cerebro()
    cerebro.addstrategy(ML_Strategy, predictions=model_predictions)
    data = CustomPandasData(dataname=df)
    cerebro.adddata(data)

    # Starting conditions
    cerebro.broker.setcash(1_000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Backtest
    print(f"Backtesting {model_name}")
    print(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")
    cerebro.run()
    ending_value = cerebro.broker.getvalue()
    print(f"Ending Portfolio Value: {ending_value}")

    # Plotting
    cerebro.plot(style='candles', numfigs=2, volume=False, barup='lightgreen', bardown='red', plot_inline=True)
    
    return ending_value

# Flatten predictions
y_pred_ann = y_pred_ann.flatten()
y_pred_lstm = y_pred_lstm.flatten()
y_pred_nb = y_pred_nb.flatten()

# Print out predictions to verify they are different
print("ANN Predictions:", y_pred_ann[:50])
print("LSTM Predictions:", y_pred_lstm[:50])
print("Naive Bayes Predictions:", y_pred_nb[:50])

print(f"Length of DataFrame: {len(df)}")
print(f"Length of ANN Predictions: {len(y_pred_ann)}")
print(f"Length of LSTM Predictions: {len(y_pred_lstm)}")
print(f"Length of Naive Bayes Predictions: {len(y_pred_nb)}")

# Run backtest
ann_ending_value = run_backtest(df, y_pred_ann, "ANN")
lstm_ending_value = run_backtest(df, y_pred_lstm, "LSTM")
nb_ending_value = run_backtest(df, y_pred_nb, "Naive Bayes")

# Collect results for plotting
backtest_results = {
    'ANN': ann_ending_value,
    'LSTM': lstm_ending_value,
    'Naive Bayes': nb_ending_value
}

# Print backtest results
print("\nBacktest Results:")
for model, ending_value in backtest_results.items():
    print(f"{model}: Ending Portfolio Value = {ending_value}")