import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetch historical data for a given symbol between start_date and end_date
def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Preprocess the data: Load CSV file, extract 'Close' prices, and create 'Target' column
def preprocess_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    data = data[['Close']]
    data['Target'] = data['Close'].shift(-1)  # 'Target' is the next day's closing price
    data = data.dropna()  # Drop rows with missing values
    return data

# Add additional features: Previous close price and daily return
def add_features(data):
    data['Prev_Close'] = data['Close'].shift(1)  # Previous day's closing price
    data['Return'] = data['Close'].pct_change()  # Daily return
    data = data.dropna()  # Drop rows with missing values
    return data

# Train a linear regression model on the features
def train_model(data):
    X = data[['Prev_Close', 'Return']]  # Features
    y = data['Target']  # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)  # Fit the model
    
    y_pred = model.predict(X_test)  # Predict on test set
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
    print(f'Mean Squared Error: {mse}')
    
    return model

# Plot actual vs predicted values
def plot_results(data, model):
    data['Predicted'] = model.predict(data[['Prev_Close', 'Return']])  # Predict values on full data
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual')
    plt.plot(data.index, data['Predicted'], label='Predicted', linestyle='--')
    plt.legend()
    plt.title('Forex Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Usage
symbol = 'GBPUSD=X'  # Currency pair
start_date = '2022-05-01'  # Start date
end_date = '2023-05-01'    # End date
data = fetch_data(symbol, start_date, end_date)  # Fetch data
data.to_csv('forex_data.csv')  # Save data to CSV

data = preprocess_data('forex_data.csv')  # Preprocess data
data = add_features(data)  # Add features
model = train_model(data)  # Train model
plot_results(data, model)  # Plot results
