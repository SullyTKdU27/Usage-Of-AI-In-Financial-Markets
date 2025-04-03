import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class EnhancedTrader:
    def __init__(self, symbol, start_date, end_date, initial_capital=10000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.position = 0
        self.data = None
        self.prediction_model = None
        self.scaler = StandardScaler()
        self.load_data()
        
    def load_data(self):
        """Load stock data from Yahoo Finance and calculate additional indicators"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date)
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Calculate technical indicators
            self.calculate_indicators()
            
            print(f"Successfully loaded data for {self.symbol}")
            print(f"Data shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def calculate_indicators(self):
        """Calculate various technical indicators"""
        # Price-based indicators
        rolling_windows = [7, 20, 50]
        for window in rolling_windows:
            self.data[f'SMA{window}'] = self.data['Close'].rolling(window=window).mean()
        
        self.data['EMA12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        self.data['MACD'] = self.data['EMA12'] - self.data['EMA26']
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        self.data['BB_middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_upper'] = self.data['BB_middle'] + (bb_std * 2)
        self.data['BB_lower'] = self.data['BB_middle'] - (bb_std * 2)
        
        # RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Check for NaN values in RSI
        print(f"NaN values in RSI: {self.data['RSI'].isna().sum()}")  # Debugging line

        # Volatility
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        self.data['Volatility_MA'] = self.data['Volatility'].rolling(window=10).mean()
        
        # Volume indicators
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # Price momentum
        self.data['ROC'] = self.data['Close'].pct_change(periods=12) * 100
        self.data['Price_Momentum'] = self.data['Close'].diff(periods=10) / self.data['Close'].shift(10)
        
        self.data = self.data.dropna()
    
    def prepare_prediction_features(self, window_size=30):
        """Prepare features for prediction model with enhanced feature engineering"""
        features = []
        targets = []
        
        feature_columns = [
            'Close', 'SMA7', 'SMA20', 'SMA50', 'MACD', 'RSI', 
            'Volatility', 'Volume_Ratio', 'ROC', 'Price_Momentum'
        ]
        
        for i in range(window_size, len(self.data) - 1):
            # Create feature window
            feature_window = np.column_stack ([
                self.data[col].iloc[i-window_size:i] for col in feature_columns
            ])
            
            # Add derived features
            bb_position = (self.data['Close'].iloc[i] - self.data['BB_lower'].iloc[i]) / \
                           (self.data['BB_upper'].iloc[i] - self.data['BB_lower'].iloc[i])
            macd_hist = self.data['MACD'].iloc[i] - self.data['Signal_Line'].iloc[i]
            
            # Combine all features
            feature_vector = np.concatenate([
                feature_window.flatten(),
                [bb_position, macd_hist]
            ])
            
            features.append(feature_vector)
            targets.append(self.data['Close'].iloc[i+1])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled, np.array(targets)
    
    def train_prediction_model(self):
        """Train an enhanced prediction model using Random Forest"""
        X, y = self.prepare_prediction_features()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.prediction_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.prediction_model.fit(X_train, y_train)
        
        # Calculate predictions and errors
        train_score = self.prediction_model.score(X_train, y_train)
        test_score = self.prediction_model.score(X_test, y_test)
        
        print(f"Training R² score: {train_score:.4f}")
        print(f"Testing R² score: {test_score:.4f}")
        
        return test_score
    
    def predict_next_price(self):
        """Predict the next day's price with confidence interval"""
        if self.prediction_model is None:
            self.train_prediction_model()
        
        last_features, _ = self.prepare_prediction_features()
        if len(last_features) > 0:
            # Get predictions from all trees in the forest
            predictions = np.array([tree.predict(last_features[-1].reshape(1, -1))
                                  for tree in self.prediction_model.estimators_])
            
            # Calculate mean and confidence interval
            mean_pred = predictions.mean()
            conf_interval = np.percentile(predictions, [5, 95])
            
            return mean_pred, conf_interval
        return None, None
    
    def plot_analysis(self, portfolio_values):
        """Plot only the two desired graphs: Price Analysis and Portfolio Value"""
        plt.figure(figsize=(15, 10))

        # Plot 1: Price Analysis with Prediction
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Price', color='blue', alpha=0.7)
        plt.plot(self.data.index, self.data['BB_upper'], 'k--', alpha=0.3)
        plt.plot(self.data.index, self.data['BB_middle'], 'k-', alpha=0.3)
        plt.plot(self.data.index, self.data['BB_lower'], 'k--', alpha=0.3)
        plt.fill_between(self.data.index, self.data['BB_upper'], self.data['BB_lower'], 
                         alpha=0.1, color='gray')

        # Add prediction with confidence interval
        last_date = self.data.index[-1]
        num_days = 365
        predicted_prices = [self.data['Close'].iloc[-1]]  # Start with the last known price
        volatility = 0.005  # Reduced daily volatility to 0.5%

        # Simulate future prices
        for _ in range(num_days):
            daily_return = np.random.normal(loc=0, scale=volatility)  # Mean = 0, Std Dev = volatility
            new_price = predicted_prices[-1] * (1 + daily_return)  # Update price
            predicted_prices.append(new_price)

        # Create dates for the next 180 days
        predicted_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

        # Plot the predicted prices
        plt.plot(predicted_dates, predicted_prices[1:], 'r--', label='Prediction')
        plt.fill_between(predicted_dates,
                         np.array(predicted_prices[1:]) * (1 - volatility),  # Lower bound
                         np.array(predicted_prices[1:]) * (1 + volatility),  # Upper bound
                         color='red', alpha=0.1)

        plt.title(f'{self.symbol} Price Analysis with Prediction')
        plt.legend()
        plt.grid(True)

        # Plot 2: Portfolio Performance
        plt.subplot(2, 1,  2)
        plt.plot(self.data.index, portfolio_values, label='Portfolio Value', color='green')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def run_backtest(self):
        """Run a backtest based on the predicted prices and calculate portfolio values."""
        portfolio_values = []
        trades = []
        
        # Start with initial capital
        current_capital = self.capital
        self.position = 0  # No initial position
        
        for i in range(len(self.data) - 1):
            # Get the predicted price for the next day
            predicted_price, _ = self.predict_next_price()
            
            if predicted_price is not None:
                # If we predict the price will go up, buy
                if predicted_price > self.data['Close'].iloc[i]:
                    # Buy one share
                    if current_capital >= self.data['Close'].iloc[i]:
                        self.position += 1
                        current_capital -= self.data['Close'].iloc[i]
                        trades.append(('BUY', self.data.index[i], self.data['Close'].iloc[i]))
                # If we predict the price will go down, sell
                elif predicted_price < self.data['Close'].iloc[i]:
                    # Sell one share if we have one
                    if self.position > 0:
                        self.position -= 1
                        current_capital += self.data['Close'].iloc[i]
                        trades.append(('SELL', self.data.index[i], self.data['Close'].iloc[i]))
        
            # Calculate current portfolio value
            current_value = current_capital + self.position * self.data['Close'].iloc[i]
            portfolio_values.append(current_value)
        
        # Final portfolio value
        final_value = current_capital + self.position * self.data['Close'].iloc[-1]
        portfolio_values.append(final_value)
        
        return portfolio_values, trades

def main():
    # Testing the enhanced trader
    symbol = 'AAPL'
    start_date = '2023-06-01'
    end_date = '2024-06-01'
    initial_capital = 10000
    
    try:
        trader = EnhancedTrader(symbol, start_date, end_date, initial_capital)
        trader.train_prediction_model()
        portfolio_values, trades = trader.run_backtest()
        
        # Print results
        print("\nTrading Results:")
        print(f"Initial Capital: £{initial_capital:,.2f}")
        print(f"Final Portfolio Value: £{portfolio_values[-1]:,.2f}")
        print(f"Total Return: {((portfolio_values[-1]/initial_capital - 1) * 100):,.2f}%")
        
        # Predict next day's price
        next_price, conf_interval = trader.predict_next_price()
        print(f"\nPredicted next day price: £{next_price:,.2f}")
        print(f"Confidence Interval: £{conf_interval[0]:,.2f} to £{conf_interval[1]:,.2f}")
        print(f"Current price: £{trader.data['Close'].iloc[-1]:,.2f}")
        print(f"Predicted change: {((next_price/trader.data['Close'].iloc[-1] - 1) * 100):,.2f}%")
        
        # Plot analysis
        trader.plot_analysis(portfolio_values)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()