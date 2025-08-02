import pandas as pd
import numpy as np
from datetime import datetime

def create_enhanced_stock_data(input_file, output_file):
    """
    Creates an enhanced CSV file with engineered features for stock prediction
    """
    # Load the original data
    df = pd.read_csv(input_file)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    
    # Sort by date to ensure chronological order
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Basic price features
    df['price_range'] = df['high'] - df['low']
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
    
    # Price ratios
    df['close_to_open_ratio'] = df['close'] / df['open']
    df['high_to_low_ratio'] = df['high'] / df['low']
    df['close_to_high_ratio'] = df['close'] / df['high']
    df['close_to_low_ratio'] = df['close'] / df['low']
    
    # Moving averages
    windows = [5, 10, 20, 50]
    for window in windows:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Price relative to moving average
        df[f'close_vs_sma_{window}'] = df['close'] / df[f'sma_{window}']
        df[f'volume_vs_sma_{window}'] = df['volume'] / df[f'volume_sma_{window}']
    
    # Exponential moving averages
    ema_windows = [12, 26]
    for window in ema_windows:
        df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI (Relative Strength Index)
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi'] = calculate_rsi(df['close'])
    
    # Bollinger Bands
    bb_window = 20
    df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
    bb_std = df['close'].rolling(window=bb_window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volatility measures
    df['volatility_5'] = df['close'].rolling(window=5).std()
    df['volatility_20'] = df['close'].rolling(window=20).std()
    df['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_price_trend'] = df['volume'] * df['price_change_pct']
    
    # Lag features (previous day values)
    lag_periods = [1, 2, 3, 5]
    for lag in lag_periods:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'price_change_lag_{lag}'] = df['price_change_pct'].shift(lag)
    
    # Rate of change
    roc_periods = [5, 10, 20]
    for period in roc_periods:
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    
    # Support/Resistance levels (simplified)
    df['recent_high'] = df['high'].rolling(window=20).max()
    df['recent_low'] = df['low'].rolling(window=20).min()
    df['distance_to_high'] = (df['recent_high'] - df['close']) / df['close'] * 100
    df['distance_to_low'] = (df['close'] - df['recent_low']) / df['close'] * 100
    
    # Time-based features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    
    # Market trend features
    df['trend_5'] = np.where(df['close'] > df['sma_5'], 1, 0)
    df['trend_20'] = np.where(df['close'] > df['sma_20'], 1, 0)
    df['golden_cross'] = np.where(df['sma_5'] > df['sma_20'], 1, 0)
    
    # Target variable engineering (if needed)
    df['next_day_return'] = df['close'].shift(-1) / df['close'] - 1
    df['buy_signal_strength'] = df['BuySignal'] * df['price_change_pct']
    
    # Remove rows with NaN values (due to rolling calculations)
    df = df.dropna()
    
    df.to_csv(output_file, index=False)
    
    print(f"Enhanced dataset created with {len(df)} rows and {len(df.columns)} columns")
    print(f"File saved as: {output_file}")
    
    # Display feature summary
    print("\nNew Features Created:")
    feature_categories = {
        'Price Features': ['price_range', 'price_change', 'price_change_pct'],
        'Ratios': ['close_to_open_ratio', 'high_to_low_ratio', 'close_to_high_ratio'],
        'Moving Averages': [col for col in df.columns if 'sma_' in col or 'ema_' in col],
        'Technical Indicators': ['macd', 'macd_signal', 'rsi', 'bb_position'],
        'Volatility': ['volatility_5', 'volatility_20', 'price_range_pct'],
        'Lag Features': [col for col in df.columns if 'lag_' in col],
        'Trend Features': ['trend_5', 'trend_20', 'golden_cross']
    }
    
    for category, features in feature_categories.items():
        print(f"\n{category}: {len(features)} features")
        for feature in features[:3]:  # Show first 3 features
            if feature in df.columns:
                print(f"  - {feature}")
        if len(features) > 3:
            print(f"  ... and {len(features)-3} more")
    
    return df

if __name__ == "__main__":
    input_file = "15 yr w BUYsig.csv"
    output_file = "enhanced_stock_data.csv"
    
    enhanced_df = create_enhanced_stock_data(input_file, output_file)
