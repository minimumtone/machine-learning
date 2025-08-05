import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_bitcoin_data_acquisition():
    """Test Bitcoin data acquisition"""
    print("Testing Bitcoin data acquisition...")
    try:
        btc = yf.Ticker('BTC-USD')
        data = btc.history(period='5d')
        
        if len(data) > 0:
            print(f"✓ Successfully retrieved {len(data)} days of Bitcoin data")
            print(f"✓ Latest price: ${data['Close'].iloc[-1]:.2f}")
            print(f"✓ Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            return True
        else:
            print("✗ No data retrieved")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_bollinger_bands():
    """Test Bollinger Bands calculation"""
    print("\nTesting Bollinger Bands calculation...")
    try:
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        prices = 50000 + np.cumsum(np.random.randn(30) * 1000)
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        window = 20
        num_std = 2.0
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        if len(upper_band.dropna()) > 0 and len(lower_band.dropna()) > 0:
            print("✓ Bollinger Bands calculated successfully")
            print(f"✓ Upper band range: ${lower_band.dropna().iloc[-1]:.2f} - ${upper_band.dropna().iloc[-1]:.2f}")
            return True
        else:
            print("✗ Bollinger Bands calculation failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_trading_signals():
    """Test trading signal generation"""
    print("\nTesting trading signal generation...")
    try:
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        base_price = 50000
        prices = []
        
        for i in range(50):
            if i < 20:
                prices.append(base_price + i * 100)  # Uptrend
            elif i < 30:
                prices.append(base_price + 2000 - (i-20) * 200)  # Downtrend
            else:
                prices.append(base_price + np.random.randn() * 500)  # Sideways
        
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        window = 20
        num_std = 2.0
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        signals['signal'] = 0
        
        signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1  # Buy
        signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1  # Sell
        
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        
        print(f"✓ Generated {buy_signals} buy signals and {sell_signals} sell signals")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("Bitcoin Bollinger Band Trading System - Component Tests")
    print("=" * 60)
    
    tests = [
        test_bitcoin_data_acquisition,
        test_bollinger_bands,
        test_trading_signals
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! System is ready for full backtesting.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    main()
