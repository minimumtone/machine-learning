import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_nikkei_data_acquisition():
    """Test Nikkei 225 data acquisition"""
    print("Testing Nikkei 225 data acquisition...")
    try:
        nikkei = yf.Ticker('^N225')
        data = nikkei.history(period='5d')
        
        if len(data) > 0:
            print(f"✓ Successfully retrieved {len(data)} days of Nikkei 225 data")
            print(f"✓ Latest price: ¥{data['Close'].iloc[-1]:.2f}")
            print(f"✓ Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            return True
        else:
            print("✗ No data retrieved")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_nikkei_2year_data():
    """Test 2-year Nikkei 225 data acquisition"""
    print("\nTesting 2-year Nikkei 225 data acquisition...")
    try:
        nikkei = yf.Ticker('^N225')
        data = nikkei.history(period='2y')
        
        if len(data) >= 365:
            print(f"✓ Successfully retrieved {len(data)} days of Nikkei 225 data")
            print(f"✓ Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"✓ Latest price: ¥{data['Close'].iloc[-1]:.2f}")
            print(f"✓ Price range: ¥{data['Close'].min():.2f} - ¥{data['Close'].max():.2f}")
            return True
        else:
            print(f"✗ Only retrieved {len(data)} days of data")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_bollinger_bands_nikkei():
    """Test Bollinger Bands calculation with Nikkei data"""
    print("\nTesting Bollinger Bands calculation with Nikkei data...")
    try:
        nikkei = yf.Ticker('^N225')
        data = nikkei.history(period='3mo')
        
        window = 20
        num_std = 2.0
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        valid_bands = upper_band.dropna()
        if len(valid_bands) > 0 and (upper_band > lower_band).dropna().all():
            print("✓ Bollinger Bands calculated successfully")
            print(f"✓ Valid data points: {len(valid_bands)}")
            print(f"✓ Band width (latest): ¥{(upper_band.iloc[-1] - lower_band.iloc[-1]):.2f}")
            return True
        else:
            print("✗ Bollinger Bands calculation failed")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_trading_signals_nikkei():
    """Test trading signal generation with Nikkei data"""
    print("\nTesting trading signal generation with Nikkei data...")
    try:
        nikkei = yf.Ticker('^N225')
        data = nikkei.history(period='6mo')
        
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
        
        signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1
        signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1
        
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        total_signals = buy_signals + sell_signals
        
        if total_signals > 0:
            print(f"✓ Generated trading signals")
            print(f"✓ Buy signals: {buy_signals}")
            print(f"✓ Sell signals: {sell_signals}")
            print(f"✓ Total signals: {total_signals}")
            return True
        else:
            print("✗ No trading signals generated")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("Nikkei 225 Bollinger Band Trading System - Component Tests")
    print("=" * 60)
    
    tests = [
        test_nikkei_data_acquisition,
        test_nikkei_2year_data,
        test_bollinger_bands_nikkei,
        test_trading_signals_nikkei
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed! Nikkei 225 system is ready for full testing.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    main()
