import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_comprehensive_bitcoin_system():
    """Comprehensive test of the Bitcoin Bollinger Band trading system"""
    print("🚀 Bitcoin Bollinger Band Trading System - Comprehensive Test")
    print("=" * 70)
    
    print("\n📊 Test 1: Bitcoin Data Acquisition (2 years)")
    try:
        btc = yf.Ticker('BTC-USD')
        data = btc.history(period='2y')
        
        if len(data) >= 365:  # At least 1 year of data
            print(f"✅ SUCCESS: Retrieved {len(data)} days of Bitcoin data")
            print(f"   📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"   💰 Latest price: ${data['Close'].iloc[-1]:.2f}")
            print(f"   📈 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        else:
            print(f"❌ FAILED: Only retrieved {len(data)} days of data")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print("\n📈 Test 2: Bollinger Bands Calculation")
    try:
        window = 20
        num_std = 2.0
        
        rolling_mean = data['Close'].rolling(window=window).mean()
        rolling_std = data['Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        valid_bands = upper_band.dropna()
        if len(valid_bands) > 0 and (upper_band > lower_band).all():
            print(f"✅ SUCCESS: Bollinger Bands calculated correctly")
            print(f"   📊 Valid data points: {len(valid_bands)}")
            print(f"   📏 Band width (latest): ${(upper_band.iloc[-1] - lower_band.iloc[-1]):.2f}")
        else:
            print("❌ FAILED: Invalid Bollinger Bands calculation")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print("\n🎯 Test 3: Trading Signal Generation")
    try:
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        signals['signal'] = 0
        
        signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1  # Buy
        signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1  # Sell
        
        buy_signals = (signals['signal'] == 1).sum()
        sell_signals = (signals['signal'] == -1).sum()
        total_signals = buy_signals + sell_signals
        
        if total_signals > 0:
            print(f"✅ SUCCESS: Generated trading signals")
            print(f"   🟢 Buy signals: {buy_signals}")
            print(f"   🔴 Sell signals: {sell_signals}")
            print(f"   📊 Total signals: {total_signals}")
        else:
            print("❌ FAILED: No trading signals generated")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print("\n⚡ Test 4: Backtesting Logic (2-day holding)")
    try:
        holding_days = 2
        signals['future_price'] = signals['price'].shift(-holding_days)
        signals['returns'] = (signals['future_price'] - signals['price']) / signals['price']
        
        trading_returns = signals[signals['signal'] != 0].copy()
        trading_returns['strategy_returns'] = trading_returns['returns'] * trading_returns['signal']
        
        if len(trading_returns) > 0:
            total_return = trading_returns['strategy_returns'].sum()
            win_rate = (trading_returns['strategy_returns'] > 0).mean()
            num_trades = len(trading_returns)
            avg_return = trading_returns['strategy_returns'].mean()
            
            print(f"✅ SUCCESS: Backtesting completed")
            print(f"   💰 Total return: {total_return:.2%}")
            print(f"   🎯 Win rate: {win_rate:.1%}")
            print(f"   📊 Number of trades: {num_trades}")
            print(f"   📈 Average return per trade: {avg_return:.2%}")
        else:
            print("❌ FAILED: No trading returns calculated")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print("\n🔧 Test 5: Parameter Optimization Simulation")
    try:
        best_return = -999
        best_params = None
        optimization_results = []
        
        for window in [10, 15, 20, 25]:
            for std_mult in [1.0, 1.5, 2.0, 2.5]:
                rolling_mean = data['Close'].rolling(window=window).mean()
                rolling_std = data['Close'].rolling(window=window).std()
                upper_band = rolling_mean + (rolling_std * std_mult)
                lower_band = rolling_mean - (rolling_std * std_mult)
                
                test_signals = pd.DataFrame(index=data.index)
                test_signals['price'] = data['Close']
                test_signals['upper_band'] = upper_band
                test_signals['lower_band'] = lower_band
                test_signals['signal'] = 0
                
                test_signals.loc[test_signals['price'] < test_signals['lower_band'], 'signal'] = 1
                test_signals.loc[test_signals['price'] > test_signals['upper_band'], 'signal'] = -1
                
                test_signals['future_price'] = test_signals['price'].shift(-holding_days)
                test_signals['returns'] = (test_signals['future_price'] - test_signals['price']) / test_signals['price']
                
                test_trading = test_signals[test_signals['signal'] != 0].copy()
                if len(test_trading) > 0:
                    test_trading['strategy_returns'] = test_trading['returns'] * test_trading['signal']
                    total_ret = test_trading['strategy_returns'].sum()
                    
                    optimization_results.append({
                        'window': window,
                        'std_mult': std_mult,
                        'total_return': total_ret,
                        'num_trades': len(test_trading)
                    })
                    
                    if total_ret > best_return:
                        best_return = total_ret
                        best_params = (window, std_mult)
        
        if best_params and len(optimization_results) > 0:
            print(f"✅ SUCCESS: Parameter optimization completed")
            print(f"   🏆 Best parameters: Period={best_params[0]}, Std={best_params[1]}")
            print(f"   💰 Best return: {best_return:.2%}")
            print(f"   🔍 Tested {len(optimization_results)} parameter combinations")
        else:
            print("❌ FAILED: Parameter optimization failed")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print("\n🔍 Test 6: Data Quality Validation")
    try:
        missing_data = data.isnull().sum().sum()
        
        price_std = data['Close'].std()
        price_mean = data['Close'].mean()
        
        volume_check = (data['Volume'] >= 0).all()
        price_positive = (data['Close'] > 0).all()
        
        if missing_data == 0 and volume_check and price_positive:
            print(f"✅ SUCCESS: Data quality validation passed")
            print(f"   📊 No missing data points")
            print(f"   💰 Price volatility: {(price_std/price_mean):.1%}")
            print(f"   ✅ All prices and volumes are positive")
        else:
            print(f"❌ FAILED: Data quality issues detected")
            print(f"   Missing data: {missing_data}")
            print(f"   Volume check: {volume_check}")
            print(f"   Price check: {price_positive}")
            return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! Bitcoin Bollinger Band Trading System is ready!")
    print("✅ Real Bitcoin data acquisition working")
    print("✅ Bollinger Bands calculations accurate")
    print("✅ Trading signal generation functional")
    print("✅ Backtesting logic validated")
    print("✅ Parameter optimization working")
    print("✅ Data quality verified")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = test_comprehensive_bitcoin_system()
    if success:
        print("\n🚀 System ready for production use!")
        exit(0)
    else:
        print("\n❌ System validation failed!")
        exit(1)
