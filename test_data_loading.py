#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all_data_loading():
    """Test all data loading functions to ensure they work correctly"""
    
    print("=" * 60)
    print("Testing Statistical Analysis Programs Data Loading")
    print("=" * 60)
    
    try:
        print('\n1. Testing Boston Housing Analysis...')
        from boston_housing_analysis import load_boston_data
        df = load_boston_data()
        print(f'   ✓ Boston data loaded: {len(df)} samples, {len(df.columns)} columns')
        print(f'   ✓ Target range: ${df["medv"].min():.1f}k - ${df["medv"].max():.1f}k')
        
        print('\n2. Testing Auto MPG Analysis...')
        from auto_mpg_analysis import load_auto_data
        df = load_auto_data()
        print(f'   ✓ Auto data loaded: {len(df)} samples, {len(df.columns)} columns')
        print(f'   ✓ MPG range: {df["mpg"].min():.1f} - {df["mpg"].max():.1f}')
        
        print('\n3. Testing Advertising Analysis...')
        from advertising_analysis import load_advertising_data
        df = load_advertising_data()
        print(f'   ✓ Advertising data loaded: {len(df)} samples, {len(df.columns)} columns')
        print(f'   ✓ Sales range: {df["Sales"].min():.1f} - {df["Sales"].max():.1f}')
        
        print('\n4. Testing Cross-Validation Analysis...')
        from cross_validation_analysis import load_auto_data_cv
        df = load_auto_data_cv()
        print(f'   ✓ CV data loaded: {len(df)} samples, {len(df.columns)} columns')
        
        print('\n5. Testing Classification Analysis...')
        from classification_analysis import load_stock_market_data, load_iris_data
        df1 = load_stock_market_data()
        df2 = load_iris_data()
        print(f'   ✓ Stock data loaded: {len(df1)} samples, {len(df1.columns)} columns')
        print(f'   ✓ Iris data loaded: {len(df2)} samples, {len(df2.columns)} columns')
        
        print('\n6. Testing Tree Methods Analysis...')
        from tree_methods_analysis import load_boston_data_trees, load_heart_data
        df1 = load_boston_data_trees()
        df2 = load_heart_data()
        print(f'   ✓ Boston trees data loaded: {len(df1)} samples, {len(df1.columns)} columns')
        print(f'   ✓ Heart data loaded: {len(df2)} samples, {len(df2.columns)} columns')
        
        print('\n' + "=" * 60)
        print('✅ ALL DATA LOADING TESTS PASSED!')
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f'\n❌ ERROR: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_data_loading()
    sys.exit(0 if success else 1)
