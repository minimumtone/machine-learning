import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestBostonHousingAnalysis:
    def test_data_loading(self):
        from boston_housing_analysis import load_boston_data
        df = load_boston_data()
        
        assert len(df) == 506
        assert 'medv' in df.columns
        assert len(df.columns) == 14
        assert df['medv'].min() >= 5
        assert df['medv'].max() <= 50
        assert not df.isnull().any().any()
    
    def test_data_types(self):
        from boston_housing_analysis import load_boston_data
        df = load_boston_data()
        
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col])

class TestAutoMPGAnalysis:
    def test_data_loading(self):
        from auto_mpg_analysis import load_auto_data
        df = load_auto_data()
        
        assert len(df) == 392
        assert 'mpg' in df.columns
        assert 'horsepower' in df.columns
        assert df['mpg'].min() >= 9
        assert df['mpg'].max() <= 47
        assert not df.isnull().any().any()
    
    def test_data_relationships(self):
        from auto_mpg_analysis import load_auto_data
        df = load_auto_data()
        
        correlation = df['mpg'].corr(df['horsepower'])
        assert correlation < 0

class TestAdvertisingAnalysis:
    def test_data_loading(self):
        from advertising_analysis import load_advertising_data
        df = load_advertising_data()
        
        assert len(df) == 200
        assert 'Sales' in df.columns
        assert 'TV' in df.columns
        assert 'Radio' in df.columns
        assert 'Newspaper' in df.columns
        assert not df.isnull().any().any()
    
    def test_sales_range(self):
        from advertising_analysis import load_advertising_data
        df = load_advertising_data()
        
        assert df['Sales'].min() >= 1
        assert df['Sales'].max() <= 30

class TestCrossValidationAnalysis:
    def test_data_loading(self):
        from cross_validation_analysis import load_auto_data_cv
        df = load_auto_data_cv()
        
        assert len(df) == 392
        assert 'mpg' in df.columns
        assert 'horsepower' in df.columns
        assert not df.isnull().any().any()

class TestClassificationAnalysis:
    def test_stock_data_loading(self):
        from classification_analysis import load_stock_market_data
        df = load_stock_market_data()
        
        assert len(df) == 1250
        assert 'Direction' in df.columns
        assert set(df['Direction'].unique()) == {'Up', 'Down'}
        assert not df.isnull().any().any()
    
    def test_iris_data_loading(self):
        from classification_analysis import load_iris_data
        df = load_iris_data()
        
        assert len(df) == 150
        assert 'species' in df.columns
        assert len(df['species'].unique()) == 3
        assert not df.isnull().any().any()

class TestTreeMethodsAnalysis:
    def test_boston_data_loading(self):
        from tree_methods_analysis import load_boston_data_trees
        df = load_boston_data_trees()
        
        assert len(df) == 506
        assert 'medv' in df.columns
        assert len(df.columns) == 14
        assert not df.isnull().any().any()
    
    def test_heart_data_loading(self):
        from tree_methods_analysis import load_heart_data
        df = load_heart_data()
        
        assert len(df) == 303
        assert 'target' in df.columns
        assert set(df['target'].unique()) == {0, 1}
        assert not df.isnull().any().any()

def run_all_tests():
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_all_tests()
