"""Unit tests for sklearn-custom-pipelines package."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def synthetic_dataset():
    """
    Pytest fixture that creates a synthetic dataset for testing transformers.
    
    Returns
    -------
    tuple
        (X, y) - Features dataframe and target series
    """
    n_samples = 100
    random_state = 42
    np.random.seed(random_state)
    
    data = {
        # Numerical features
        'battery_level': np.random.uniform(0, 100, n_samples),
        'gps_location_lat': np.random.uniform(-90, 90, n_samples),
        'gps_location_lon': np.random.uniform(-180, 180, n_samples),
        
        # Categorical features
        'is_charging': np.random.choice(['Yes', 'No', None], n_samples, p=[0.4, 0.4, 0.2]),
        'device_model': np.random.choice(
            ['iPhone', 'Samsung', 'Xiaomi', 'OnePlus', None],
            n_samples,
            p=[0.3, 0.3, 0.2, 0.1, 0.1]
        ),
        'device_type': np.random.choice(
            ['Phone', 'Tablet', None],
            n_samples,
            p=[0.7, 0.2, 0.1]
        ),
        'is_vpn_connected': np.random.choice(['Yes', 'No', None], n_samples, p=[0.2, 0.7, 0.1]),
        'os_version': np.random.choice(
            ['Android 10', 'Android 11', 'iOS 14', 'iOS 15', None],
            n_samples,
            p=[0.2, 0.3, 0.25, 0.2, 0.05]
        ),
        'telco_carrier': np.random.choice(
            ['Carrier1', 'Carrier2', 'Carrier3', None],
            n_samples,
            p=[0.4, 0.35, 0.2, 0.05]
        ),
        'network_type': np.random.choice(
            ['WiFi', '4G', '5G', None],
            n_samples,
            p=[0.4, 0.4, 0.15, 0.05]
        ),
    }
    
    X = pd.DataFrame(data)
    
    # Create a somewhat correlated target
    y = pd.Series(
        (X['battery_level'] > 50).astype(int) & (np.random.rand(n_samples) > 0.3),
        name='y'
    )
    
    return X, y


def test_synthetic_data_creation(synthetic_dataset):
    """Test that synthetic data can be created successfully."""
    X, y = synthetic_dataset
    
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert y.isin([0, 1]).all()
    assert list(y.name) == list('y')


class TestSimpleFeaturesTransformer:
    """Test SimpleFeaturesTransformer class."""
    
    def test_transformer_initialization(self):
        """Test that transformer can be initialized."""
        from sklearn_custom_pipelines import SimpleFeaturesTransformer
        
        num_features = {'battery_level': 50.0}
        cat_features = {'device_model': '__MISSING__'}
        
        transformer = SimpleFeaturesTransformer(num_features, cat_features)
        
        assert transformer.num_features_missings_dct == num_features
        assert transformer.cat_features_missings_dct == cat_features
    
    def test_transformer_fit(self, synthetic_dataset):
        """Test that transformer can be fitted."""
        from sklearn_custom_pipelines import SimpleFeaturesTransformer
        
        X, y = synthetic_dataset
        
        num_features = {'battery_level': 50.0, 'gps_location_lat': 0.0}
        cat_features = {'device_model': '__MISSING__'}
        
        transformer = SimpleFeaturesTransformer(num_features, cat_features)
        transformer.fit(X, y)
        
        assert transformer is not None
    
    def test_transformer_transform(self, synthetic_dataset):
        """Test that transformer can transform data."""
        from sklearn_custom_pipelines import SimpleFeaturesTransformer
        
        X, y = synthetic_dataset
        
        num_features = {'battery_level': 50.0}
        cat_features = {'device_model': '__MISSING__'}
        
        transformer = SimpleFeaturesTransformer(num_features, cat_features)
        transformer.fit(X, y)
        
        X_transformed = transformer.transform(X, y)
        
        assert 'num__battery_level' in X_transformed.columns
        assert 'cat__device_model' in X_transformed.columns
        assert X_transformed.shape[0] == X.shape[0]
    
    def test_transformer_pipeline_compatibility(self, synthetic_dataset):
        """Test that transformer works with sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn_custom_pipelines import SimpleFeaturesTransformer
        
        X, y = synthetic_dataset
        
        num_features = {'battery_level': 50.0}
        cat_features = {'device_model': '__MISSING__'}
        
        pipeline = Pipeline([
            ('simple_features', SimpleFeaturesTransformer(num_features, cat_features))
        ])
        
        X_transformed = pipeline.fit_transform(X, y)
        
        assert 'num__battery_level' in X_transformed.columns
        assert 'cat__device_model' in X_transformed.columns


class TestRareCategoriesTransformer:
    """Test RareCategoriesTransformer class."""
    
    def test_transformer_initialization(self):
        """Test that transformer can be initialized."""
        from sklearn_custom_pipelines import RareCategoriesTransformer
        
        transformer = RareCategoriesTransformer(tol=0.01, n_categories=5)
        
        assert transformer.tol == 0.01
        assert transformer.n_categories == 5
    
    def test_transformer_fit(self, synthetic_dataset):
        """Test that transformer can be fitted."""
        from sklearn_custom_pipelines import RareCategoriesTransformer, SimpleFeaturesTransformer
        
        X, y = synthetic_dataset
        
        # First add categorical prefixes
        num_features = {}
        cat_features = {'device_model': '__MISSING__', 'device_type': '__MISSING__'}
        
        simple_tf = SimpleFeaturesTransformer(num_features, cat_features)
        X_simple = simple_tf.fit_transform(X, y)
        
        transformer = RareCategoriesTransformer(tol=0.01)
        transformer.fit(X_simple, y)
        
        assert transformer.cat_features_lst is not None
        assert len(transformer.encoder_dict_) > 0
    
    def test_transformer_transform(self, synthetic_dataset):
        """Test that transformer can transform data."""
        from sklearn_custom_pipelines import RareCategoriesTransformer, SimpleFeaturesTransformer
        
        X, y = synthetic_dataset
        
        num_features = {}
        cat_features = {'device_model': '__MISSING__', 'device_type': '__MISSING__'}
        
        simple_tf = SimpleFeaturesTransformer(num_features, cat_features)
        X_simple = simple_tf.fit_transform(X, y)
        
        transformer = RareCategoriesTransformer(tol=0.05)
        transformer.fit(X_simple, y)
        
        X_transformed = transformer.transform(X_simple, y)
        
        assert X_transformed.shape[0] == X_simple.shape[0]


class TestWoeEncoderTransformer:
    """Test WoeEncoderTransformer class."""
    
    def test_transformer_initialization(self):
        """Test that transformer can be initialized."""
        from sklearn_custom_pipelines import WoeEncoderTransformer
        
        transformer = WoeEncoderTransformer(zero_filler=0.001)
        
        assert transformer.zero_filler == 0.001
    
    def test_woe_encoder_workflow(self, synthetic_dataset):
        """Test complete WOE encoding workflow."""
        from sklearn_custom_pipelines import (
            SimpleFeaturesTransformer,
            RareCategoriesTransformer,
            BinningCategoriesTransformer,
            WoeEncoderTransformer
        )
        
        X, y = synthetic_dataset
        
        # For this test, we'll use the fixture data (100 samples is sufficient)
        # Create pipeline
        num_features = {}
        cat_features = {'device_model': '__MISSING__', 'device_type': '__MISSING__'}
        
        simple_tf = SimpleFeaturesTransformer(num_features, cat_features)
        X_simple = simple_tf.fit_transform(X, y)
        
        rare_tf = RareCategoriesTransformer(tol=0.05)
        X_rare = rare_tf.fit_transform(X_simple, y)
        
        binning_tf = BinningCategoriesTransformer(max_n_bins=3)
        X_binned = binning_tf.fit_transform(X_rare, y)
        
        woe_tf = WoeEncoderTransformer()
        woe_tf.fit(X_binned, y)
        
        X_woe = woe_tf.transform(X_binned, y)
        
        # Check that WOE columns were created
        woe_cols = [col for col in X_woe.columns if '__woe' in col]
        assert len(woe_cols) > 0


class TestFeatureEliminationTransformer:
    """Test FeatureEliminationTransformer class."""
    
    def test_transformer_initialization(self):
        """Test that transformer can be initialized."""
        from sklearn_custom_pipelines import FeatureEliminationTransformer
        
        transformer = FeatureEliminationTransformer(correlation_thr=0.9)
        
        assert transformer.correlation_thr == 0.9
    
    def test_transformer_fit(self, synthetic_dataset):
        """Test that transformer can be fitted."""
        from sklearn_custom_pipelines import (
            SimpleFeaturesTransformer,
            FeatureEliminationTransformer
        )
        
        X, y = synthetic_dataset
        
        num_features = {
            'battery_level': 50.0,
            'gps_location_lat': 0.0,
            'gps_location_lon': 0.0
        }
        cat_features = {'device_model': '__MISSING__'}
        
        simple_tf = SimpleFeaturesTransformer(num_features, cat_features)
        X_simple = simple_tf.fit_transform(X, y)
        
        elim_tf = FeatureEliminationTransformer()
        elim_tf.fit(X_simple, y)
        
        assert len(elim_tf.features_to_drop) >= 0


class TestDecorrelationTransformer:
    """Test DecorrelationTransformer class."""
    
    def test_transformer_initialization(self):
        """Test that transformer can be initialized."""
        from sklearn_custom_pipelines import DecorrelationTransformer
        
        transformer = DecorrelationTransformer(correlation_thr=0.85)
        
        assert transformer.correlation_thr == 0.85


class TestCalculateIV:
    """Test calculate_iv helper function."""
    
    def test_calculate_iv_basic(self):
        """Test basic IV calculation with simple data."""
        from sklearn_custom_pipelines.utils.helpers import calculate_iv
        
        # Create simple test data
        X = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'A', 'B']
        })
        y = pd.Series([1, 1, 0, 0, 1, 0])
        
        iv = calculate_iv(X, y, 'feature')
        
        # IV should be a float value
        assert isinstance(iv, (float, np.floating))
        # IV should be non-negative
        assert iv >= 0
    
    def test_calculate_iv_perfect_separation(self):
        """Test IV with perfectly separable feature."""
        from sklearn_custom_pipelines.utils.helpers import calculate_iv
        
        # Perfect separation: A always has y=1, B always has y=0
        X = pd.DataFrame({
            'feature': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        y = pd.Series([1, 1, 1, 0, 0, 0])
        
        iv = calculate_iv(X, y, 'feature')
        
        # IV should be high for perfect separation
        assert iv > 0.5
    
    def test_calculate_iv_no_separation(self):
        """Test IV with no predictive power."""
        from sklearn_custom_pipelines.utils.helpers import calculate_iv
        
        # No separation: both categories have same distribution
        X = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B']
        })
        y = pd.Series([1, 0, 1, 0])
        
        iv = calculate_iv(X, y, 'feature')
        
        # IV should be low or zero for no separation
        assert iv <= 0.1


class TestCalculateWOE:
    """Test calculate_woe helper function."""
    
    def test_calculate_woe_basic(self):
        """Test basic WOE calculation."""
        from sklearn_custom_pipelines.utils.helpers import calculate_woe
        
        # Create simple test data
        X = pd.DataFrame({
            'feature': ['A', 'A', 'B', 'B', 'A', 'B']
        })
        y = pd.Series([1, 1, 0, 0, 1, 0])
        
        woe_dict = calculate_woe(X, y, 'feature')
        
        # Should return a dictionary
        assert isinstance(woe_dict, dict)
        # Should have entries for each category
        assert len(woe_dict) == 2
        # Should have keys for the categories
        assert 'A' in woe_dict
        assert 'B' in woe_dict
    
    def test_calculate_woe_values(self):
        """Test WOE values are numeric and have correct properties."""
        from sklearn_custom_pipelines.utils.helpers import calculate_woe
        
        X = pd.DataFrame({
            'feature': ['A', 'A', 'A', 'B', 'B', 'B']
        })
        y = pd.Series([1, 1, 1, 0, 0, 0])
        
        woe_dict = calculate_woe(X, y, 'feature')
        
        # All WOE values should be numeric
        for key, value in woe_dict.items():
            assert isinstance(value, (float, np.floating))
        
        # WOE should be positive for categories with higher event rate
        # A has 100% events, B has 0% events
        assert woe_dict['A'] > 0
        assert woe_dict['B'] < 0
    
    def test_calculate_woe_zero_filler(self):
        """Test zero_filler parameter prevents log(0) errors."""
        from sklearn_custom_pipelines.utils.helpers import calculate_woe
        
        # Case where one category has no events
        X = pd.DataFrame({
            'feature': ['A', 'A', 'A', 'B']
        })
        y = pd.Series([1, 1, 1, 0])
        
        # Should not raise an error
        woe_dict = calculate_woe(X, y, 'feature', zero_filler=0.01)
        
        # Should return valid dictionary
        assert isinstance(woe_dict, dict)
        assert len(woe_dict) == 2
        
        # All values should be finite
        for value in woe_dict.values():
            assert np.isfinite(value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

