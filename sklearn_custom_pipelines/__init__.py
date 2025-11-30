"""
sklearn-custom-pipelines: Custom scikit-learn transformers for ML pipelines.

A collection of custom-made transformers based on Scikit-Learn pipelines
for feature engineering, encoding, and modeling.
"""

__version__ = "0.1.0"
__author__ = "Alex Kats"
__license__ = "MIT"

try:
    from sklearn_custom_pipelines.core.featurizers import (
        SimpleFeaturesTransformer,
        FeatureEliminationTransformer,
        DecorrelationTransformer,
        PairedFeaturesTransformer,
        CustomPCATransformer,
    )

    from sklearn_custom_pipelines.core.encoders import (
        WoeEncoderTransformer,
        RareCategoriesTransformer,
        BinningNumericalTransformer,
        BinningCategoriesTransformer,
        PairedBinaryFeaturesTransformer,
        CustomMappingTransformer,
    )

    from sklearn_custom_pipelines.core.models import (
        CustomLogisticRegressionClassifier,
        CustomCatBoostClassifier,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import core transformers: {e}")


__all__ = [
    # Featurizers
    "SimpleFeaturesTransformer",
    "FeatureEliminationTransformer",
    "DecorrelationTransformer",
    "PairedFeaturesTransformer",
    "PairedBinaryFeaturesTransformer",
    "CustomPCATransformer",
    # Encoders
    "WoeEncoderTransformer",
    "RareCategoriesTransformer",
    "BinningNumericalTransformer",
    "BinningCategoriesTransformer",
    "CustomMappingTransformer",
    # Models
    "CustomLogisticRegressionClassifier",
    "CustomCatBoostClassifier",
]
