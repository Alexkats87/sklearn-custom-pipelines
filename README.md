# sklearn-custom-pipelines

A comprehensive collection of custom scikit-learn transformers for building end-to-end machine learning pipelines. Includes feature engineering, encoding, and modeling transformers organized in a properly structured Python package.

## Features

This package provides production-ready transformers for:

### 1. Feature Engineering

- **SimpleFeaturesTransformer** - Add new columns with prefixes `cat__` and `num__` for specified columns, that will be treated as initial features on the next steps of pipeline

- **FeatureEliminationTransformer** - Drops features that are constant or quasi-constant overall dataset, detects and drops duplicated features, filters by information value (IV). Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib

- **DecorrelationTransformer** - Detects and removes correlated features. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib
  - Detects groups of correlated features based on correlation threshold
  - From every group only one feature is selected based on mean target performance

- **PairedFeaturesTransformer** - Creates interaction features from high-cardinality categorical features

- **CustomPCATransformer** - Applies PCA transformation to numerical features with automatic scaling

### 2. Feature Encoding

- **WoeEncoderTransformer** - Transforms categorical features into **weight of evidence (WOE)** values (see detailed explanation [here](https://www.analyticsvidhya.com/blog/2021/06/understand-weight-of-evidence-and-information-value/)). Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib

- **RareCategoriesTransformer** - Encodes rare values of categorical features into one value `Others` to reduce features cardinality. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib

- **BinningCategoriesTransformer** - Applies categories grouping for categorical features into bigger groups with similar WOE values to reduce cardinality. Based on [optbinning](https://github.com/guillermo-navas-palencia/optbinning) lib.

- **BinningNumericalTransformer** - Applies discretisation for numerical features. Based on [optbinning](https://github.com/guillermo-navas-palencia/optbinning) lib.

- **CustomMappingTransformer** - Applies custom mappings to specified features

### 3. Modeling

- **CustomLogisticRegressionClassifier** - Fits Logistic Regression model (from [statsmodels](https://www.statsmodels.org/stable/index.html)), performs iterative feature selection excluding features with high p-values after every model fit

- **CustomCatBoostClassifier** - Fits CatBoostClassifier model, performs iterative feature selection excluding features with low feature importance after every model fit. Uses evaluation set for early stopping and includes probability calibration.

## Installation

### From PyPI (Coming Soon)

```bash
pip install sklearn-custom-pipelines
```

### From GitHub with UV (Recommended - Fast & Reproducible)

```bash
git clone https://github.com/Alexkats87/sklearn-custom-pipelines.git
cd sklearn-custom-pipelines
uv venv sk-custom
source sk-custom/bin/activate
uv pip install -r requirements.lock
```

See [UV_SETUP.md](UV_SETUP.md) for detailed UV setup instructions.

### From GitHub with pip

```bash
git clone https://github.com/Alexkats87/sklearn-custom-pipelines.git
cd sklearn-custom-pipelines
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage with CatBoost

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn_custom_pipelines import (
    SimpleFeaturesTransformer,
    FeatureEliminationTransformer,
    DecorrelationTransformer,
    RareCategoriesTransformer,
    CustomCatBoostClassifier,
)
from sklearn_custom_pipelines.utils.const import NUM, MISSING

# Load your data
X = pd.read_csv('features.csv')
y = pd.read_csv('target.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define feature handling
num_features_missings = {
    'age': 50.0,
    'income': X_train['income'].mean(),
}

cat_features_missings = {
    'city': MISSING,
    'country': MISSING,
}

# Create pipeline
pipeline = Pipeline([
    ('simple_features', SimpleFeaturesTransformer(
        num_features_missings, cat_features_missings
    )),
    ('feature_elimination', FeatureEliminationTransformer()),
    ('decorrelation', DecorrelationTransformer(features_pattern=fr"^{NUM}.*")),
    ('rare_categories', RareCategoriesTransformer()),
    ('model', CustomCatBoostClassifier(verbose=False)),
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Example with Logistic Regression and WOE Encoding

```python
from sklearn_custom_pipelines import (
    SimpleFeaturesTransformer,
    RareCategoriesTransformer,
    BinningCategoriesTransformer,
    BinningNumericalTransformer,
    WoeEncoderTransformer,
    CustomLogisticRegressionClassifier,
)
from sklearn_custom_pipelines.utils.const import BIN, WOE, MISSING

# Create pipeline with WOE encoding
pipeline = Pipeline([
    ('simple_features', SimpleFeaturesTransformer(
        num_features_missings, cat_features_missings
    )),
    ('rare_categories', RareCategoriesTransformer()),
    ('binning_num', BinningNumericalTransformer()),
    ('binning_cat', BinningCategoriesTransformer()),
    ('woe_encoding', WoeEncoderTransformer()),
    ('model', CustomLogisticRegressionClassifier()),
])

# Fit and predict
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Running Examples

Example scripts are provided that can be run directly without installation:

```bash
python examples/example_catboost.py
python examples/example_logreg.py
```

## Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=sklearn_custom_pipelines --cov-report=html
```

## Project Structure

```
sklearn-custom-pipelines/
├── sklearn_custom_pipelines/       # Main package
│   ├── core/                       # Core transformers
│   │   ├── featurizers.py
│   │   ├── encoders.py
│   │   └── models.py
│   └── utils/                      # Utilities
│       ├── const.py
│       ├── helpers.py
│       └── custom_mappings.py
├── tests/                          # Unit tests
├── examples/                       # Example scripts
├── setup.py                        # Setup configuration
├── requirements.txt                # Dependencies
├── train_catboost.py              # Training script
├── train_logreg.py                # Training script
└── DEVELOPMENT.md                 # Development guide
```

## Documentation

For detailed information on installation, development, and publishing, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Key Concepts

### Feature Prefixes

The package uses consistent prefixes for features at different stages:

- `cat__` - Categorical features
- `num__` - Numerical features
- `__bin` - Binned features
- `__woe` - WOE-encoded features

### Constants

Key constants are defined in `sklearn_custom_pipelines.utils.const`:

```python
from sklearn_custom_pipelines.utils.const import (
    MISSING,      # Placeholder for missing values
    OTHER,        # Placeholder for rare categories
    CAT, NUM,     # Feature prefixes
    BIN, WOE,     # Feature suffixes
    TARGET,       # Default target column name
)
```

## Requirements

- Python >= 3.7
- scikit-learn >= 0.23.2
- pandas >= 1.1.3
- numpy >= 1.19.2
- feature-engine >= 1.2.0
- statsmodels >= 0.13.5
- optbinning >= 0.11.0
- catboost >= 1.2.5

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Alex Kats ([@Alexkats87](https://github.com/Alexkats87))

## References

- [Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Feature-engine Documentation](https://feature-engine.trainindata.com/)
- [OptBinning Documentation](https://gnp.github.io/optbinning/)
- [CatBoost Documentation](https://catboost.ai/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [Weight of Evidence and Information Value](https://www.analyticsvidhya.com/blog/2021/06/understand-weight-of-evidence-and-information-value/)

