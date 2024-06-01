# sklearn-custom-pipelines

My collection of custom-made transformers based on [Scikit-Learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

**1. BASIC FEATURES MANIPULATIONS**

- **SimpleFeaturesTransformer** - add new columns with prefixes `cat__` and `num__` for specified columns, that will be treated as initial features on the next steps of pipeline
  
- **FeatureEliminationTransformer** - drops features that are constant or quasi-constant overall dataset, and so are useless. Detects and drops duplicated features. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib
  
- **DecorrelationTransformer** - detects and removes correlated features. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib
  - Detects groups of correlated features based on correlation threshold set up
  - From every group only one feature is selected based on mean target performance. Other features within this group will be dropped




**2. ENCODERS**

- **WoeEncoderTransformer** - transforms categirical features into ***weight of evidence (WOE)***  values (see detailed explonation [here](https://www.analyticsvidhya.com/blog/2021/06/understand-weight-of-evidence-and-information-value/)). Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib

- **RareCategoriesTransformer** - encodes rare values of categorical features into one value `Others` to reduce features cardinality. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib

- **BinningCategoriesTransformer** - applies categories grouping for categorical features into bigger groups with similar WOE values to reduce cardinality. Based on [optbinning](https://github.com/guillermo-navas-palencia/optbinning) lib.

- **BinningNumericalTransformer** - applies discretisation for numerical features. Based on [optbinning](https://github.com/guillermo-navas-palencia/optbinning) lib.


**3. MODELING**

- **CustomLogisticRegressionClassifier** - fits Logistic Regression model (from [statsmodels](https://www.statsmodels.org/stable/index.html)), performs iterative feature selection excluding features with high p-values after every model fit

- **CustomCatBoostClassifier** - fits CatBoostClassifier model, performs iterative feature selection excluding features with low feature importance after every model fit. Uses evaluation set for early stopping.
