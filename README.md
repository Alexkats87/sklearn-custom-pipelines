# sklearn-custom-pipelines

My collection of custom-made transformers based on [Scikit-Learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

**1. BASIC FEATURES MANIPULATIONS**

- **SimpleFeaturesTransformer** - add new columns with prefixes `cat__` and `num__` for specified columns, that will be treated as initial features on the next steps of pipeline
  
- **FeatureEliminationTransformer** - drops features that are constant or quasi-constant overall dataset, and so are useless. Detects and drops duplicated features. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib
  
- **DecorrelationTransformer** - detects and removes correlated features. Based on [Feature-engine](https://feature-engine.trainindata.com/en/latest/) lib
  - Detects groups of correlated features based on correlation threshold set up
  - From every group only one feature is selected based on mean target performance. Other features within this group will be dropped






**2. ENCODERS**
