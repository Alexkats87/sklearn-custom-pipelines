"""Constants used throughout sklearn-custom-pipelines."""

import string
import numpy as np

# Missing and other value placeholders
MISSING = "__MISSING__"
OTHER = "__OTHER__"

# Feature prefixes for categorical and numerical features
CAT = "cat__"
NUM = "num__"

# Feature suffixes for binned and WOE-encoded features
BIN = "__bin"
WOE = "__woe"

# Separator for paired features
SEP = "____"

# Numpy NaN representation
NAN = np.nan

# Groups for feature names (A, B, C, ...)
GROUPS = string.ascii_uppercase

# Default target column name
TARGET = "y"
