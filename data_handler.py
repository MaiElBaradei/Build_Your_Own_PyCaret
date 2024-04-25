from pycaret.datasets import get_data
from pycaret.classification import *
from pycaret.regression import *
import pandas as pd
from typing import Union

def dataSetup(
    data: str,
    target: str,
    problem_type: {"Classification", "Regression"} = "Regression",
    numeric_imputation: {int, float, "drop", "mean", "median", "mode", "knn"} = "mean",
    categorical_imputation: {"drop", "mode", str} = "mode",
    numeric_features: list = None,
    categorical_features: list = None,
    date_features: list = None,
    ignore_features: list = None,
    keep_features: list = None,
    ordinal_features: dict = None,
    max_encoding_ohe: int = 25,
    remove_outliers: bool = False,
    outliers_method: {"iforest","ee","lof"}= "iforest",
    outliers_threshold: float = 0.05,
) -> Union[ClassificationExperiment, RegressionExperiment]:
    """
    Setup function for PyCaret's Classification or Regression Experiment.

    Parameters:
    - data (str): The name of the dataset or path to the CSV file.
    - target (str): The name of the target variable in the dataset.
    - problem_type ({"Classification", "Regression"}, optional): Type of problem to solve, defaults to "Regression".
    - numeric_imputation ({int, float, "drop", "mean", "median", "mode", "knn"}, optional):
        Method for numeric feature imputation, defaults to "mean".
    - categorical_imputation ({"drop", "mode", str}, optional):
        Method for categorical feature imputation, defaults to "mode".
    - numeric_features (list, optional): List of numeric feature names, defaults to None (automatically inferred).
    - categorical_features (list, optional): List of categorical feature names, defaults to None (automatically inferred).
    - date_features (list, optional): List of date feature names, defaults to None (automatically inferred).
    - ignore_features (list, optional): List of feature names to ignore, defaults to None.
    - keep_features (list, optional): List of feature names to keep, defaults to None.
    - ordinal_features (dict, optional): Dictionary mapping ordinal features to their categories, defaults to None.
    - max_encoding_ohe (int, optional): Maximum categories for one-hot encoding, defaults to 25.
    - remove_outliers (bool, optional): Whether to remove outliers, defaults to False.
    - outliers_method ({"iforest","ee","lof"}, optional): Method for outlier detection, defaults to "iforest".
    - outliers_threshold (float, optional): Threshold for outlier detection, defaults to 0.05.

    Returns:
    - ClassificationExperiment or RegressionExperiment: The PyCaret experiment object.

    Raises:
    - ValueError: If an invalid value is provided for any parameter.

    Notes:
    - This function sets up the data for a Classification or Regression experiment using PyCaret.
    - The `data` parameter can be the name of a dataset available in PyCaret's datasets or a path to a CSV, Excel, or JSON file or a Pandas dataframe.
    - Numeric and categorical feature imputation methods can be specified, along with feature selection and outlier removal options.
    """
    
    if problem_type == "Classification":
        s = ClassificationExperiment()
    elif problem_type == "Regression":
        s = RegressionExperiment()
    s.setup(
        data,
        target=target,
        numeric_imputation=numeric_imputation,
        categorical_imputation=categorical_imputation,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
        ignore_features=ignore_features,
        keep_features=keep_features,
        ordinal_features=ordinal_features,
        max_encoding_ohe=max_encoding_ohe,
        remove_outliers=remove_outliers,
        outliers_method=outliers_method,
        outliers_threshold=outliers_threshold,
    )
    return s
