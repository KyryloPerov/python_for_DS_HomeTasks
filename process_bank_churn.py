"""
Module for preprocessing Bank Customer Churn data.

Provides functions for training data preprocessing and new (test) data
transformation using already fitted scaler and encoder.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional


TARGET_COL = 'Exited'


def split_data(
    raw_df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw dataframe into train and validation subsets with stratification.

    Args:
        raw_df: Input dataframe with raw data.
        test_size: Proportion of the validation set (0 to 1).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[TARGET_COL]
    )
    return train_df, val_df


def get_column_types(
    train_inputs: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Detect and separate numeric, binary numeric, and categorical columns.

    Removes 'Surname' from categorical columns as it is not used as a feature.
    Binary numeric columns (e.g. HasCrCard, IsActiveMember) are separated
    from continuous numeric columns to avoid unnecessary scaling.

    Args:
        train_inputs: Training input dataframe (without target column).

    Returns:
        Tuple of:
          - numeric_cols_cleaned: Continuous numeric columns (to be scaled).
          - binary_numeric_cols: Binary numeric columns (0/1, not scaled).
          - categorical_cols: Categorical columns excluding 'Surname'.
    """
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    if 'Surname' in categorical_cols:
        categorical_cols.remove('Surname')

    binary_numeric_cols = [
        col for col in numeric_cols if train_inputs[col].nunique() == 2
    ]
    numeric_cols_cleaned = [
        col for col in numeric_cols if col not in binary_numeric_cols
    ]

    return numeric_cols_cleaned, binary_numeric_cols, categorical_cols


def scale_numeric_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    numeric_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale continuous numeric features using StandardScaler.

    The scaler is fitted on training data only and then applied to both
    training and validation sets to prevent data leakage.

    Args:
        train_inputs: Training input dataframe.
        val_inputs: Validation input dataframe.
        numeric_cols: List of numeric columns to scale.

    Returns:
        Tuple of (train_inputs, val_inputs, scaler) with scaled values
        and the fitted scaler object.
    """
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])

    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()

    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])

    return train_inputs, val_inputs, scaler


def encode_categorical_features(
    train_inputs: pd.DataFrame,
    val_inputs: pd.DataFrame,
    categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list[str]]:
    """
    Apply One-Hot Encoding to categorical features.

    The encoder is fitted on training data only and then applied to both
    training and validation sets to prevent data leakage.

    Args:
        train_inputs: Training input dataframe.
        val_inputs: Validation input dataframe.
        categorical_cols: List of categorical columns to encode.

    Returns:
        Tuple of (train_inputs, val_inputs, encoder, encoded_cols) with
        encoded columns added, the fitted encoder, and new column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    train_inputs = train_inputs.copy()
    val_inputs = val_inputs.copy()

    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])

    return train_inputs, val_inputs, encoder, encoded_cols


def preprocess_data(raw_df: pd.DataFrame, scale_numeric: bool = True) -> dict:
    """
    Main preprocessing function for training data.

    Performs the full preprocessing pipeline:
      1. Train/validation split with stratification.
      2. Feature selection (columns 1 to -1, excluding target).
      3. Column type detection (numeric, binary, categorical).
      4. Scaling of continuous numeric features (StandardScaler) — optional.
      5. One-Hot Encoding of categorical features.
      6. Assembly of final feature matrices X_train and X_val.

    Args:
        raw_df: Raw input dataframe loaded from train.csv (index = 'id').
        scale_numeric: Whether to scale continuous numeric features.
            Set to True for models sensitive to feature scale (e.g. Logistic
            Regression, SVM, KNN). Set to False for tree-based models
            (e.g. Decision Tree, Random Forest) where scaling is not needed.
            Defaults to True.

    Returns:
        Dictionary with keys:
          - 'X_train': pd.DataFrame — training features ready for the model.
          - 'train_targets': pd.Series — training target values.
          - 'X_val': pd.DataFrame — validation features ready for the model.
          - 'val_targets': pd.Series — validation target values.
          - 'input_cols': list[str] — ordered list of feature column names in X.
          - 'scaler': fitted StandardScaler instance, or None if scale_numeric=False.
          - 'encoder': fitted OneHotEncoder instance.
    """
    # 1. Split into train and validation
    train_df, val_df = split_data(raw_df)

    # 2. Select feature columns (skip first col = id, last col = Exited)
    input_cols = list(train_df.columns)[1:-1]

    train_inputs = train_df[input_cols].copy()
    train_targets = train_df[TARGET_COL].copy()
    val_inputs = val_df[input_cols].copy()
    val_targets = val_df[TARGET_COL].copy()

    # 3. Detect column types
    numeric_cols_cleaned, binary_numeric_cols, categorical_cols = get_column_types(train_inputs)

    # 4. Scale continuous numeric features (optional)
    if scale_numeric:
        train_inputs, val_inputs, scaler = scale_numeric_features(
            train_inputs, val_inputs, numeric_cols_cleaned
        )
    else:
        scaler = None

    # 5. Encode categorical features
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(
        train_inputs, val_inputs, categorical_cols
    )

    # 6. Assemble final feature matrices
    final_cols = numeric_cols_cleaned + binary_numeric_cols + encoded_cols
    X_train = train_inputs[final_cols]
    X_val = val_inputs[final_cols]

    return {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": final_cols,
        "scaler": scaler,
        "encoder": encoder,
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
    input_cols: list[str],
) -> pd.DataFrame:
    """
    Preprocess new (test) data using already fitted scaler and encoder.

    Use this function to transform test.csv before making predictions.
    The scaler and encoder must be obtained from preprocess_data() to
    ensure consistent transformation and avoid data leakage.

    Args:
        new_df: Raw dataframe with new data (e.g. loaded from test.csv).
        scaler: Fitted StandardScaler from preprocess_data().
        encoder: Fitted OneHotEncoder from preprocess_data().
        input_cols: Ordered list of feature column names from preprocess_data().

    Returns:
        pd.DataFrame X_new with preprocessed features ready for model.predict().
    """
    inputs = new_df.copy()

    # Detect column types (same logic as in preprocess_data)
    numeric_cols_cleaned, binary_numeric_cols, categorical_cols = get_column_types(inputs)

    # Apply fitted scaler (only if scaling was used during training)
    if scaler is not None:
        inputs[numeric_cols_cleaned] = scaler.transform(inputs[numeric_cols_cleaned])

    # Apply fitted encoder
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])

    # Return columns in the same order as during training
    X_new = inputs[input_cols]

    return X_new
