import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BMIImputer(BaseEstimator, TransformerMixin):
    """Impute Obesity values based on BMI.

    Parameters
    ------------
    bmi_col: str
        Column name for the BMI column

    target: str
        Column name for the target (Obesity) column
    """
    def __init__(self, bmi_col, target):
        self.bmi_col = bmi_col
        self.target = target

    def fit(self, X, y=None):
        # find rows with na as value
        self.impute_map_ = X[X[self.bmi_col].isnull()]
        return self

    def transform(self, X, y=None):
        # make sure fit is done
        check_is_fitted(self, 'impute_map_')

        X_ = X.copy()

        for index, row in X_.iterrows():
            if row[self.target] not in ["yes", "no"]:
                if row[self.bmi_col] >= 30:
                    X_.loc[index, self.target] = "yes"
                else:
                    X_.loc[index, self.target] = "no"

        return X_


class FlexibleImputer(BaseEstimator, TransformerMixin):
    """Flexible imputation of missing values by drawing randomly from known values.

    Approach based on approach by S.v.Buuren, 2018: https://stefvanbuuren.name/fimd/

    Parameters
    ------------
    random_state: int
        Seed for pseudo-random random generation.
    """
    #

    def __init__(self, random_state):
        self.random_state = random_state

    def fit(self, X, y=None):
        """Necessary for interface reason. No real use here."""
        return self

    def transform(self, X, y=None):
        """Fill in missing values by drawing randomly from the known values of the same feature.

        Parameters
        ----------
        X: pd.Dataframe of shape (n_samples, n_features)
            Data to fill in.
        y: not used

        Returns
        -------
        X_: pd.DataFrame of shape (n_samples, n_features)
            Input data as given, with missing values filled in.

        """
        # Only columns that still have NaN's
        X_ = X.copy()
        nan_columns = X_.columns[X_.isnull().any()]
        np.random.seed(self.random_state)
        for col in nan_columns:
            # rate of yes vs no's
            value_ratios = X_[col].value_counts(normalize=True)

            # rows with a missing value
            missing = X_[col].isnull()

            # pick randomly, weighted by ratio
            X_.loc[missing, col] = np.random.choice(value_ratios.index, size=len(X_[missing]), p=value_ratios.values)

        return X_


class BinarizerWithNan(TransformerMixin, BaseEstimator):
    """Binarize data (set feature values to 0 or 1) according to a threshold.

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    This custom implementation of sklearn.preprocessing.Binarizer also allows NaN values,
    which are simply ignored.

    Parameters
    ------------
    threshold: float, default=0.0
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : bool, default=True
        Set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).

    """

    def __init__(self, col_name, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy
        self.col_name = col_name

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        return self

    def transform(self, X, copy=None):
        """Binarize each element of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to binarize, element by element.
        copy : bool
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        col_name = self.col_name
        #    X = pd.DataFrame({col_name: X})

        copy = copy if copy is not None else self.copy
        X_ = X.copy()

        # Set values to 0 or 1 based on threshold
        X_ = self.binarize(X_, col_name, threshold=self.threshold, copy=False)

        return X_

    @staticmethod
    def binarize(X, col_name, threshold=0.0, copy=True):
        """Set values to 0 or 1, based on the given threshold.

        Note: NaN-values are kept as is, and not filled in in any way.

        Parameters
        ----------
        X: pd.Dataframe of shape (n_samples, n_features)
            Input data
        col_name: str
            Column name of column to binarize
        threshold: float
            If value is above threshold, output is 1. If below, output is 0. Default: 0.0
        copy: bool
            Not used.

        Returns
        -------
        X: pd.Dataframe of shape (n_samples, n_features)
            Binarized version of input X.
        """
        conditions = [
            (X[col_name] > threshold),
            (X[col_name] <= threshold),
        ]
        choices = [1, 0]

        # Make decision: 0 if below threshold, 1 if above
        # Note: We keep NaN values
        X_arr = np.select(conditions, choices, default=np.nan)
        X[col_name] = X_arr

        return X


class Debug(BaseEstimator, TransformerMixin):
    # Used as Pipeline step to inspect intermediate results

    def transform(self, X, copy=None):
        print("Debug here")
        print(type(X))
        print(X.shape)
        if isinstance(X, np.ndarray):
            print(X[:5, :].shape)
            print(X[:5, :])
        elif isinstance(X, pd.DataFrame):
            print(X.head(5))
        self.shape = X.shape
        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):
        return self


class SelectFeatures(BaseEstimator, TransformerMixin):
    """Select features based on the numpy array from previous steps in the pipeline.
       Mandatory: list of integers indicating the desired column index.

    We use it to select the same three features we had in Assignment 1: 
    - Urination
    - Gender
    - Irritability

    Parameters
    ------------
    features_index: List[int]
        List of integers indicating the desired column index to extract from the
        nd-numpy array obtained from previous steps in the pipeline.
    """

    # Used as Pipeline step to select features from Project 1

    def __init__(self, features_index):
        self.features_index = features_index

    def fit(self, X, y=None):
        """ Do nothing and return the estimator unchanged.
            This method is just there to implement the usual API and hence
            work in pipelines.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted transformer.
        """
        return self

    def transform(self, X):
        """ Create a new nd-numpy array with only the desired columns using self.features_index

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to perform feature selection.

        Returns
        -------
        X_ : {ndarray, sparse matrix} of shape (n_samples, len(self.features_index))
            Transformed array of only desired features.
        """
        # Make copy to avoid double reference
        X_ = X.copy()
        # Get the feature indexes in a numpy format
        features_index = np.array(self.features_index)
        # Select the desired columns from the nd-numpy array
        X_ = X_[:, features_index]

        return X_