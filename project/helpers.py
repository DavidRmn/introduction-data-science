from typing import Union

import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, threshold: float = 3.0):
        """
        Initialize the transformer with the target column for outlier removal and the Z-score threshold.
        
        Args:
            target (str): The target column for outlier removal.
            threshold (float): Z-score threshold to consider a data point an outlier.
        """
        self.target = target
        self.threshold = threshold
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. This transformer does not need to learn anything from the data,
        so it just returns itself.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Not used, for compatibility with scikit-learn's transformer requirements.
        
        Returns:
            self: The fitted transformer.
        """
        return self
    
    def transform(self, X):
        """
        Remove outliers from the DataFrame based on the Z-score threshold in the target column.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with outliers removed from the specified target column.
        """
        X = X.copy()  # Work on a copy of the DataFrame to avoid altering original data
        if self.target in X.columns:
            # Calculate Z-scores for the target column
            z_scores = zscore(X[self.target].dropna())
            # Identify outliers
            outliers = abs(z_scores) > self.threshold
            # Drop outliers
            X = X[~X[self.target].index.isin(X[self.target].dropna().index[outliers])]
            print(f"{X.shape[0]} rows remaining after removing outliers.")
        return X
    
class RegressionImputationOutlier(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, feature_columns: list, threshold: float = 3.0):
        """
        Initialize the transformer with the target column for outlier imputation, the feature columns
        to use for predicting the target, and the Z-score threshold.
        
        Args:
            target (str): The target column for outlier imputation.
            feature_columns (list): The list of column names to be used as features for regression.
            threshold (float): Z-score threshold to consider a data point an outlier.
        """
        self.target = target
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.model = LinearRegression()
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. Fits a linear regression model using non-outlier data points.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Not used, for compatibility with scikit-learn's transformer requirements.
        
        Returns:
            self: The fitted transformer.
        """
        X = X.copy()
        if self.target in X.columns and all(col in X.columns for col in self.feature_columns):
            # Calculate Z-scores for the target column
            z_scores = zscore(X[self.target].dropna())
            non_outliers = abs(z_scores) <= self.threshold
            
            # Fit the model using non-outlier data
            self.model.fit(X.loc[non_outliers, self.feature_columns], X.loc[non_outliers, self.target])
        return self
    
    def transform(self, X):
        """
        Impute outliers in the DataFrame based on predictions from the fitted linear regression model.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with outliers in the specified target column imputed.
        """
        X = X.copy()
        if self.target in X.columns and all(col in X.columns for col in self.feature_columns):
            # Calculate Z-scores for the target column again
            z_scores = zscore(X[self.target].dropna())
            outliers = abs(z_scores) > self.threshold
            
            # Predict and impute values for outliers using the fitted model
            predicted_values = self.model.predict(X.loc[outliers, self.feature_columns])
            X.loc[X[self.target].dropna().index[outliers], self.target] = predicted_values.astype(int)
        return X

class DatetimeConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list, date_format=None):
        """
        Initialize the transformer with target columns and optional datetime format.
        
        Args:
            columns (list): List of column names to convert to datetime.
            date_format (str, optional): The datetime format to use for conversion. Defaults to None.
        """
        self.columns = columns
        self.date_format = date_format
    
    def fit(self, X, y=None):
        """
        Fit method. This transformer does not need to learn anything from the data,
        so it just returns itself.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Ignored. Defaults to None.
        
        Returns:
            self: The fitted transformer.
        """
        return self
    
    def transform(self, X):
        """
        Apply the datetime conversion to the target columns.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with the target columns converted to datetime.
        """
        # Ensure we don't modify the original DataFrame
        X = X.copy()
        
        # Convert each target column to datetime
        for column in self.columns:
            X[column] = pd.to_datetime(X[column], format=self.date_format)
        
        return X

class SeasonImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, date_column: str):
        """
        Initialize the transformer with the name of the target column for the season
        and the date column from which the season will be determined.
        
        Args:
            target_column (str): The name of the column to store the imputed season.
            date_column (str): The name of the date column to determine the season from.
        """
        self.target_column = target_column
        self.date_column = date_column
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. Since this transformer does not need to learn
        anything from the data, it just returns itself.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Not used, for compatibility with scikit-learn's transformer requirements.
        
        Returns:
            self: The fitted transformer.
        """
        return self  # No fitting necessary
    
    def transform(self, X):
        """
        Apply the season imputation to the DataFrame.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with the season imputed.
        """
        # Ensure we don't modify the original DataFrame
        X = X.copy()
        
        # Define the function to determine the season based on the day of the year
        def get_season(date):
            day_of_year = date.timetuple().tm_yday
            if 80 <= day_of_year < 172:
                return 2  # Spring
            elif 172 <= day_of_year < 265:
                return 3  # Summer
            elif 265 <= day_of_year < 355:
                return 4  # Fall
            else:
                return 1  # Winter
        
        # Check if the date column exists and apply the season calculation
        if self.date_column in X.columns:
            X[self.target_column] = X[self.date_column].apply(lambda x: get_season(x) if pd.notnull(x) else x)
        
        return X

class ThresholdRowRemover(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, threshold: Union[int, float] = 0.0):
        """
        Initialize the transformer with the target column for row removal and the threshold value.
        
        Args:
            target (str): The target column for row removal.
            threshold (Union[int, float]): The threshold below which rows will be removed.
        """
        self.target_column = target_column
        self.threshold = threshold
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. This transformer does not need to learn anything from the data,
        so it just returns itself.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Not used, for compatibility with scikit-learn's transformer requirements.
        
        Returns:
            self: The fitted transformer.
        """
        return self  # No fitting necessary
    
    def transform(self, X):
        """
        Remove rows from the DataFrame where the target column's value is below the threshold.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with rows removed where the target column's value is below the threshold.
        """
        X = X.copy()  # Work on a copy of the DataFrame to avoid altering original data
        if self.target_column in X.columns:
            X = X[X[self.target_column] >= self.threshold]
        return X