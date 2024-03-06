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
        self.outliers = None
        
    def fit(self, X, y=None):
        """
        Fit method for the transformer. Determines which data points in the target column are outliers
        based on the Z-score threshold.

        Args:
            X (pd.DataFrame): The input DataFrame with the target column.
            y (None, optional): Not used, for compatibility with scikit-learn's transformer requirements.

        Returns:
            self: The fitted transformer, with outliers identified.
        """
        if self.target in X.columns:
            # Calculate Z-scores for the target column
            z_scores = zscore(X[self.target].dropna())
            # Identify outliers
            self.outliers = abs(z_scores) > self.threshold
        return self
    
    def transform(self, X, y=None):
        """
        Transform method for the transformer. Removes outliers from the DataFrame based on the logic
        determined in the fit method.

        Args:
            X (pd.DataFrame): The input DataFrame to transform by removing outliers.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed from the specified target column.
        """
        X = X.copy()  # Work on a copy of the DataFrame to avoid altering original data
        if self.outliers is not None:
            # Remove the outliers identified in the fit method
            X = X[~X[self.target].index.isin(X[self.target].dropna().index[self.outliers])]
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
    def __init__(self, target_column: str, date_format=None):
        """
        Initialize the transformer with target columns and optional datetime format.
        
        Args:
            columns (list): List of column names to convert to datetime.
            date_format (str, optional): The datetime format to use for conversion. Defaults to None.
        """
        self.target_column = target_column
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
        X[self.target_column] = pd.to_datetime(X[self.target_column], format=self.date_format)
        
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
    
class ThresholdImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, threshold: Union[int, float] = 0.0, impute_above: bool = False):
        """
        Initialize the transformer with the target column for imputation, the threshold value, and the direction of imputation.
        
        Args:
            target_column (str): The target column for imputation.
            threshold (Union[int, float]): The threshold value for imputation.
            impute_above (bool): If True, values above the threshold will be imputed; if False, values below the threshold will be imputed.
        """
        self.target_column = target_column
        self.threshold = threshold
        self.impute_above = impute_above
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. Calculates the mean of the target column values either above or below the threshold based on the direction of imputation.
        
        Args:
            X (pd.DataFrame): The input DataFrame.
            y (None, optional): Not used, for compatibility with scikit-learn's transformer requirements.
        
        Returns:
            self: The fitted transformer.
        """
        if self.target_column in X.columns:
            if self.impute_above:
                # Calculate mean of values below the threshold for imputing values above it
                self.mean_value_ = X.loc[X[self.target_column] <= self.threshold, self.target_column].mean()
            else:
                # Calculate mean of values above the threshold for imputing values below it
                self.mean_value_ = X.loc[X[self.target_column] >= self.threshold, self.target_column].mean()
        return self
    
    def transform(self, X):
        """
        Apply the imputation to the target column based on the threshold value and direction of imputation.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with values in the target column imputed based on the specified criteria.
        """
        X = X.copy()  # Work on a copy of the DataFrame to avoid altering original data
        if self.target_column in X.columns:
            if self.impute_above:
                # Impute values above the threshold
                X.loc[X[self.target_column] > self.threshold, self.target_column] = self.mean_value_
            else:
                # Impute values below the threshold
                X.loc[X[self.target_column] < self.threshold, self.target_column] = self.mean_value_
        return X
    
class WeekdayMapper(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, date_column: str):
        """
        Initialize the transformer with the name of the target column for the weekday
        and the date column from which the weekday will be determined.
        
        Args:
            target_column (str): The name of the column to store the mapped weekday.
            date_column (str): The name of the date column to determine the weekday from.
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
        Apply the weekday mapping to the DataFrame based on the date column.
        
        Args:
            X (pd.DataFrame): The input DataFrame to transform.
        
        Returns:
            pd.DataFrame: The DataFrame with the weekday column mapped.
        """
        # Ensure we don't modify the original DataFrame
        X = X.copy()
        
        # Check if the date column exists and map the weekday
        if self.date_column in X.columns:
            # Maps the weekday based on the date column, where Monday=0, Sunday=6, then adjusts to Monday=1, Sunday=0
            X[self.target_column] = (X[self.date_column].dt.dayofweek + 1) % 7
        
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
    
class GroupOneHotEncodedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, grouping_info):
        """
        Initialize the transformer with the grouping information for one-hot encoded columns.
        
        Parameters:
        - grouping_info: Dictionary where keys are the new column names for the groups
                         and values are lists of columns to be grouped.
        """
        self.grouping_info = grouping_info
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. This transformer does not need to learn anything from the data,
        so it just returns itself.
        
        Parameters:
        - X: pandas DataFrame containing one-hot encoded columns.
        - y: Not used, for compatibility with scikit-learn's transformer requirements.
        
        Returns:
        - self: The fitted transformer.
        """
        return self  # No fitting necessary
    
    def transform(self, X):
        """
        Apply the grouping to the one-hot encoded columns based on the provided grouping information.
        
        Parameters:
        - X: pandas DataFrame to transform.
        
        Returns:
        - DataFrame with grouped columns.
        """
        # Ensure we don't modify the original DataFrame
        X_transformed = X.copy()
        
        for new_col, columns_to_group in self.grouping_info.items():
            # Create a new column for the group, using `max` as an example aggregation
            X_transformed[new_col] = X_transformed[columns_to_group].max(axis=1)
            
            # Drop the original columns that were grouped
            X_transformed.drop(columns=columns_to_group, inplace=True)
        
        return X_transformed
    
def clean_column_names(df):
    """
    Cleans DataFrame column names by removing 'remainder__' prefixes.

    Args:
        df (pandas.DataFrame): The DataFrame with column names to clean.

    Returns:
        pandas.DataFrame: A DataFrame with cleaned column names.
    """
    # Use a dictionary comprehension to create a mapping of old to new names
    rename_map = {col: '__'.join(col.split('__')[-1:]) for col in df.columns}
    
    # Rename the columns using the mapping
    df_renamed = df.rename(columns=rename_map)
    
    return df_renamed