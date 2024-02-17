# IDSTools Package Configuration Documentation

This documentation provides a comprehensive overview of the default configuration settings for the IDSTools package. The IDSTools package is designed to facilitate various stages of a data science project, including data exploration, data preparation, and model optimization.

## Default Configuration

The default configuration is structured into three main modules:

1. **Data Exploration**
2. **Data Preparation**
3. **Model Optimization**

Each module is configurable to suit different datasets and project requirements.

### 1. Data Exploration Module

The `data_explorer` section configures the `DataExplorer` class, which is responsible for the initial analysis of the dataset.

#### Configuration Options:

- `target_data`: Reference to the `_idstools_data` object containing the dataset.
- `label`: Target variable for analysis.
- `index`: Index column for the dataset.
- `input_path`: Path to the input data file.
- `input_delimiter`: Delimiter used in the input file.
- `output_path`: Output path for saving results. If null, results are not saved to a file.

#### Pipeline:

The pipeline allows enabling or disabling various analyses:

- `descriptive_analysis`: Enables analysis of basic descriptive statistics (mean, median, etc.).
- `missing_value_analysis`: Enables analysis of missing values in the dataset.
- `correlation_analysis`: Enables analysis of feature correlations.
- `outlier_analysis`: Enables detection and analysis of outliers.
- `distribution_analysis`: Enables analysis of feature distributions.
- `scatter_analysis`: Enables scatter plot analysis for feature relationships.

### 2. Data Preparation Module

The `data_preparation` section configures the `DataPreparation` class for preprocessing the dataset.

#### Configuration Options:

Similar to the Data Explorer module, with additional options for the data preparation pipeline including data imputation, encoding, and feature selection.

#### Pipeline:

- `_SimpleImputer`: Configures missing value imputation.
- `_OneHotEncoder`: Configures one-hot encoding for categorical variables.
- `_FeatureDropper`: Allows dropping specific features.
- `_CustomTransformer`: Applies custom transformations to the dataset.

### 3. Model Optimization Module

The `model_optimization` section is used for tuning and evaluating models.

#### Configuration Options:

- `target_data`: Reference to the `_idstools_data` object.
- `input_path`: Path to the input data file.
- `input_delimiter`: Delimiter used in the input file.
- `output_path`: Path for saving optimization results.

#### Evaluation:

- `metric`: Metric for model evaluation, such as Root Mean Squared Error.
- `cv`: Number of folds for cross-validation.

## TargetData Configuration

This is a special configuration under the default settings, which specifies the dataset details.

- `input_path`: Path to the dataset.
- `input_delimiter`: Delimiter used in the dataset.
- `output_path`: Path for saving any output.
- `label`: Name of the target variable.
- `index`: Column used as the dataset index.

This configuration serves as the basis for dataset handling across various modules of the IDSTools package.
