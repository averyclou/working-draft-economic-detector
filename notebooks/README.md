# Economic Downturn Detector Notebooks

This directory contains Jupyter notebooks for analyzing economic indicators and developing a recession prediction model using Multiple Discriminant Analysis (MDA).

## Notebook Structure

The notebooks follow a logical progression:

1. **01_data_exploration.ipynb**: Explores the economic indicators data and their relationships with recession periods
2. **02_feature_engineering.ipynb**: Creates lag variables and other features to capture leading indicators
3. **03_multiple_discriminant_analysis.ipynb**: Applies MDA to identify the most significant predictors of recessions
4. **04_combined_data_analysis.ipynb**: Analyzes the combined dataset with consumer sentiment data
5. **05_feature_engineering_optimization.ipynb**: Optimizes feature engineering with advanced techniques

## DRY Structure

The notebooks have been optimized to follow the DRY (Don't Repeat Yourself) principle by leveraging the `econ_downturn` package. This is achieved through:

1. **notebook_utils.py**: A utility module that provides common imports, configurations, and helper functions
2. **econ_downturn package**: The main package that contains all the core functionality

## How to Use

1. Start by running the notebooks in order (01 → 05)
2. Each notebook builds on the previous ones, so it's important to follow the sequence
3. Use the `notebook_utils.py` module for common functionality
4. For custom analysis, import specific functions from the `econ_downturn` package

## Utility Functions

The `notebook_utils.py` module provides several utility functions:

- `setup_notebook()`: Sets up the notebook environment with common configurations
- `load_data(use_cached=True)`: Loads all data sources and merges them
- `display_data_info(data)`: Displays information about the dataset
- `save_figure(fig, filename, output_dir=None)`: Saves a figure to the output directory

## Example Usage

```python
# Import notebook utilities
from notebook_utils import (
    setup_notebook, load_data, display_data_info, save_figure,
    plot_indicator_with_recessions, plot_correlation_matrix
)

# Set up the notebook environment
setup_notebook()

# Load data
merged_data = load_data()

# Display information about the dataset
display_data_info(merged_data)

# Plot an indicator with recession periods
fig = plot_indicator_with_recessions(
    merged_data, 
    'UNEMPLOYMENT',
    title='Unemployment Rate with Recession Periods'
)

# Save the figure
save_figure(fig, "unemployment_over_time.png")
```

## Advanced Feature Engineering

For advanced feature engineering, use the functions from the `econ_downturn.features.advanced_engineering` module:

- `engineer_features_with_custom_lags()`: Creates lag variables with custom lag periods
- `create_interaction_terms()`: Creates interaction terms between sentiment and economic indicators
- `apply_sentiment_transformations()`: Applies various transformations to sentiment data
- `select_features()`: Selects the most predictive features

These functions are imported through the `notebook_utils.py` module for convenience.
