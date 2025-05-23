"""
Utility functions for Jupyter notebooks.

This module provides common imports, configurations, and helper functions
for the economic downturn detector notebooks.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import warnings

# Add the parent directory to the path to import the econ_downturn package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from econ_downturn package
from econ_downturn import (
    # Data loading functions
    get_fred_data, get_nber_data, get_all_data, load_merged_data,
    load_umich_data, load_fred_data, load_nber_data,

    # Feature engineering functions
    engineer_features, normalize_data, apply_pca, prepare_data_for_modeling,

    # Advanced feature engineering functions
    engineer_features_with_custom_lags, create_interaction_terms,
    apply_sentiment_transformations, select_features,

    # Model functions
    apply_mda, create_discriminant_time_series,

    # Visualization functions
    plot_indicator_with_recessions, plot_correlation_matrix,
    plot_recession_correlations, plot_feature_importance,
    plot_mda_projection, plot_discriminant_time_series,
    plot_sentiment_vs_indicator, plot_sentiment_correlation_matrix,

    # Utility functions
    setup_logger, load_environment, get_data_paths, get_output_paths
)

def setup_notebook():
    """
    Set up the notebook environment with common configurations.
    """
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('deep')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

    # Display all columns in pandas
    pd.set_option('display.max_columns', None)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Load environment variables
    load_environment()

    # Print setup message
    print("Notebook environment set up successfully.")
    print("Available data paths:")
    for key, path in get_data_paths().items():
        print(f"  {key}: {path}")
    print("\nAvailable output paths:")
    for key, path in get_output_paths().items():
        print(f"  {key}: {path}")

def load_data(use_cached=True):
    """
    Load all data sources and merge them.

    Parameters
    ----------
    use_cached : bool
        Whether to use cached data if available

    Returns
    -------
    pandas.DataFrame
        Merged dataset
    """
    # Get data paths
    data_paths = get_data_paths()

    # Check if merged data exists and use_cached is True
    merged_data_path = data_paths.get('merged_data')
    if use_cached and merged_data_path and os.path.exists(merged_data_path):
        print(f"Loading cached merged data from {merged_data_path}")
        merged_data = load_merged_data()
        print(f"Loaded merged data with shape: {merged_data.shape}")
        return merged_data

    # Otherwise, load and merge data from original sources
    print("Loading data from original sources...")
    merged_data = get_all_data()

    if merged_data is not None and not merged_data.empty:
        print(f"Loaded and merged data with shape: {merged_data.shape}")
        return merged_data
    else:
        print("Failed to load data.")
        return None

def display_data_info(data):
    """
    Display information about the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset to analyze
    """
    if data is None or data.empty:
        print("No data to display.")
        return

    # Display basic information
    print("Dataset Information:")
    print(f"Time Range: {data.index.min()} to {data.index.max()}")
    print(f"Number of Observations: {len(data)}")
    print(f"Number of Features: {data.shape[1]}")

    # Display summary statistics
    print("\nSummary Statistics:")
    display(data.describe())

    # Check for missing values
    print("\nMissing Values:")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    display(missing_df.sort_values('Missing Values', ascending=False))

def save_figure(fig, filename, output_dir=None):
    """
    Save a figure to the output directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename for the figure
    output_dir : str, optional
        Output directory. If None, uses the default images directory.
    """
    if output_dir is None:
        output_dir = get_output_paths()['images_dir']

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
