#!/usr/bin/env python3
"""
Functions to fix and regenerate visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def preprocess_data(data):
    """
    Preprocess the data to handle missing values and ensure consistent frequency.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data

    Returns
    -------
    pandas.DataFrame
        Preprocessed data
    """
    # Convert to monthly frequency for consistency
    # First, ensure the index is datetime
    data.index = pd.to_datetime(data.index)

    # For each column, resample to monthly frequency
    monthly_data = {}

    for col in data.columns:
        # Skip columns with all NaN values
        if data[col].isna().all():
            continue

        # For economic indicators, use the last value of the month
        if col in ['GDP', 'UNEMPLOYMENT', 'CPI', 'FED_FUNDS', 'YIELD_CURVE',
                  'INDUSTRIAL_PROD', 'RETAIL_SALES', 'HOUSING_STARTS',
                  'CONSUMER_SENTIMENT', 'SENTIMENT', 'INFLATION_1Y']:
            # Resample to monthly frequency, using last value
            series = data[col].resample('M').last()

        # For initial claims (weekly), use the mean of the month
        elif col == 'INITIAL_CLAIMS':
            series = data[col].resample('M').mean()

        # For recession indicator, use max (if any day in month is recession, month is recession)
        elif col == 'recession':
            series = data[col].resample('M').max()

        else:
            # Default to last value
            series = data[col].resample('M').last()

        monthly_data[col] = series

    # Combine all series into a single DataFrame
    monthly_df = pd.DataFrame(monthly_data)

    # Forward fill missing values for economic indicators (carry forward last known value)
    for col in monthly_df.columns:
        if col != 'recession':  # Don't fill recession indicator
            monthly_df[col] = monthly_df[col].fillna(method='ffill')

    # Ensure recession column exists and is filled with 0 for NaN
    if 'recession' in monthly_df.columns:
        monthly_df['recession'] = monthly_df['recession'].fillna(0).astype(int)

    return monthly_df

def fix_indicator_plots(data, output_dir, logger=None):
    """
    Fix the indicator plots.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed data
    output_dir : str
        Output directory for images
    logger : logging.Logger, optional
        Logger instance
    """
    from econ_downturn.visualization.plotting import plot_indicator_with_recessions
    
    indicators = {
        'GDP': 'Gross Domestic Product',
        'UNEMPLOYMENT': 'Unemployment Rate',
        'CPI': 'Consumer Price Index',
        'FED_FUNDS': 'Federal Funds Rate'
    }

    for indicator, title in indicators.items():
        if indicator in data.columns:
            if logger:
                logger.info(f"Fixing {indicator} plot...")

            # Create the plot
            fig = plot_indicator_with_recessions(
                data,
                indicator,
                title=f"{title} Over Time",
                save_path=os.path.join(output_dir, f"{indicator.lower()}_over_time.png")
            )

            plt.close(fig)

def fix_sentiment_plots(data, output_dir, logger=None):
    """
    Fix the sentiment plots.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed data
    output_dir : str
        Output directory for images
    logger : logging.Logger, optional
        Logger instance
    """
    from econ_downturn.visualization.plotting import plot_indicator_with_recessions, plot_sentiment_vs_indicator
    
    # Fix consumer sentiment plot
    if 'SENTIMENT' in data.columns:
        if logger:
            logger.info("Fixing consumer sentiment plot...")

        # Create the plot
        fig = plot_indicator_with_recessions(
            data,
            'SENTIMENT',
            title='Consumer Sentiment with Recession Periods',
            save_path=os.path.join(output_dir, 'consumer_sentiment.png')
        )

        plt.close(fig)

    # Fix sentiment vs unemployment plot
    if 'SENTIMENT' in data.columns and 'UNEMPLOYMENT' in data.columns:
        if logger:
            logger.info("Fixing sentiment vs unemployment plot...")

        # Create the plot
        fig = plot_sentiment_vs_indicator(
            data,
            sentiment_col='SENTIMENT',
            indicator_col='UNEMPLOYMENT',
            save_path=os.path.join(output_dir, 'sentiment_vs_unemployment.png')
        )

        plt.close(fig)

    # Fix sentiment vs GDP plot
    if 'SENTIMENT' in data.columns and 'GDP' in data.columns:
        if logger:
            logger.info("Fixing sentiment vs GDP plot...")

        # Create the plot
        fig = plot_sentiment_vs_indicator(
            data,
            sentiment_col='SENTIMENT',
            indicator_col='GDP',
            save_path=os.path.join(output_dir, 'sentiment_vs_gdp.png')
        )

        plt.close(fig)

def create_correlation_heatmap(data, output_dir, logger=None):
    """
    Create a correlation heatmap for the economic indicators.

    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed data
    output_dir : str
        Output directory for images
    logger : logging.Logger, optional
        Logger instance
    """
    from econ_downturn.visualization.plotting import plot_correlation_matrix
    
    if logger:
        logger.info("Creating correlation heatmap...")

    # Select only numeric columns for correlation
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Remove recession column if present (it's a binary indicator)
    if 'recession' in numeric_cols:
        numeric_cols.remove('recession')

    # Create a subset of data with only the numeric columns
    data_subset = data[numeric_cols].copy()

    # Create the correlation heatmap
    fig = plot_correlation_matrix(
        data_subset,
        figsize=(14, 12),
        annot=True,
        save_path=os.path.join(output_dir, 'correlation_heatmap.png')
    )

    plt.close(fig)

def fix_all_visualizations(data, output_dir, logger=None):
    """
    Fix all visualizations.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data
    output_dir : str
        Output directory for images
    logger : logging.Logger, optional
        Logger instance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess data
    if logger:
        logger.info("Preprocessing data...")
    processed_data = preprocess_data(data)

    # Fix indicator plots
    if logger:
        logger.info("Fixing indicator plots...")
    fix_indicator_plots(processed_data, output_dir, logger)

    # Fix sentiment plots
    if logger:
        logger.info("Fixing sentiment plots...")
    fix_sentiment_plots(processed_data, output_dir, logger)

    # Create correlation heatmap
    create_correlation_heatmap(processed_data, output_dir, logger)

    if logger:
        logger.info("Visualization fixes completed successfully.")
