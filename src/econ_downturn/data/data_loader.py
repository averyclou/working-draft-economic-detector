"""
Functions for loading and merging data from different sources.

This module provides utilities to load data from FRED, NBER, UMich, and other sources,
and merge them into a single dataset for analysis.
"""

import os
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)


def load_fred_data(file_path='data/fred/all_indicators.csv'):
    """
    Load FRED economic indicators data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file with FRED data

    Returns
    -------
    pandas.DataFrame
        DataFrame with FRED economic indicators
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded FRED data with shape: {data.shape}")
        return data
    else:
        logger.warning(f"FRED data file not found at {file_path}")
        return pd.DataFrame()


def load_nber_data(file_path='data/nber/recession_indicator.csv'):
    """
    Load NBER recession indicator data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file with NBER recession indicator

    Returns
    -------
    pandas.DataFrame
        DataFrame with recession indicator
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded NBER recession data with shape: {data.shape}")
        return data
    else:
        logger.warning(f"NBER data file not found at {file_path}")
        return pd.DataFrame()


def load_umich_data(file_path='data/umich/all_sentiment.csv'):
    """
    Load University of Michigan Consumer Sentiment data.

    Parameters
    ----------
    file_path : str
        Path to the CSV file with UMich data

    Returns
    -------
    pandas.DataFrame
        DataFrame with consumer sentiment data
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded UMich data with shape: {data.shape}")
        return data
    else:
        logger.warning(f"UMich data file not found at {file_path}")
        return pd.DataFrame()


def merge_datasets(datasets, output_path=None):
    """
    Merge multiple datasets into a single dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets to merge, with keys as dataset names
    output_path : str, optional
        Path to save the merged dataset, if None, the dataset is not saved

    Returns
    -------
    pandas.DataFrame
        Merged dataset
    """
    merged_data = None

    for name, data in datasets.items():
        if data.empty:
            logger.warning(f"Dataset '{name}' is empty and will be skipped")
            continue

        if merged_data is None:
            merged_data = data.copy()
            logger.info(f"Initialized merged dataset with '{name}' data")
        else:
            merged_data = merged_data.join(data, how='outer')
            logger.info(f"Added '{name}' data to merged dataset")

    if merged_data is not None:
        logger.info(f"Merged dataset shape: {merged_data.shape}")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            merged_data.to_csv(output_path)
            logger.info(f"Saved merged dataset to {output_path}")
    else:
        logger.warning("No data available to merge")

    return merged_data


def load_merged_data(file_path='data/processed/merged_data.csv'):
    """
    Load the merged dataset.

    Parameters
    ----------
    file_path : str
        Path to the CSV file with merged data

    Returns
    -------
    pandas.DataFrame
        DataFrame with merged data
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logger.info(f"Loaded merged data with shape: {data.shape}")
        return data
    else:
        logger.warning(f"Merged data file not found at {file_path}")
        return pd.DataFrame()


def get_all_data(fred_path='data/fred/all_indicators.csv',
                nber_path='data/nber/recession_indicator.csv',
                umich_path='data/umich/all_sentiment.csv',
                output_path='data/processed/merged_data.csv'):
    """
    Convenience function to load and merge all data in one step.

    Parameters
    ----------
    fred_path : str
        Path to the CSV file with FRED data
    nber_path : str
        Path to the CSV file with NBER recession indicator
    umich_path : str
        Path to the CSV file with UMich consumer sentiment data
    output_path : str
        Path to save the merged dataset

    Returns
    -------
    pandas.DataFrame
        Merged dataset
    """
    # Load data from different sources
    fred_data = load_fred_data(fred_path)
    nber_data = load_nber_data(nber_path)
    umich_data = load_umich_data(umich_path)

    # Merge datasets
    datasets = {
        'FRED': fred_data,
        'NBER': nber_data,
        'UMICH': umich_data
    }

    merged_data = merge_datasets(datasets, output_path)

    return merged_data
