"""
Configuration utilities for the economic downturn detector.

This module provides functions to load and manage configuration settings.
"""

import os
from dotenv import load_dotenv


def load_environment():
    """
    Load environment variables from .env file.
    
    Returns
    -------
    dict
        Dictionary of environment variables
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys
    fred_api_key = os.getenv('FRED_API_KEY')
    
    # Create config dictionary
    config = {
        'fred_api_key': fred_api_key,
        'data_dir': os.getenv('DATA_DIR', 'data'),
        'output_dir': os.getenv('OUTPUT_DIR', 'docs/images')
    }
    
    return config


def get_data_paths(base_dir='data'):
    """
    Get paths to data directories and files.
    
    Parameters
    ----------
    base_dir : str
        Base data directory
        
    Returns
    -------
    dict
        Dictionary of data paths
    """
    paths = {
        'fred_dir': os.path.join(base_dir, 'fred'),
        'nber_dir': os.path.join(base_dir, 'nber'),
        'processed_dir': os.path.join(base_dir, 'processed'),
        'fred_all_indicators': os.path.join(base_dir, 'fred', 'all_indicators.csv'),
        'nber_recession_indicator': os.path.join(base_dir, 'nber', 'recession_indicator.csv'),
        'merged_data': os.path.join(base_dir, 'processed', 'merged_data.csv'),
        'data_with_features': os.path.join(base_dir, 'processed', 'data_with_features.csv'),
        'data_normalized': os.path.join(base_dir, 'processed', 'data_normalized.csv'),
        'data_pca': os.path.join(base_dir, 'processed', 'data_pca.csv')
    }
    
    return paths


def get_output_paths(base_dir='docs/images'):
    """
    Get paths to output directories and files.
    
    Parameters
    ----------
    base_dir : str
        Base output directory
        
    Returns
    -------
    dict
        Dictionary of output paths
    """
    paths = {
        'images_dir': base_dir,
        'feature_importance': os.path.join(base_dir, 'feature_importance.png'),
        'mda_projection': os.path.join(base_dir, 'mda_projection.png'),
        'discriminant_time': os.path.join(base_dir, 'discriminant_time.png'),
        'correlation_matrix': os.path.join(base_dir, 'correlation_matrix.png'),
        'recession_correlations': os.path.join(base_dir, 'recession_correlations.png'),
        'pca_explained_variance': os.path.join(base_dir, 'pca_explained_variance.png')
    }
    
    return paths
