#!/usr/bin/env python3
"""
Example script demonstrating the usage of the econ_downturn package.

This script shows how to:
1. Fetch economic data from FRED and NBER
2. Perform feature engineering
3. Apply Multiple Discriminant Analysis (MDA)
4. Visualize the results
"""

import os
import logging
from dotenv import load_dotenv

# Import the econ_downturn package
from econ_downturn import (
    get_fred_data, get_nber_data, get_all_data,
    engineer_features, normalize_data, apply_pca,
    apply_mda, create_discriminant_time_series,
    plot_indicator_with_recessions, plot_correlation_matrix,
    plot_feature_importance, plot_mda_projection, plot_discriminant_time_series,
    setup_logger, load_environment, get_output_paths
)

# Set up logging
logger = setup_logger(level=logging.INFO)

def main():
    """Run the complete analysis pipeline."""
    # Load environment variables
    load_dotenv()
    config = load_environment()
    output_paths = get_output_paths()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_paths['images_dir'], exist_ok=True)
    
    # Step 1: Fetch data
    logger.info("Step 1: Fetching data...")
    
    # Check if we have a FRED API key
    fred_api_key = config.get('fred_api_key')
    if not fred_api_key:
        logger.error("FRED API key not found. Set the FRED_API_KEY environment variable.")
        return
    
    # Fetch FRED data
    fred_data = get_fred_data(fred_api_key)
    
    # Fetch NBER recession data
    nber_data = get_nber_data()
    
    # Get all data merged
    merged_data = get_all_data()
    
    if merged_data is None or merged_data.empty:
        logger.error("Failed to load or merge data.")
        return
    
    logger.info(f"Merged data shape: {merged_data.shape}")
    
    # Step 2: Feature engineering
    logger.info("Step 2: Performing feature engineering...")
    
    # Engineer features
    data_with_features = engineer_features(merged_data)
    
    # Step 3: Normalize data and apply PCA
    logger.info("Step 3: Normalizing data and applying PCA...")
    
    # Prepare data for modeling
    modeling_data = prepare_data_for_modeling(data_with_features)
    
    # Step 4: Apply MDA
    logger.info("Step 4: Applying Multiple Discriminant Analysis...")
    
    # Apply MDA to normalized data
    normalized_data = modeling_data['normalized']
    
    if 'recession' not in normalized_data.columns:
        logger.error("Recession indicator not found in the data.")
        return
    
    X = normalized_data.drop(columns=['recession'])
    y = normalized_data['recession']
    
    mda_results = apply_mda(X, y)
    
    # Create discriminant time series
    discriminant_df = create_discriminant_time_series(
        mda_results['model'], X, y
    )
    
    # Step 5: Visualize results
    logger.info("Step 5: Visualizing results...")
    
    # Plot an economic indicator with recession periods
    plot_indicator_with_recessions(
        merged_data, 
        'UNEMPLOYMENT', 
        title='Unemployment Rate with Recession Periods',
        save_path=os.path.join(output_paths['images_dir'], 'unemployment.png')
    )
    
    # Plot correlation matrix
    plot_correlation_matrix(
        data_with_features,
        save_path=output_paths['correlation_matrix']
    )
    
    # Plot feature importances
    if mda_results['feature_importance'] is not None:
        plot_feature_importance(
            mda_results['feature_importance'],
            save_path=output_paths['feature_importance']
        )
    
    # Plot MDA projection
    plot_mda_projection(
        mda_results,
        save_path=output_paths['mda_projection']
    )
    
    # Plot discriminant function over time
    plot_discriminant_time_series(
        discriminant_df,
        save_path=output_paths['discriminant_time']
    )
    
    logger.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
