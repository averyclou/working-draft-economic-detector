#!/usr/bin/env python3
"""
Script to demonstrate how to use the combined dataset (FRED, NBER, UMich)
and create visualizations that show relationships between economic indicators
and consumer sentiment.
"""

import os
import sys
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Add the parent directory to the path to import the econ_downturn package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the econ_downturn package
from econ_downturn import (
    get_all_data, load_umich_data, load_fred_data, load_nber_data,
    engineer_features, normalize_data, apply_pca,
    apply_mda, create_discriminant_time_series,
    plot_indicator_with_recessions, plot_correlation_matrix,
    plot_recession_correlations, plot_feature_importance,
    plot_mda_projection, plot_discriminant_time_series,
    plot_sentiment_vs_indicator, plot_sentiment_correlation_matrix,
    setup_logger, load_environment, get_output_paths
)

# Set up logging
logger = setup_logger('visualize_combined_data')

def main():
    """Main function to run the visualization script."""
    # Load environment variables
    config = load_environment()

    # Get output paths
    output_paths = get_output_paths()

    # Create output directories if they don't exist
    os.makedirs(output_paths['images_dir'], exist_ok=True)

    # Step 1: Load and merge all data
    logger.info("Step 1: Loading and merging all data...")

    # Load data from different sources
    fred_data = load_fred_data()
    nber_data = load_nber_data()
    umich_data = load_umich_data()

    # Check if data was loaded successfully
    if fred_data.empty:
        logger.error("Failed to load FRED data.")
        return

    if nber_data.empty:
        logger.error("Failed to load NBER recession data.")
        return

    if umich_data.empty:
        logger.error("Failed to load UMich consumer sentiment data.")
        return

    # Merge all data
    merged_data = get_all_data()

    if merged_data is None or merged_data.empty:
        logger.error("Failed to merge data.")
        return

    logger.info(f"Merged data shape: {merged_data.shape}")
    logger.info(f"Merged data columns: {merged_data.columns.tolist()}")

    # Step 2: Create visualizations with the combined dataset
    logger.info("Step 2: Creating visualizations with the combined dataset...")

    # Plot consumer sentiment over time with recession periods
    sentiment_plot = plot_indicator_with_recessions(
        merged_data,
        'SENTIMENT',
        title='Consumer Sentiment with Recession Periods',
        save_path=os.path.join(output_paths['images_dir'], 'consumer_sentiment.png')
    )

    # Plot consumer sentiment vs unemployment rate
    sentiment_vs_unemployment = plot_sentiment_vs_indicator(
        merged_data,
        sentiment_col='SENTIMENT',
        indicator_col='UNEMPLOYMENT',
        save_path=os.path.join(output_paths['images_dir'], 'sentiment_vs_unemployment.png')
    )

    # Plot consumer sentiment vs GDP growth
    sentiment_vs_gdp = plot_sentiment_vs_indicator(
        merged_data,
        sentiment_col='SENTIMENT',
        indicator_col='GDP',
        save_path=os.path.join(output_paths['images_dir'], 'sentiment_vs_gdp.png')
    )

    # Plot correlations between consumer sentiment and economic indicators
    sentiment_correlations = plot_sentiment_correlation_matrix(
        merged_data,
        sentiment_cols=['SENTIMENT'],
        top_n=10,
        save_path=os.path.join(output_paths['images_dir'], 'sentiment_correlations.png')
    )

    # Step 3: Feature engineering with the combined dataset
    logger.info("Step 3: Performing feature engineering with the combined dataset...")

    # Engineer features
    data_with_features = engineer_features(merged_data)

    logger.info(f"Data with features shape: {data_with_features.shape}")

    # Step 4: Run preliminary MDA test
    logger.info("Step 4: Running preliminary MDA test...")

    # Normalize data
    data_normalized, scaler = normalize_data(data_with_features)

    # Separate features and target
    X = data_normalized.drop(columns=['recession'])
    y = data_normalized['recession']

    # Apply MDA
    mda_results = apply_mda(X, y)

    # Create discriminant time series
    discriminant_df = create_discriminant_time_series(
        mda_results['model'], X, y
    )

    # Plot discriminant function over time
    discriminant_plot = plot_discriminant_time_series(
        discriminant_df,
        save_path=os.path.join(output_paths['images_dir'], 'discriminant_time_series.png')
    )

    # Plot feature importances
    if mda_results['feature_importance'] is not None:
        feature_importance_plot = plot_feature_importance(
            mda_results['feature_importance'],
            save_path=os.path.join(output_paths['images_dir'], 'feature_importance.png')
        )

    logger.info("Visualization and analysis completed successfully.")
    logger.info(f"All plots saved to {output_paths['images_dir']}")

if __name__ == "__main__":
    main()
