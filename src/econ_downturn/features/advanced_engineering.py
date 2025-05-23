"""
Advanced feature engineering functions.

This module contains advanced feature engineering functions for the economic downturn detector,
including custom lag creation, interaction terms, and sentiment transformations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

def engineer_features_with_custom_lags(data, sentiment_lags=[1, 3, 6, 12, 18, 24], other_lags=[1, 3, 6, 12]):
    """
    Engineer features with custom lag periods for sentiment and other indicators.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
    sentiment_lags : list
        List of lag periods for sentiment indicators
    other_lags : list
        List of lag periods for other economic indicators
        
    Returns
    -------
    pandas.DataFrame
        Dataset with engineered features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Resample to monthly frequency if needed
    if df.index.freq != 'M':
        df = df.resample('M').last()
        print(f"Resampled data to M frequency, new shape: {df.shape}")
    
    # Identify sentiment columns
    sentiment_cols = [col for col in df.columns if 'SENTIMENT' in col]
    
    # Create lag variables for sentiment columns with custom lags
    for col in sentiment_cols:
        for lag in sentiment_lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Create lag variables for other columns
    other_cols = [col for col in df.columns if col not in sentiment_cols and col != 'recession']
    for col in other_cols:
        for lag in other_lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    print(f"Created lag variables, new shape: {df.shape}")
    
    # Calculate rate of change for all columns except recession
    for col in df.columns:
        if col != 'recession':
            # 1-month rate of change
            df[f"{col}_roc1"] = df[col].pct_change(1)
            
            # 3-month rate of change
            df[f"{col}_roc3"] = df[col].pct_change(3)
            
            # 12-month rate of change
            df[f"{col}_roc12"] = df[col].pct_change(12)
    
    print(f"Calculated rate of change, new shape: {df.shape}")
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"Dropped rows with missing values, new shape: {df.shape}")
    
    return df

def create_interaction_terms(data):
    """
    Create interaction terms between consumer sentiment and economic indicators.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
        
    Returns
    -------
    pandas.DataFrame
        Dataset with interaction terms
    """
    # Make a copy of the data
    df = data.copy()
    
    # Identify sentiment columns (base columns, not lagged or transformed)
    sentiment_cols = [col for col in df.columns if col == 'SENTIMENT']
    
    # Identify key economic indicators (base columns, not lagged or transformed)
    key_indicators = ['GDP', 'UNEMPLOYMENT', 'FED_FUNDS', 'YIELD_CURVE']
    
    # Create interaction terms
    for sentiment_col in sentiment_cols:
        for indicator in key_indicators:
            if sentiment_col in df.columns and indicator in df.columns:
                # Create interaction term
                df[f"{sentiment_col}_x_{indicator}"] = df[sentiment_col] * df[indicator]
                
                # Create ratio term
                df[f"{sentiment_col}_div_{indicator}"] = df[sentiment_col] / df[indicator].replace(0, np.nan)
                
                # Create difference term
                df[f"{sentiment_col}_minus_{indicator}"] = df[sentiment_col] - df[indicator]
    
    # Fill NaN values that might have been created
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Created interaction terms, new shape: {df.shape}")
    
    return df

def apply_sentiment_transformations(data):
    """
    Apply different transformations to consumer sentiment data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset
        
    Returns
    -------
    pandas.DataFrame
        Dataset with transformed sentiment features
    """
    # Make a copy of the data
    df = data.copy()
    
    # Identify sentiment columns (base columns, not lagged or transformed)
    sentiment_cols = [col for col in df.columns if col == 'SENTIMENT']
    
    for col in sentiment_cols:
        if col in df.columns:
            # Log transformation
            df[f"{col}_log"] = np.log(df[col].replace(0, np.nan))
            
            # Square root transformation
            df[f"{col}_sqrt"] = np.sqrt(df[col])
            
            # Square transformation
            df[f"{col}_squared"] = df[col] ** 2
            
            # Z-score normalization
            df[f"{col}_zscore"] = (df[col] - df[col].mean()) / df[col].std()
            
            # Min-max scaling
            df[f"{col}_minmax"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            # Moving average (3-month)
            df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
            
            # Moving average (6-month)
            df[f"{col}_ma6"] = df[col].rolling(window=6).mean()
            
            # Moving average (12-month)
            df[f"{col}_ma12"] = df[col].rolling(window=12).mean()
            
            # Exponential moving average (3-month)
            df[f"{col}_ema3"] = df[col].ewm(span=3).mean()
            
            # Exponential moving average (6-month)
            df[f"{col}_ema6"] = df[col].ewm(span=6).mean()
            
            # Exponential moving average (12-month)
            df[f"{col}_ema12"] = df[col].ewm(span=12).mean()
    
    # Fill NaN values that might have been created
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Applied transformations to sentiment data, new shape: {df.shape}")
    
    return df

def select_features(X, y, method='anova', k=20):
    """
    Select the most predictive features using different methods.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    method : str
        Feature selection method ('anova', 'rfe')
    k : int
        Number of features to select
        
    Returns
    -------
    pandas.DataFrame
        Selected features
    list
        Names of selected features
    """
    if method == 'anova':
        # ANOVA F-value feature selection
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create a DataFrame with scores
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })
        feature_scores = feature_scores.sort_values('Score', ascending=False)
        
        print(f"Selected {k} features using ANOVA F-value")
        print("Top 10 features:")
        print(feature_scores.head(10))
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFE(estimator, n_features_to_select=k, step=1)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.support_].tolist()
        
        # Create a DataFrame with rankings
        feature_rankings = pd.DataFrame({
            'Feature': X.columns,
            'Ranking': selector.ranking_
        })
        feature_rankings = feature_rankings.sort_values('Ranking')
        
        print(f"Selected {k} features using Recursive Feature Elimination")
        print("Top 10 features:")
        print(feature_rankings.head(10))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame with selected features
    X_selected_df = X[selected_features]
    
    return X_selected_df, selected_features
