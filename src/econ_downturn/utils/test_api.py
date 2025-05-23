#!/usr/bin/env python3
"""
Utility functions for testing API connections.
"""

import os
import requests
from dotenv import load_dotenv

def test_fred_api_connection(api_key=None):
    """
    Test the FRED API connection.
    
    Parameters
    ----------
    api_key : str, optional
        FRED API key. If None, will try to load from environment variable.
        
    Returns
    -------
    dict
        Dictionary with test results
    """
    # Load environment variables if api_key is not provided
    if api_key is None:
        load_dotenv()
        api_key = os.getenv('FRED_API_KEY')
    
    if not api_key:
        return {
            'success': False,
            'error': 'FRED API key not found. Please set the FRED_API_KEY environment variable.',
            'message': 'You can get a FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html'
        }
    
    # Make a simple API request
    url = f"https://api.stlouisfed.org/fred/series?series_id=UNRATE&api_key={api_key}&file_type=json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        return {
            'success': True,
            'series_title': data['seriess'][0]['title'],
            'units': data['seriess'][0]['units'],
            'frequency': data['seriess'][0]['frequency'],
            'observation_start': data['seriess'][0]['observation_start'],
            'observation_end': data['seriess'][0]['observation_end'],
            'message': 'API connection successful!'
        }
        
    except requests.exceptions.HTTPError as e:
        error_message = f"API connection failed with HTTP error: {e}"
        if response.status_code == 400:
            error_message += "\nThis might be due to an invalid API key."
        
        return {
            'success': False,
            'error': error_message,
            'response_content': response.text
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"API connection failed with error: {e}"
        }
