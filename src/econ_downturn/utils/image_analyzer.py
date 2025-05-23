#!/usr/bin/env python3
"""
Utility functions for analyzing image files to identify potential issues.
"""

import os
import numpy as np
import matplotlib.image as mpimg

def analyze_image(img_path):
    """
    Analyze an image file for potential issues.
    
    Parameters
    ----------
    img_path : str
        Path to the image file
        
    Returns
    -------
    dict
        Dictionary with analysis results
    """
    try:
        # Load the image
        img = mpimg.imread(img_path)

        # Get image dimensions
        height, width, _ = img.shape

        # Check if image is mostly white (empty plot)
        white_percentage = np.mean(img > 0.9) * 100

        # Check if image has very low contrast (might indicate empty or problematic plot)
        contrast = np.std(img)

        # Check if image has text (by looking for non-white pixels in typical text regions)
        has_text = np.mean(img[int(height*0.1):int(height*0.2), int(width*0.1):int(width*0.9)] < 0.9) > 0.05

        # Check if image has a plot (by looking for non-white pixels in the center)
        has_plot = np.mean(img[int(height*0.3):int(height*0.7), int(width*0.2):int(width*0.8)] < 0.9) > 0.05

        # Check if image has axes (by looking for lines at the edges)
        left_edge = np.mean(img[int(height*0.3):int(height*0.7), int(width*0.1):int(width*0.15)] < 0.9) > 0.05
        bottom_edge = np.mean(img[int(height*0.8):int(height*0.9), int(width*0.2):int(width*0.8)] < 0.9) > 0.05
        has_axes = left_edge and bottom_edge

        # Determine if the image might be problematic
        is_problematic = (white_percentage > 95 or
                          contrast < 0.05 or
                          not has_text or
                          not has_plot or
                          not has_axes)

        return {
            'filename': os.path.basename(img_path),
            'dimensions': (width, height),
            'white_percentage': white_percentage,
            'contrast': contrast,
            'has_text': has_text,
            'has_plot': has_plot,
            'has_axes': has_axes,
            'is_problematic': is_problematic,
            'issue': 'Potential issues detected' if is_problematic else 'Looks good'
        }
    except Exception as e:
        return {
            'filename': os.path.basename(img_path),
            'error': str(e),
            'is_problematic': True,
            'issue': f'Error analyzing image: {str(e)}'
        }

def analyze_images_in_directory(images_dir):
    """
    Analyze all images in a directory.
    
    Parameters
    ----------
    images_dir : str
        Directory containing image files
        
    Returns
    -------
    list
        List of dictionaries with analysis results
    """
    results = []

    # Get all PNG files in the directory
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]

    # Analyze each image
    for img_path in image_files:
        result = analyze_image(img_path)
        results.append(result)

    return results

def get_problematic_images(results):
    """
    Get a list of problematic images from analysis results.
    
    Parameters
    ----------
    results : list
        List of dictionaries with analysis results
        
    Returns
    -------
    list
        List of dictionaries for problematic images
    """
    return [r for r in results if r.get('is_problematic', False)]
