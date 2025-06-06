# Installation Guide

This guide provides instructions for setting up the Economic Downturn Detector project environment.

## Prerequisites

- Python 3.8 or higher
- Git
- FRED API key (for data acquisition)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/economic-downturn-detector.git
cd economic-downturn-detector
```

### 2. Create a Virtual Environment

#### Using venv (recommended)

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

#### Using conda

```bash
# Create a conda environment
conda create -n econ-detector python=3.8
conda activate econ-detector
```

### 3. Install Dependencies

```bash
# Install required dependencies
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory with your FRED API key:

```
FRED_API_KEY=your_fred_api_key_here
```

#### Obtaining a FRED API Key

To obtain a FRED API key:
1. Create an account on [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/)
2. After logging in, go to your account settings by clicking on your username in the top right corner
3. Select "API Keys" from the menu
4. Click "Request API Key" if you don't already have one
5. Once approved, copy your API key and paste it into the `.env` file

The API key is required to access economic data from the FRED database, which is a primary data source for this project.

### 5. Collect the Data

#### Option 1: Using the Data Collection Notebook (Recommended)

If you have a FRED API key, we recommend using the comprehensive data collection notebook:

```bash
# Start Jupyter notebook
jupyter notebook

# Open and run notebooks/00_data_collection.ipynb
# This will fetch all data sources:
# - FRED economic indicators
# - NBER recession data
# - University of Michigan consumer sentiment
# - Conference Board consumer confidence
# - Business sentiment indicators
```

#### Option 2: Using Individual Scripts

You can also run individual data collection scripts if preferred:

```bash
# Fetch FRED data
python src/fetch_fred_data.py

# Fetch NBER recession data
python src/fetch_nber_data.py

# Fetch University of Michigan Consumer Sentiment data
python src/fetch_umich_data.py
```

#### Option 2: Using Sample Data (For Development/Testing)

If you don't have a FRED API key or want to quickly test the pipeline, you can generate synthetic data:

```bash
# Generate sample data
python src/generate_sample_data.py
```

This will create synthetic economic indicators that mimic the structure of real data, allowing you to test the analysis pipeline without requiring API access.

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Navigate to the `notebooks` directory to access the analysis notebooks.

## Project Structure

- `data/`: Raw and processed datasets
  - `fred/`: Federal Reserve Economic Data
  - `nber/`: National Bureau of Economic Research data
  - `umich/`: University of Michigan Consumer Sentiment data
  - `processed/`: Processed datasets for analysis
- `notebooks/`: Jupyter notebooks for analysis
  - `01_data_exploration.ipynb`: Data exploration and visualization
  - `02_feature_engineering.ipynb`: Feature engineering and preprocessing
  - `03_multiple_discriminant_analysis.ipynb`: MDA modeling and evaluation
- `src/`: Python modules and scripts
  - `econ_downturn/`: Main package with all functionality
  - `fetch_fred_data.py`: Script to fetch data from FRED API
  - `fetch_nber_data.py`: Script to fetch NBER recession data
  - `fetch_umich_data.py`: Script to fetch UMich Consumer Sentiment data
- `docs/`: Project documentation

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter errors related to the FRED API key:
- Verify that your API key is correct
- Check that the `.env` file is in the correct location
- Ensure that the `python-dotenv` package is installed

#### Data Fetching Issues

If you encounter issues fetching data:
- Check your internet connection
- Verify that the API endpoints are accessible
- Check the console output for specific error messages

#### Package Installation Issues

If you encounter issues installing packages:
- Update pip: `pip install --upgrade pip`
- Install packages individually to identify problematic dependencies
- Check for compatibility issues between packages

For additional help, please open an issue on the project repository.
