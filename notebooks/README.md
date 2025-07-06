# Economic Downturn Detector – Data Exploration Script

## Background

As given in the project instructions, I have productionized a notebook I have already created using a .py script. The notebook is Notebook 01, which is from my Milestone project. The full citation is in my script (src/econ_downturn/scripts/explore_data.py).

## Purpose

This script automates data exploration steps from Notebook 01. As instructed, it loads the clean data, creates output graphs, and saves them in the inputted (non hard-coded) output folder. It has been tested and works great!

## Location

Here are my key locations for you to grade:
- Script: src/econ_downturn/scripts/explore_data.py (this is the main script for the assignment instructions)
- Utilities: notebooks/notebook_utils.py (pre-existing, supported project notebooks)
- Input data: data/processed/merged_data.csv
- Output folder: docs/images/explore

When running the script, you need to provide:

- The input CSV file path (that merged dataset)
- The output folder (where the pics will be saved)
- Optionally, but the number of indicators to plot

An example of how to run is under the "How to Run" section, see that.

## What It Does

- Loads economic data from CSV (as in the Milestone project)
- Prints dataset summary
- Plots for the indicators
- Generates a correlation matrix (see Notebook 01)
- Plots correlation with recession flags (also see Notebook 01)

## How to Run

- From the project root folder:

- Example:
  - python src/econ_downturn/scripts/explore_data.py data/processed/merged_data.csv docs/images/explore -n 5

## Setup Steps

- Clone repo: git clone https://github.com/mads643v2/convert-averyclou.git
- Create virtual env: python -m venv .venv
- Activate: .venv\Scripts\activate
- Install: pip install -r requirements.txt
- Add .env with: FRED_API_KEY=your_fred_api_key (you will have to source this from FRED)

## Output

- Indicator plots (PNG)
- correlation_matrix.png
- recession_correlations.png

## Confirmed Working

- Script runs cleanly in VS Code terminal
- Modular with no hardcoded paths (function modularization taught in Milestone 1 was a huge help for me to implement this)
- Functions are very clear
- Passes linter and meets every requirement!
