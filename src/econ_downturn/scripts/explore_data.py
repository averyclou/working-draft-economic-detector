#!/usr/bin/env python3
"""
Explore Economic Data: productionize the 01_data_exploration notebook.

Citation: This script is built off the scaffolding in my Milestone 1
project, as we were instructed to use a pre-existing notebook.
Credit to that submitted project incldues joint work with my
teammates, Matthew Grohotolski and Sruthi Rayasam, as well as my
(Avery Cloutier) contributions. The elements for this assignment
are completely original and not re-used group work.

This script contains steps for loading the data, summarizing, and 
generating the plots. This fulfills every step of the required 
instructions. A correlation matrix and recession-flag correlations
are returned as PNGs in docs -> images.
"""

import os
import sys
import argparse
from pathlib import Path

# Path setup: allow imports from src/econ_downturn and notebooks
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "notebooks"))

#Importing custom utilizy packages
from notebook_utils import init_notebook, load_data, display_data_info, save_figure
from econ_downturn import (
    plot_indicator_with_recessions,
    plot_correlation_matrix,
    plot_recession_correlations)

def parse_args() -> argparse.Namespace:
    """
    Parsing module. This is the tricky part of the script documentation. I'll expand below.

    It Returns:
        Namespace: with attributes
        input_csv: path to merged_data.csv
        output_dir: directory where PNGs will be saved
        num_indicators: the num of indicators to plot
    """
    parser = argparse.ArgumentParser(
        description="Run data exploration plots on the merged dataset")
    parser.add_argument(
        "input_csv",
        help="merged_data.csv with date index and recession flag")
    parser.add_argument(
        "output_dir",
        help="directory to save the generated PNG files")
    parser.add_argument(
        "-n", "--num_indicators",
        type = int,
        default = 5,
        help = "how many indicators to plot (default: 5)")
    return parser.parse_args()

def main() -> None:
    """
    Main entry point: load data, summarize, and generate plots.
    """
    args = parse_args()

    # we're creating an output directory, in case it's needed
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_notebook()

    # loading our merged dataset just like in our standard notebooks
    df = load_data(use_cached=True)
    print("Data load is successful!")

    # summary (to give the reader a dataset architecture understanding)
    print("Here is the summary")
    display_data_info(df)

    # plotting the actual time series (indicators)
    indicators = [col for col in df.columns if col != "recession"]
    indicators = indicators[: args.num_indicators]
    for ind in indicators:
        print("Plotting the indicator:", ind)
        fig = plot_indicator_with_recessions(
            df,
            ind,
            title = ind + " over time")
        filename = ind.lower() + "_over_time.png"
        save_figure(fig, filename, str(out_dir))

    # plot overall correlation matrix
    print("Correlation matrix:")
    fig = plot_correlation_matrix(df)
    save_figure(fig, "correlation_matrix.png", str(out_dir))

    # plot correlations with recession flag if present (which it alwasy should be)
    if "recession" in df.columns:
        print("Recession correls:")
        fig = plot_recession_correlations(df)
        save_figure(fig, "recession_correlations.png", str(out_dir))

if __name__ == "__main__":
    main()
