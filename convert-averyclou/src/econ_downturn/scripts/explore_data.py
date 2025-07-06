#!/usr/bin/env python3
"""
Explore Economic Data: automate the 01_data_exploration notebook.
"""
import os
import sys
import argparse
from pathlib import Path

# 1) allow imports from src/econ_downturn and from notebooks/
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "notebooks"))

# 2) now notebook_utils and econ_downturn will resolve
from notebook_utils import init_notebook, load_data, display_data_info, save_figure
from econ_downturn import (
    plot_indicator_with_recessions,
    plot_correlation_matrix,
    plot_recession_correlations
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run data exploration plots on the merged dataset"
    )
    p.add_argument("input_csv", help="merged_data.csv with date index and recession flag")
    p.add_argument("output_dir", help="where to save the PNGs")
    p.add_argument("-n", "--num_indicators", type=int, default=5,
                   help="how many indicators to plot")
    return p.parse_args()

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)

    init_notebook()

    df = load_data(use_cached=True)
    print("Data load is successful!")

    print("Here is the summary")
    display_data_info(df)

    inds = [c for c in df.columns if c != "recession"][: args.num_indicators]
    for ind in inds:
        print("Plotting the indicator:", ind)
        fig = plot_indicator_with_recessions(df, ind, title=ind + " over time")
        save_figure(fig, ind.lower() + "_over_time.png", str(out))

    print("Correlation matrix:")
    fig = plot_correlation_matrix(df)
    save_figure(fig, "correlation_matrix.png", str(out))

    if "recession" in df.columns:
        print("Recession correls:")
        fig = plot_recession_correlations(df)
        save_figure(fig, "recession_correlations.png", str(out))

if __name__ == "__main__":
    main()
