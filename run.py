import argparse
import numpy as np 

from helpers import plot_binned_histogram 
from parse_distribution import load_and_interpolate
from extrapolate_data import preprocess_extrapolants
from random_walk_composite import run_random_walk, best_estimate

# parse user defined arguments
parser = argparse.ArgumentParser(description="Read data and calculate CBS extrapolations.")
parser.add_argument("filename", type=str, help="The name or path of the file to read")
parser.add_argument("--N_samples", type=int, default=100_000, help="Number of Monte Carlo patched tables")
parser.add_argument("--bin_width", type=float, default=1e-4, help="Width of histogram bins in units of data")
parser.add_argument("--levels", nargs="+", type=float, default=[0.75,0.95,0.99], help="At what confidence levels is uncertainty to be printed")
parser.add_argument("--mem", type=int, default=None, help="Maximum allowed memory in MB to be used for each batch of processed data. If not set, everything is done in one batch. ")
parser.add_argument("--N_cpu", type=int, default=1, help="Number of processes to be run when generating Monte Carlo patched tables")
parser.add_argument("--center", action='store_true', default=False, help="Whether center pdf on single bin")
parser.add_argument("--graph", action='store_true', default=False, help="Whether to save a graph of the final histogram")
parser.add_argument("--csv", action='store_true', default=False, help="Whether to save a CSV file of the final histogram")
args, unknown = parser.parse_known_args()


pdf = load_and_interpolate("merged_data.csv")

extrapolants,original_data =  preprocess_extrapolants(args.filename)
extr_val, extr_vals = best_estimate(extrapolants)
print(f" Extrapolated value {extr_val:.6f}")

baseline = np.sum(extr_vals[0]) - np.sum(extr_vals[1])

stats_tracker = run_random_walk(pdf, extrapolants, args.N_samples, args.bin_width, args.levels, args.mem, args.N_cpu, baseline, args.center)

if args.csv or args.graph:
    print("\n--- EXPORTING RESULTS ---")
    base_name = args.filename 

if args.csv:
    csv_out_file = f"{base_name}_histogram_data.csv"
    stats_tracker.export_to_csv(csv_out_file)

if args.graph:
    plot_out_file = f"{base_name}_histogram_plot.png"
    plot_binned_histogram(
        stats_tracker,
        filename=plot_out_file,
        title=f"Random Walk Limit Distribution",
        xlabel="Estimated quantity",
        ylabel="Frequency"
    )

print("\n--- FINISHED ---")
