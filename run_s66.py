import argparse
import numpy as np 

from random_walk_composite import run_random_walk, best_estimate
from parse_distribution import load_and_interpolate
from extrapolate_data import preprocess_extrapolants, calculate_extrapolants, read_dynamic_data 
from helpers import print_time, plot_binned_histogram, print_table, print_memory_usage, estimate_peak_memory

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read data and calculate CBS extrapolations.")
    parser.add_argument("filename", type=str, help="The name or path of the file to read")
    parser.add_argument("--N_samples", type=int, default=100_000, help="Number of Monte Carlo patched tables")
    parser.add_argument("--bin_width", type=float, default=1e-4, help="Width of histogram bins in units of data")
    parser.add_argument("--levels", nargs="+", type=float, default=[0.75,0.95,0.99], help="At what confidence levels is uncertainty to be printed")
    parser.add_argument("--mem", type=int, default=None, help="Maximum allowed memory in MB to be used for each batch of processed data. If not set, everything is done in one batch. ")
    parser.add_argument("--N_cpu", type=int, default=10, help="Number of processes to be run when generating Monte Carlo patched tables")
    parser.add_argument("--center", action='store_true', default=False, help="Whether center pdf on single bin")
    parser.add_argument("--graph", action='store_true', default=False, help="Whether to save a graph of the final histogram")
    parser.add_argument("--csv", action='store_true', default=False, help="Whether to save a CSV file of the final histogram")
    args, unknown = parser.parse_known_args()

    pdf = load_and_interpolate("/net/home/plgrid/plgjlang/calcs/random_walk/merged3_lin.csv")

    assert len(unknown) >= 2
    from parse_s66 import fill_table
    _extrapolants, _original_data,  _refs = fill_table(args.filename)
    sys = unknown[unknown.index("--sys") + 1]
    print("sys",sys)
    extrapolants, original_data, refs = _extrapolants[sys], _original_data[sys], _refs[sys]
    headers_ext = [f"{'bas'}",f"{'SCF'}",f"{'MP2'}",f"{'CCSD(T)'}"]
    print_table(original_data, headers_ext)
    SCF = original_data[-1][1]
    print("")
    headers_ext = [f"{'ext'}",f"{'SCF'}",f"{'MP2'}",f"{'CCSD(T)'}"]
    print_table(extrapolants, headers_ext)

    
    keep_cols = [0, 2, 3]
    if unknown.index("--per-partes") >= 0:
        idx = int(unknown[unknown.index("--per-partes") + 1]) 
        keep_cols = [0, idx]
        print("Running single method S66 run for",headers_ext[idx])
        SCF = 0

    original_data = [[row[i] for i in keep_cols] for row in original_data]
    extrapolants = [[row[i] for i in keep_cols] for row in extrapolants]

    calc_sum, calc_list = best_estimate(original_data)

    idx = np.argmax(np.abs(calc_list))
    baseline = calc_list[idx]

    # baseline = 1
    stats_tracker = run_random_walk(pdf, extrapolants, args.N_samples, args.bin_width, args.levels, args.mem, args.N_cpu, baseline, args.center, constant = SCF)

    print("\n--- EXPORTING RESULTS ---")
    base_name = args.filename 

    if args.csv:
        csv_out_file = f"{base_name}_{sys}_histogram_data.csv"
        stats_tracker.export_to_csv(csv_out_file)

    if args.graph:
        plot_out_file = f"{base_name}_{sys}_histogram_plot.png"
        plot_binned_histogram(
            stats_tracker,
            filename=plot_out_file,
            title=f"Random Walk Limit Distribution",
            xlabel="Estimated quantity",
            ylabel="Frequency"
        )

    average = stats_tracker.get_average()
    baseline_sum,baseline_list = best_estimate(extrapolants)
    calc_sum, calc_list = best_estimate(original_data)
    unc = 0
    for b,c in zip(baseline_list, calc_list):
        unc += (b-c)**2
    unc = np.sqrt(unc) 
    print("DATA_OUT")
    headers_ext = [f"{'ext'}",f"{'SCF'}",f"{'MP2'}",f"{'CCSD(T)'}"]
    print_table(_extrapolants[sys], headers_ext)
    headers_ext = [f"{'bas'}",f"{'SCF'}",f"{'MP2'}",f"{'CCSD(T)'}"]
    print_table(_original_data[sys], headers_ext)
    print(f" Reference value {refs:.6f} extrapolated value {SCF+baseline_sum:.6f} with quadrature uncertainty {unc:.6f} {refs-SCF+baseline_sum:.6f}")
    for thr in args.levels:
      lower, upper = stats_tracker.get_confidence_interval(confidence_level=thr)
      el = lower-average 
      eu = upper-average 
      print(f"  {thr*100}% Confidence Interval: [{el:.6f}, {eu:.6f}] ... {SCF + baseline_sum + el:.6f} <= {refs:.6f} <= {SCF + baseline_sum + eu:.6f}",(SCF + baseline_sum + el<refs) or (SCF + baseline_sum + eu > refs))
                  

