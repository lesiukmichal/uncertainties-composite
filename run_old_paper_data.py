import argparse
import numpy as np 

from random_walk_composite import run_random_walk
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

    pdf = load_and_interpolate("/net/home/plgrid/plgjlang/calcs/random_walk/merged4.csv")
    refs = None

    assert args.N_samples == 1
    refs = None
    extrapolants = []
    original_data = []
    with open( args.filename, 'r') as fin:
        for line in fin:
            line = line.split()
            if "inf" == line[0]:
                refs = float(line[1])
            else:
                original_data.append( [int(line[0]),float(line[1])] )

    headers_ext = [f"{'bas'}",f"{'col0'}"]
    print_table(original_data, headers_ext)

    for i in range(1,len(original_data)):
        n,e = original_data[i-1]
        Nn,Ne = original_data[i]
        extrapolants.append([f"{n}-> {Nn}",(Ne*Nn**3 - e*n**3) / (Nn**3 - n**3)])

    headers_ext = [f"{'ext'}",f"{'col0'}"]
    print_table(extrapolants, headers_ext)

    baseline = extrapolants[-1][1] - extrapolants[-2][1]

    stats_tracker = run_random_walk(pdf, extrapolants, args.N_samples, args.bin_width, args.levels, args.mem, args.N_cpu, baseline, args.center)

    print("\n--- EXPORTING RESULTS ---")
    base_name = args.filename #"_".join(f"{k}={v}" for k, v in vars(args).items())

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

    if refs is not None:
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
        print(f" Reference value {refs:.6f} extrapolated value {baseline_sum:.6f} with quadrature uncertainty {unc:.6f} {refs-baseline_sum:.6f}")
        for thr in args.levels:
          lower, upper = stats_tracker.get_confidence_interval(confidence_level=thr)
          el = lower-average 
          eu = upper-average 
          print(f"  {thr*100}% Confidence Interval: [{el:.6f}, {eu:.6f}] ... {baseline_sum + el:.6f} <= {refs:.6f} <= {baseline_sum + eu:.6f}")
                      
    
