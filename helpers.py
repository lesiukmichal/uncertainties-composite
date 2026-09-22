import time
import psutil
import numpy as np
import matplotlib.pyplot as plt

def print_time(_start_time,msg="",end='\n'):
    """ 
    Print timings
    """
    prefix = f"{msg}: " if msg else ""
    elapsed_seconds = time.perf_counter() - _start_time
    if elapsed_seconds > 60:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        print(f"{prefix}execution time: {minutes}m {seconds:.0f}s",end=end)
    elif elapsed_seconds > 10:
        print(f"{prefix}execution time: {elapsed_seconds:.1f} seconds",end=end)
    elif elapsed_seconds > 1:
        print(f"{prefix}execution time: {elapsed_seconds:.2f} seconds",end=end)
    else:
        print(f"{prefix}execution time: {elapsed_seconds:.3f} seconds",end=end)

def print_table(table, headers=None):
    """Helper function to print a clean, aligned table."""
    if headers:
        print("\t".join(f"{h:>12}" for h in headers))
        print("-" * (14 * len(headers)))

    for row in table:
        formatted_row = []
        for val in row:
            if isinstance(val, str):
                formatted_row.append(f"{val:>12}")
            elif np.isnan(val):
                formatted_row.append(f"{'NaN':>12}")
            else:
                formatted_row.append(f"{val:>12.6f}")
        print("\t".join(formatted_row))

def plot_binned_histogram(stats_tracker, filename, title="Data Distribution", xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram from the dynamically generated bins and saves it to a file.
    """
    if not stats_tracker.counts:
        return

    sorted_bins = sorted(stats_tracker.counts.keys())
    x_vals = [(b + 0.5) * stats_tracker.bin_width * stats_tracker.baseline for b in sorted_bins]
    y_vals = [stats_tracker.counts[b] for b in sorted_bins]

    plt.figure(figsize=(8, 5))
    plt.bar(x_vals, y_vals, width=stats_tracker.bin_width * 1.0 * stats_tracker.baseline,
            color='lightsteelblue', edgecolor='black', alpha=0.7)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # --- CHANGED: Save instead of show ---
    plt.savefig(filename, dpi=300)
    # plt.show()
    plt.close() # Free up the memory
    print(f"[*] Histogram plot saved to: {filename}")

def print_memory_usage(msg="",just_return=False):
    """
    Print (and return) currently used memory
    """
    proc_mb = psutil.Process().memory_info().rss / 1024**2
    vm = psutil.virtual_memory()
    prefix = f"{msg}: " if msg else ""
    outmsg =  f"{prefix} {proc_mb:8.0f} MB  (system RAM in use: {vm.percent:4.1f}%)"
    
    if not just_return:
        print(outmsg)
    return outmsg

def estimate_peak_memory(N_samples, N_bas, N_method, mids_size, bulk_size, bulk_batches, n_batch_arrays = 6):
    """
    Estimate the peak memory usage during the run and print memory layout information
    """

    sample_mb =  N_samples * N_bas * N_method * 8 / 1024**2
    batch_set_mb = bulk_size * mids_size * n_batch_arrays * 8 / 1024**2
    entry_mb = bulk_size * mids_size * 8 / 1024**2
    total_mb = sample_mb + batch_set_mb
    print("")
    print("-"*75)
    print("Estimated memory requirements:")
    print_memory_usage(" currently using")
    print(f" current_samples (persistent): {sample_mb:8.0f} MB ({N_samples:,} x {N_bas} x {N_method})")
    print(f" single entry size:            {entry_mb:8.0f} MB ({bulk_size:,} x {mids_size:,})")           
    print(f" per-batch working set:        {batch_set_mb:8.0f} MB  ({n_batch_arrays} arrays x {bulk_size:,} x {mids_size:,})")
    print(f" total estimated peak:         {total_mb:8.0f} MB")
    print(f" number of required batches:   {bulk_batches:8d}")
    print("-"*75)
    print("")

