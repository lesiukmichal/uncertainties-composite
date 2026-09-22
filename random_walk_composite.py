import math
import time
import psutil

import pandas as pd
import numpy as np

from functools import partial
from multiprocessing import Pool, cpu_count

from helpers import print_time, plot_binned_histogram, print_table, print_memory_usage, estimate_peak_memory

def generate_sample(extrapolants_table,n_samples):
    """
    Scans the table and replaces NaN values with a random walk step based on
    the two preceding values in that column.
    Does not modify the original table.

	Example of extrapolants_table
        2->3	    0.000000	    0.000000	    0.041282	    0.038259
        3->4	    0.000000	    0.624616	    0.043507	    0.037786
        4->5	    0.000000	    0.373608	         NaN	    0.038824
        5->6	    9.939109	    0.371940	         NaN	         NaN
        6->7	    4.232656	         NaN	         NaN	         NaN
        7->8	    4.232154	         NaN	         NaN	         NaN
    
    Returns a full random value table for all composite methods at 7->8 basis, 
	where NaNs where substituted with random walk estimates of extrapolated values.
        2->3	    0.000000	    0.000000	    0.041282	    0.038259
        3->4	    0.000000	    0.624616	    0.043507	    0.037786
        4->5	    0.000000	    0.373608	    0.044061	    0.038824
        5->6	    9.939109	    0.371940	    0.044484        0.039098
        6->7	    4.232656	    0.373101        0.044431        0.038957	
        7->8	    4.232154	    0.372179        0.044472        0.038874

    """

    sample = [row[1:] for row in extrapolants_table]

    num_rows = len(sample)
    if num_rows == 0:
        return sample

    num_cols = len(sample[0])
    # create a new rng for each process
    rng = np.random.default_rng()
    rng_samples = rng.uniform(0,1,min(n_samples*num_cols*num_rows,1024**2))


    _i = 0
    samples = []
    for j in range(n_samples):
        sample = [row[1:] for row in extrapolants_table]
        # move along each column, skipping the first (index 0, which holds the labels)
        for col in range(num_cols):
            for row in range(num_rows):

                # check if the current value is missing (NaN)
                if np.isnan(sample[row][col]):

                    # we need at least two values immediately above Nan
                    if row >= 2:
                        e_x = sample[row-1][col]
                        e_x_prev = sample[row-2][col]

                        # ensure the two values above are valid numbers
                        if not np.isnan(e_x) and not np.isnan(e_x_prev):
                            # calculate the bounding interval spread
                            spread = abs(e_x - e_x_prev)

                            # draw a uniform random variable in the interval
                            # random.uniform(a, b) includes all real floats between a and b
                            new_val = (e_x - spread) + 2*spread*rng_samples[_i]

                            # replace the NaN with the generated step
                            sample[row][col] = new_val
                            _i += 1
                            if _i >= rng_samples.size:
                                # print('Loading new samples',_i,rng_samples.size )
                                rng_samples = rng.uniform(0,1,min(n_samples*num_rows*num_cols,1024**2))
                                _i = 0
        samples.append(sample)
    return samples


def best_estimate(extrapolants_table):
    """
    Finds the extrapolant with the largest X (the last non-NaN value)
    in each energy column, and returns their sum.

	Returns the best estimate and its contributions.
    """
    if not extrapolants_table:
        raise RuntimeError("Missing extrapolants table")

    num_cols = len(extrapolants_table[0])
    best_values = []
    second_best = [] 
    # iterate through each energy column (skipping column 0)
    for col in range(1, num_cols):

        # read the column backwards: from the last row up to the first row
        for row in range(len(extrapolants_table) - 1, -1, -1):
            val = extrapolants_table[row][col]

            if not np.isnan(val):
                best_values.append(val)
                second_best.append(extrapolants_table[row-1][col])
                break  # the largest valid X found, so break out of the row loop

    return math.fsum(best_values), [best_values,second_best]

def sum_rows(sample_table):
    """
    Calculates the sum of each row in the sample table,
    completely neglecting the first column (index 0).

    Returns a list of sums corresponding to Row 0, Row 1, Row 2, etc.
    """
    row_sums = np.sum(sample_table[:,1:].astype(float),axis=1)
    return row_sums

def build_positive_bins(pdf, bin_width, x0, xl):
    """ 
    Builds bins for positive half of pdf starting with x0 and bin width bin_width.
    For values below analytic_y0 uses analytical solution around zero while for larger values
    data from numerical integration of Eq. (5) are used.

    Returns a list of bin mids, integral of pdf between two ends of bins, mask of which bins were obtained from analytical solution, left and right ends of bins
    """
    N_bins = int(np.ceil((xl-x0) / bin_width))
    left_edges = x0 + np.arange(N_bins) * bin_width
    right_edges = np.minimum(left_edges + bin_width, xl)
    mids = (left_edges + right_edges) / 2.0

    integrals = np.empty_like(mids)
    analytical_mask = (left_edges <= pdf.analytical_y0)

    left_anal, right_anal = left_edges[analytical_mask], right_edges[analytical_mask]
    integrals[analytical_mask] = pdf.analytical_integration(left_anal,right_anal) 
    if (analytical_mask.sum() < 5):
        print("Number of analytical bins",analytical_mask.sum(),list(zip(left_anal,right_anal)))
    else:
        print("Number of analytical bins",analytical_mask.sum())
   
    numerical_mask = ~analytical_mask 
    left_num, right_num = left_edges[numerical_mask], right_edges[numerical_mask]
    integrals[numerical_mask] = pdf.cdf(right_num) - pdf.cdf(left_num)
    print(f"Number of numerical bins {numerical_mask.sum():,}")
   
    return mids, integrals, analytical_mask

def generate_bins(pdf, bin_width, centered=False):
    """
    Bins the probability distribution to bins with bin width equal to bin_width. 
    If centered is True will center distribution on single middle bin, 
    otherwise will create positive sided bins starting at zero and mirroring it on other half.

    Returns a final list of bin mids, list of integrals of pdf between two ends of bins,
    """
    xl = pdf._x[-1] 
    _start_time = time.perf_counter()

    print("")
    if not centered:
        x0 = 0.0
        N_bins = int(np.ceil((xl-x0) / bin_width))
        print(f"Compressing PDF to histogram on range [{x0},{xl}] with {bin_width:g} width -> {N_bins:,} bins")
    else:
        half = bin_width / 2.0
        x0 = half
        N_bins = int(np.ceil((xl-x0) / bin_width))
        print(f"Compressing PDF to CENTERED histogram on range [{-xl},{xl}] with {bin_width:g} width -> {2*N_bins+1:,} bins")
 
    pos_mids, pos_integrals, analytical_mask = build_positive_bins(pdf, bin_width, x0, xl)
    numerical_mask = ~analytical_mask 
 
    if centered:
        center_bin = 2.0 * pdf.analytical_integration(0.0, half)
 
        mids = np.concatenate(([0.0], pos_mids))
        integrals = np.concatenate(([center_bin / 2.0], pos_integrals))
        numerical_mask = np.concatenate(([False],numerical_mask))
        analytical_mask = np.concatenate(([True],analytical_mask))
        print("\nBinning summary (centered)")
        print(f" Final size of binned grid is {integrals.size*2-1:,d} including both halves (one shared center bin)")
        print(f" Center bin: frequency={center_bin:.5e}")
        print(f" Number of numerical grids is {mids[numerical_mask].size*2:,d}")
        print(f" Mids of numerical grid go from {-mids[numerical_mask][::-1][0]:.3e} to {-mids[numerical_mask][::-1][-1]:.3e} and from {mids[numerical_mask][0]:.3e} to {mids[numerical_mask][-1]:.3e}")
        print(f" Number of analytical grids is {mids[~numerical_mask].size*2-1:,d}")
        print(f" Mids of analytical grid go from {-mids[analytical_mask][::-1][0]:.3e} to {-mids[analytical_mask][::-1][-1]:.3e} and from {mids[analytical_mask][0]:.3e} to {mids[analytical_mask][-1]:.3e}")

    else:
        mids, integrals = pos_mids, pos_integrals
        print("\nBinning summary")
        print(f" Final size of binned grid is {integrals.size*2:,d} including both halves")
        print(f" Number of numerical grids is {mids[numerical_mask].size*2:,d}")
        print(f" Mids of numerical grid go from {-mids[numerical_mask][::-1][0]:.3e} to {-mids[numerical_mask][::-1][-1]:.3e} and from {mids[numerical_mask][0]:.3e} to {mids[numerical_mask][-1]:.3e}")
        print(f" Number of analytical grids is {mids[~numerical_mask].size*2:,d}")
        print(f" Mids of analytical grid go from {-mids[~numerical_mask][::-1][0]:.3e} to {-mids[~numerical_mask][::-1][-1]:.3e} and from {mids[~numerical_mask][0]:.3e} to {mids[~numerical_mask][-1]:.3e}")
    print_memory_usage("Currently used")
    print_time(_start_time,'Binning')
 
    return mids, integrals

def estimate_bulk_size(max_mem, N_samples, N_bas, N_method, mids_size, n_batch_array = 6):
    """
    Estimates the size of bulk batch based on the provided maximum memory. 
    If no memory restriction is provided, it will create one single batch

    Return number of batches and size of single batch
    """
    if max_mem is None:
        return 1,N_samples

    max_mem_size = max_mem * 1024**2
    sample_size = N_samples * N_bas * N_method * 8
    free_memory = max_mem_size - psutil.Process().memory_info().rss 
    per_sample_batch = mids_size * 8 * n_batch_array 
    bulk_size = int(free_memory // per_sample_batch ) 
    bulk_batches = int(np.ceil(N_samples/bulk_size))

    return bulk_batches, bulk_size


class StreamingStats:
    """
    Class collecting distribution across all samples in a form of global histogram
    Calculates the average across all samples, together with uncertainties
    """
    def __init__(self, bin_width=1e-6, baseline=1):
        """
        Initialization method

        Parameters
        ----------
        bin_width : (int) Bin width of the binned pdf. Default value 1e-6.
        baseline  : (float) Baseline with which to renormalize data so bin width is rougly equal to relative error in final uncertainties.
                            Default value is 1 (no renormalization)
        """
                            
        self.bin_width = bin_width
        self.baseline = baseline

        self.counts = {}    
        self.bin_sums = {}  
        self.total_count = 0


    def get_average(self, constant=0):
        """Calculates the exact average using fsum across the bins."""
        if self.total_count == 0:
            return 0.0
        # math.fsum prevents floating-point drift over millions of additions
        return math.fsum(self.bin_sums.values()) / self.total_count * self.baseline + constant

    def get_confidence_interval(self, confidence_level=0.95):
        """Extracts the empirical bounds from the cumulative distribution of the bins."""
        if self.total_count == 0:
            return None, None

        # calculate the exact counts where we need to chop the tails i.e. 25% and 75%
        tail_fraction = (1.0 - confidence_level) / 2.0
        lower_target = (self.total_count * tail_fraction)
        upper_target = (self.total_count * (1.0 - tail_fraction)) 

        sorted_bins = sorted(self.counts.keys())

        lower_bound = None
        upper_bound = None
        cumulative = 0

        # find at which bins the cummulative count crosses the targets
        for i,b in enumerate(sorted_bins):
            cumulative += self.counts[b]

            if lower_bound is None and cumulative > lower_target:
                # estimate the value as the center of the bin
                lower_bound = (b + 0.5) * self.bin_width

            if upper_bound is None and cumulative > upper_target:
                upper_bound = (b + 0.5) * self.bin_width
                break

        return lower_bound * self.baseline, upper_bound * self.baseline

    def add_bulk(self, value, count):
        """
        Processes a bulk of samples into a static coarse global histogram with user defined bin width.
        Assumes a global histogram i.e. value of 10 belongs to bin with index floor(10/bin_width). 
        Only unique bin indices are kept to save memory due to the global bin
        """
        bin_idx = (np.floor(value / self.bin_width)).astype(int) # get indices of "global" histogram bin to which each points of value belongs

        self.total_count += np.sum(count) 
        inverse, unique_idx  = pd.factorize(bin_idx) # get only unique indices to save memory. Pandas factorize is much faster than numpy's unique

        counts_sum = np.bincount(inverse, weights=count)  # calculate how many sample values belong to the same bin of the global histogram.
        sums_sum   = np.bincount(inverse, weights=value * count) # calculate the final pdf bin integral for multiple finer bins belonging to the same global histogram bin.

        # add this bulks data to the final histogram from previous calls.
        for idx, c, s in zip(unique_idx.tolist(), counts_sum, sums_sum):
          self.counts[idx]   = self.counts.get(idx, 0) + c
          self.bin_sums[idx] = self.bin_sums.get(idx, 0.0) + s

    def export_to_csv(self, filename):
        """Exports the binned data to a CSV file for later reconstruction."""
        if not self.counts:
            print("Warning: No data to export.")
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # write headers
            writer.writerow(['Bin_Center', 'Frequency'])

            # sort the bins so the data is ordered logically from low to high energy
            sorted_bins = sorted(self.counts.keys())
            for b in sorted_bins:
                if self.left_edges is not None:
                    center = ( self.left_edges[i] + self.right_edges[i] ) / 2 
                else:
                    center = (b + 0.5) * self.bin_width * self.baseline
                count = self.counts[b]
                writer.writerow([f"{center:.8f}", count])

        print(f"[*] Histogram data saved to: {filename}")

# main function to run uncertainty estimation in composite schemes. 
def run_random_walk(pdf, extrapolants, N_samples=100_000, bin_width=1e-4, confidence_levels = [0.75,0.95,0.99], batch_mem = None, N_cpu = 1, baseline = 1, centered = False, constant = 0, **kwargs):
    """
    Method to run the composite random walk

    Parameters
    ----------
    pdf          : (interpolatedPDF) Custom class containing the information about the probability distribution. See module parse_distribution for more info
    extrapolants : (list of lists) List of lists of extrapolated values for which uncertainty is to be estimated. 
    N_samples    : (int) Number of random samples for which we run the uncertainty estimation.  Has to be larger than 0.
                         Default is 100 000. For production run at least 1 000 000 of samples should be used.
    bin_width    : (float) Bin width of the final histogram across all samples. Roughly translates to relative error in uncertainty estimation
                           Default values is 0.0001.
    confidence_levesl: (list) List of floats at which level final uncertainty is printed.
    batch_mem        : (int) Maximum allowed memory in MB which can be used during estimation run. Default is unrestricted run.
    N_cpu            : (int) Number of processes used to generate sample table. Default is 10.
    baseline         : (float) Number used to renormalize the data to keep the bin_width roughly equal to relative error in final uncertainty.
    centered         : (bool) Whether the pdf histogram is centered around single bin with middle at zero or histogram is just mirrored around zero with first bin starting at zero 
                              and middle of the bin being at bin_width/2. Default is False (mirrored histogram)
    constant         : (float) Constant float to move the distribution for the final printing. Usefull if for example one has SCF energy already at CBS levels and needs to estimate uncertainty 
                               post HF contributions.
    """
 
    class settings():
        """
        Class for user defined settings required to run calculation.
        See definition of particular variables in parent function.
        """
        def __init__(self, extrapolants, N_samples, bin_width, confidence_levels, batch_mem, N_cpu, baseline, centered = False, constant = 0):
            
            self.N_samples = N_samples
            self.bin_width = bin_width
            self.confidence_levels = confidence_levels            
            self.batch_mem = batch_mem
            self.N_cpu = N_cpu
            self.baseline = baseline
            self.centered = centered
            self.constant = constant           
            assert self.N_samples >= self.N_cpu 

        def __str__(self):
            """
            Return string of user settings.
            """
            string  = "\n"
            string += "+"*80
            string += "\nUser settings\n"
            string += f"Number of samples:   {self.N_samples:,d}\n"
            string += f"Histogram bin width: {self.bin_width}\n"
            string += f"Requested confidence levels of extrapolation uncertainty: {self.confidence_levels}\n"
            if self.batch_mem is not None:
                string += print_memory_usage("Currently used",True)+"\n"
                string += f"Will do batches to keep memory usage under {int(self.batch_mem):d} MB\n"
            string += f"Renormalization with {self.baseline}\n"
            string += f"Constant value added to the average is {self.constant}\n"
            if self.centered:       
                string += f"PDF will be centered with single bin\n"
            string += f"Sample generation will run on {self.N_cpu} core\n"
            string += "+"*80
            string += "\n\n"
                
            return string

    args = settings(extrapolants, N_samples, bin_width, confidence_levels, batch_mem, N_cpu, baseline, centered, constant)
    print(r"_________                                    .__  __             ____ ___                           __         .__        __             ___________         __  .__                __                 ")  
    print(r"\_   ___ \  ____   _____ ______   ____  _____|__|/  |_  ____    |    |   \____   ____  ____________/  |______  |__| _____/  |_ ___.__.   \_   _____/ _______/  |_|__| _____ _____ _/  |_  ___________  ")
    print(r"/    \  \/ /  _ \ /     \\____ \ /  _ \/  ___/  \   __\/ __ \   |    |   /    \_/ ___\/ __ \_  __ \   __\__  \ |  |/    \   __<   |  |    |    __)_ /  ___/\   __\  |/     \\__  \\   __\/  _ \_  __ \ ")
    print(r"\     \___(  <_> )  Y Y  \  |_> >  <_> )___ \|  ||  | \  ___/   |    |  /   |  \  \__\  ___/|  | \/|  |  / __ \|  |   |  \  |  \___  |    |        \\___ \  |  | |  |  Y Y  \/ __ \|  | (  <_> )  | \/ ")
    print(r" \______  /\____/|__|_|  /   __/ \____/____  >__||__|  \___  >  |______/|___|  /\___  >___  >__|   |__| (____  /__|___|  /__|  / ____|   /_______  /____  > |__| |__|__|_|  (____  /__|  \____/|__|    ")
    print(r"        \/             \/|__|              \/              \/                \/     \/    \/                 \/        \/      \/                \/     \/                \/     \/                    ") 
    # print user settings
    print(args)

    print("\n==== Master extrapolant table ====\n")
    headers_ext = [f"{'(X-1,X)'}"] + [f"extr{i}" for i in range(len(extrapolants[0])-1)]
    print_table(extrapolants, headers_ext)

    start_time = time.perf_counter() # start total execution timer

    # set up internal bin_width for pdf histogram. The final result will be calculated with the user specified value.
    ref_bin_width = args.bin_width / 10

    # prepare the middle bin depending if we have centered or non centered variant
    midbin = 0
    mids,integrals = generate_bins(pdf,ref_bin_width, centered = args.centered)
    if args.centered:
        midbin = 1


    baseline_sum,_ = best_estimate(extrapolants) # calculated the final value of extrapolation
    stats_tracker = StreamingStats(args.bin_width, baseline=args.baseline ) # set up StreamingStats class

    N_samples = args.N_samples
    total_iterations = N_samples * (2 * mids.size - midbin)

    print(f"\n--- Running composite random walk ---")
    print(f"Generating {N_samples:,} random walk tables...")
    print(f"Running {2 * mids.size - midbin:,} bins per table (Total: {total_iterations:,} limit projections)")

    # get basic extrapolation information (how many basis were used, how many composite methods we have)
    N_bas, N_method = np.asarray(extrapolants).shape
    N_method -= 1
    print(f" Number of extrapolants:      {N_bas:2d}")
    print(f" Number of composite methods: {N_method:2d}")

    

    # preprare samples, if we have only one method then one sample is needed as pdf gives exact results
    if N_method == 1 and N_samples > 1:
        raise ValueError("Only one composite method provided. Number of samples has to be one, as pdf provides exact results")
    
    _start_time = time.perf_counter()
    if N_method == 1 and N_samples == 1:
        current_samples = np.asarray([row[1:] for row in extrapolants]).reshape(1,-1,1)
    else:
        N_CPU = args.N_cpu
        pal_random = partial(generate_sample, (extrapolants)) # prepare the datapack for partial function
        with Pool(N_CPU) as pool:
            current_samples = np.asarray(pool.map(pal_random,[N_samples//N_CPU]*N_CPU)).reshape((N_samples,N_bas,N_method))
    print_memory_usage("Currently used")
    print_time(_start_time,'Composite sampling')

    # estimate the size of batched bulk and number of batches required
    bulk_batches, bulk_size = estimate_bulk_size(args.batch_mem, N_samples, N_bas, N_method, 2*mids.size - midbin, 6) 
    estimate_peak_memory(N_samples, N_bas, N_method, 2*mids.size - midbin, bulk_size, bulk_batches, 6)
    d = str(int(np.log10(N_samples)))+"d"
    d0 = str(int(np.log10(bulk_batches)))+"d"

    for ibulk in range(0, bulk_batches):
        bulk_start = ibulk * bulk_size
        bulk_end = min(bulk_start+bulk_size,N_samples)

        # cut out the batch from samples and renormalize it
        # batch has dimensions of bulk_size x N_bas x N_method
        print(f"Running bulk {ibulk+1:{d0}} of {bulk_batches:{d0}} {bulk_start:{d}} -> {bulk_end:{d}} / {N_samples}", end=' ')
        bulk_end = min(bulk_start + bulk_size, N_samples)
        batch = np.asarray(current_samples[bulk_start:bulk_end], dtype=float)/stats_tracker.baseline # cut out the batch from samples and renormalize it

        _start_time = time.perf_counter()
        # sum results for each composite method
        row_totals = np.sum(batch, axis=2)
        # calculate the  difference between final extrapolants
        gap = np.abs(row_totals[:, -1] - row_totals[:, -2])
        # rescale the original pdf with samples difference between extrapolants
        # values has dimensions of bulk_size x number_of_bins
        values = np.concatenate((-mids[midbin:][::-1],mids))[None, :] * gap[:, None] + row_totals[:, -1][:, None] # rescale the original pdf with samples difference between extrapolants 

        # copy the integrals to match dimensions of values array
        freq_tiled = np.broadcast_to(np.concatenate((integrals[midbin:][::-1],integrals)), values.shape) # 
        print_time(_start_time,'Bulk sampling',end= ' ')

        _start_time = time.perf_counter()
        stats_tracker.add_bulk(values.ravel(), freq_tiled.ravel()) # linearize arrays and add them in bulk to final histogram
        print_time(_start_time,'Bulk adding',end= ' ')

        elapsed_seconds = time.perf_counter() - start_time

        if elapsed_seconds > 60:
            minutes = int(elapsed_seconds // 60)
            seconds = elapsed_seconds % 60
            print(f" Total execution time: {minutes}m {seconds:.0f}s")
        else:
            print(f" Total execution time: {elapsed_seconds:.2f} seconds")

    # get average from the final histogram, should be almost exactly the same as value taken from table of extrapolants
    average = stats_tracker.get_average()

    print(f"\n- Best estimate (deterministic): {baseline_sum:.6f}")
    print(f"- Best estimate (stochastic)   : {average:.6f}\n")
    # determine the uncertainty at user defined confidence intervals
    for thr in args.confidence_levels: 
      lower, upper = stats_tracker.get_confidence_interval(confidence_level=thr)
      print(f"  {thr*100}% Confidence Interval: [{lower-average:.6f}, {upper-average:.6f}]")

    # prints the final execution timings
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    if elapsed_seconds > 60:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        print(f"\n Total execution time: {minutes}m {seconds:.2f}s")
    else:
        print(f"\n Total execution time: {elapsed_seconds:.3f} seconds")

    return stats_tracker



