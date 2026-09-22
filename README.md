<p align="center">
  <img src="toc2.png?raw=true" alt="Table Of Content"/>
</p>
<h1>Uncertainty prediction in composite quantum chemistry approaches</h1>

This repository contains a PYTHON implementation of a method of estimating the uncertainty of a result obtained through extrapolation to the complete basis set limit for composite quantum chemical approaches.
The method is based on an ensemble of random walks which simulate all possible extrapolation outcomes that could have been obtained if results from larger basis sets had been available.
The results assembled from a large collection of random walks can be then analyzed statistically, providing a route for uncertainty prediction at a confidence level required in a particular application.
The method is free of empirical parameters and compatible with any extrapolation scheme. The proposed technique has been tested in a series of numerical trials by comparing the determined confidence intervals with reliable reference data.

<h2>Usage</h2>

The script can be used as it is or function <b>run_random_walk</b> from random_walk module can be imported in other work.
If run as a "main" script it requires a path to a file containing the data to be extrapolated.<br>
An example of input file is:<br><br>
             &emsp;2 &emsp;   3.22401832 &emsp;  	-0.8730818492<br>
             &emsp;3 &emsp;   4.599041773&emsp;	  -0.7819656691<br>
             &emsp;4 &emsp;   5.132946242&emsp; 	-0.7954372818<br>
             &emsp;5 &emsp;   5.329771198&emsp;     nan      <br>

with each line containing a list of (<i>X, e<sub>0 </sub>, e<sub>1 </sub>, e<sub>2 </sub>, ...</i>), where <i>X</i> is a cardinal number of basis and <i>e<sub>i </sub></i> are particular components of the composite method to be extrapolated. Missing data for higher basis sets can be either a string or empty.

<b>As a script</b>

    """
    Method to run the composite random walk

    Parameters
    ----------
    filename  : (str) path to the file containing data to be extrapolated and their uncertainty estimated
    N_samples : (int) Number of random samples for which we run the uncertainty estimation.  Has to be larger than 0.
                      Default is 100 000. For production run at least 1 000 000 of samples should be used.
    bin_width : (float) Bin width of the final histogram across all samples. Roughly translates to relative error in uncertainty estimation
                        Default values is 0.0001.
    levels    : (list) List of floats at which level final uncertainty is printed.
    mem       : (int) Maximum allowed memory in MB which can be used during estimation run. Default is unrestricted run.
    N_cpu     : (int) Number of processes used to generate sample table. Default is 10.
    graph     : Whether to save a graph of the final histogram
    csv       : Whether to save a csv file of the final histogram
    """


Run as 

````
python3 random_walk_composite.py --N_samples 1_000_000 --bin_width 1e-4 --levels 0.75 0.95 0.99 --mem 6_000 --N_cpu 10 --graph --csv n2.dat
````
<b>As an imported function with custom extrapolation</b>
````{python3}
from random_walk import run_random_walk
from parse_distribution import load_and_interpolate

pdf = load_and_interpolate("merged_data.csv")
extrapolants =  some_custom_function_to_load_or_extrapolate_data(filename)

stats_tracker = **run_random_walk**(pdf, extrapolants, N_samples=1_000_000, bin_width=1e-4, levels=[0.75, 0.95, 0.99], mem=6_000, N_cpu=10)

````

Example files are `water.dat`, and `n2.dat` which contain necessary data to reproduce results in section IV. Results should be the same within some stochastic noise on the last digit.

File `merged_data.csv` contains the numerical integration results for Eq. (5) from 0.01 to 6.2. 

File `supplementary_material.xlsx` contains the data to reproduce results in section V.


<b>CITATION:</b> [Lang, J.; Raczko, K.; Lesiuk, M.;Uncertainty prediction in composite quantum chemistry approaches]()
         

