import time
import numpy as np
from scipy import integrate, interpolate

from helpers import print_time

# coefficients for the analytical solution near 0
_A = 0.2758797978316264
_B = 0.05318817623062113
_C = 0.1787181037859664
_D = 0.02233976297324580

def sol_at_zero(x):
    """ Return analytical solution """
    return _A - _B * x**2 - _C * np.log(np.abs(x)) + _D * x**2 * np.log(np.abs(x))

class interpolatedPDF():
    """
    Class containing information about probability distribution, its interpolation and antiderivative.
    As Eq. (5) is numerically unstable and even not defined for y = 0, class object also provides results from analytical solution obtained near zero.
    """
    def __init__(self, y, rhoy, k=1, analytical_y = 0.001):
        """
        Initialization method
        
        parameters
        ----------
        y           : (list) values at which Eq. (5) was sampled and calculated. Has to be defined for positive values. 
        rhoy        : (list) values of the Eq. (5) at y points
        k           : (int) order of interpolation. Default is linear interpolation.
        analytical_y: (float) value below which analytical solution of Eq. (5) should be used instead of numerical results. Default is the smallest value of y.
        """

        assert np.all(y>0)
        self._pdf = interpolate.make_interp_spline(y,rhoy,k=k)
        self._cdf = self._pdf.antiderivative()
        self._x = y
        if analytical_y is None:
            self.analytical_y0 = self._x[0]
        else:
            self.analytical_y0 = analytical_y

    def __call__(self, x):
        """ Get a value of pdf at x. """
        assert np.all(x > 0)
        return self._pdf(x)

    def cdf(self, x):
        """ Calculate the antiderivative of pdf at x. """
        assert np.all(x > 0)
        return self._cdf(x)

    def analytical_integration(self, left_anal, right_anal):
        """ Calculate integral values of bin in analytical region. """
        def F(x):      
            """ Get analytical integral of pdf at x """
            if np.any(x) < 0:
                raise Exception("Negative X in analytical integration") 
            x_safe = np.where(x > 0, x, 1.0)
            log_x = np.where(x > 0, np.log(x_safe), 0.0)
            return (_A + _C) * x - (_B + _D / 3) / 3 * x**3 - _C * x * log_x + (_D / 3) * x**3 * log_x

        res = F(right_anal) - F(left_anal)

        if isinstance(res,np.ndarray):
            pass # do not print information and it would be too long
        else:
            print(f"Analytical solution for ∫_{{{left_anal:.3e}}}^{{{right_anal:.3e}}} = {res:.6e}")
        return res


def check_data(x, y):
    """
    Checks the loaded distribution to determine any potential problems for interpolation

    Parameters:
    -----------
    x, y : (arrays) arrays of monotonic data points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    sort_idx = np.argsort(x)
    x, y = x[sort_idx], y[sort_idx]

    dx = np.diff(x)
    dy = np.diff(y)
    x_range = x[-1] - x[0]
    y_range = np.ptp(y)


    print("\n")
    print("*"*80)
    imax = np.argmax(x)
    imin = np.argmin(x)
    print(f"Max x={x[imax]:.5f} y={y[imax]:.6e}, Min x={x[imin]:.5f} y={y[imin]:.6e}")
    print(f"Max y={np.max(y):.6e} , Min y={np.min(y):.6e}")

    print("\n=== MESH DENSITY ===")
    min_dx, max_dx = np.min(dx), np.max(dx)
    dx_ratio = max_dx / min_dx
    print(f"Min Δx: {min_dx:.2e} | Max Δx: {max_dx:.2e}")
    print(f"Grid anisotropy (max Δx / min Δx): {dx_ratio:.2f}x")
    if dx_ratio > 10:
        print("  HIGH DENSITY VARIATION")
    if min_dx < 1e-10 * x_range:
        print("  NEAR-DUPLICATE X")
        idx = np.argmin(dx)
        print("  X with smallest step",x[idx-1],x[idx],x[idx+1])

    scale_aspect = y_range / x_range if x_range != 0 else 0
    print(f"Scale aspect ratio (Δy_tot / Δx_tot): {scale_aspect:.2e}")
    if scale_aspect > 1e4 or scale_aspect < 1e-4:
        print("  POOR SCALING")


    print("\n=== MONOTONICITY ===")
    positive_slope_mask = dy/dx > 1e-12
    if np.any(positive_slope_mask):
        idx_max_pos_slope = np.argmax(dy)
        max_pos_slope = dy[idx_max_pos_slope]
        print(f"  MONOTONICITY IN DATA POINTS VIOLATED: Max positive slope = {max_pos_slope:.3e} at x={x[idx_max_pos_slope]}")
        for i in np.where(positive_slope_mask)[0]:
            print(f"  Index {i} -> {i+1}: ({x[i]:.6f}, {y[i]:.12f}) -> ({x[i+1]:.6f}, {y[i+1]:.12f}) | dy/dx = {dy[i]/dx[i]:.4e}")
    else:
        print("  Data maintains decreasing direction.")

    print("*"*80)
    print()

def load_and_interpolate(filename):
    """ 
    Function to load the probability distribution function data from CSV file and create object containing all the necessary information about distribution for the calculations.
    Assumes the analytical solution is valid till 0.01

    Returns an interpolatedPDF class object
    """
    
    start_time = time.perf_counter()
    data = np.genfromtxt(filename,delimiter=',') #load PDF datapoints
    print(f"Number of points in CSV file {data.shape[0]-1} with {data.shape[1]} entries")
    x = data[1:,1]
    y = data[1:,2]

    x0 = x[0]
    xl = x[-1]

    analytical_x = 0.01

    pdf = interpolatedPDF(x,y,k=1, analytical_y = analytical_x)

    idx = np.where((x>=analytical_x) & (x<=xl))[0]
    x = x[idx]
    y = y[idx]
    check_data(x, y)

    numl_norm = pdf.cdf(xl) - pdf.cdf(analytical_x)
    anal_norm = pdf.analytical_integration(0,analytical_x)
    full_norm = numl_norm + anal_norm

    # check if the normalization of half of pdf is equal ot 0.5. 
    print(f"Numerical norm from {analytical_x:.3e} to {xl:.3e}  is {numl_norm:.5e}")
    print(f"Analytical integral from 0 to {analytical_x:.3e}     is {anal_norm:.5e}")
    print(f"Total norm from {x0:.3e} to {xl:.3e}      is {full_norm:.5e}")
    assert np.isclose(full_norm,0.5)

    print_time(start_time, "Interpolatng distribution")
    print("")

    return pdf
