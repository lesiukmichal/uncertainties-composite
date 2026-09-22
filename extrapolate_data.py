import numpy as np
from helpers import print_table

def read_dynamic_data(filename):
    """
    Reads a data file with an unknown, variable number of columns.
    Pads any missing trailing columns with NaN so all rows are the same length.
    """
    raw_data = []
    max_cols = 0

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            columns = line.split()
            if not columns:
                continue
            try:
                parsed_row = [float(val) for val in columns]
                raw_data.append(parsed_row)
                if len(parsed_row) > max_cols:
                    max_cols = len(parsed_row)
            except ValueError as e:
                print(f"Warning: Could not parse line {line_number}: {line.strip()} | Error: {e}")

    padded_data = []
    for row in raw_data:
        missing_count = max_cols - len(row)
        row.extend([np.nan] * missing_count) # if empty cells are found, pad with NaN.
        padded_data.append(row)

    return padded_data, max_cols

def calculate_extrapolants(data):
    """
    Calculates the extrapolated values for adjacent pairs of X.
    Helgaker's formula: (E_X * X^3 - E_{X-1} * X_{X-1}^3) / (X^3 - X_{X-1}^3)
    """
    extrapolants_table = []

    for i in range(1, len(data)):
        row_prev = data[i-1]
        row_curr = data[i]

        x_prev = row_prev[0]
        x_curr = row_curr[0]

        pair_label = f"{int(x_prev)}Z->{int(x_curr)}Z"
        extrapolated_row = [pair_label]

        denom = (x_curr**3) - (x_prev**3)

        for col_idx in range(1, len(row_curr)):
            e_prev = row_prev[col_idx]
            e_curr = row_curr[col_idx]

            # if data is missing or denominator is zero, append NaN
            if np.isnan(e_prev) or np.isnan(e_curr) or denom == 0:
                extrapolated_row.append(np.nan)
            else:
                num = (e_curr * (x_curr**3)) - (e_prev * (x_prev**3))
                extrapolated_row.append(num / denom)

        extrapolants_table.append(extrapolated_row)

    return extrapolants_table


def preprocess_extrapolants(filename):
    """
    Load and extrapolated data from file
    Expected format of file is (header just for column illustration)
    L   method1  method2 ....
    2    val      val
    3    val      val
    4    val      val
    5    val      nan 
    6    val      nan
    , where val is either a value in specific basis or zero. At least 3 vals has to be non zero for each column 
    to kickstart extrapolations. nan is either some string or empty. L specifies the cardinal number of basis set
    """

    data, total_columns = read_dynamic_data(filename)
    headers_raw = ["X"] + [f"Col {i}" for i in range(1, total_columns)]
    print("\n--- RAW DATA ---")
    print_table(data, headers_raw)

    extrapolants = calculate_extrapolants(data)

    headers_ext = ["Pair"] + [f"Extrap {i}" for i in range(1, total_columns)]
    print("\n--- EXTRAPOLANTS ---")
    print_table(extrapolants, headers_ext)

    return extrapolants, data
