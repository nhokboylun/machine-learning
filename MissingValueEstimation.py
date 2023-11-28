import numpy as np

def performEM(data_path, filled_data_path):
    # Load data, replace 1e+99 with NaN for missing values
    data = np.loadtxt(data_path, delimiter='\t')
    missing_mask = (data == 1e+99)  # Identify missing values
    data[missing_mask] = np.nan

    # Initialize missing values with column means
    col_means = np.nanmean(data, axis=0)
    data[missing_mask] = np.take(col_means, np.where(missing_mask)[1])

    # EM algorithm
    for iteration in range(150):  
        # E-step: Estimate missing values
        col_means = np.nanmean(data, axis=0)
        col_stddev = np.nanstd(data, axis=0)

        # Only update missing values
        data[missing_mask] = np.take(col_means, np.where(missing_mask)[1]) + np.random.randn(np.sum(missing_mask)) * np.take(col_stddev, np.where(missing_mask)[1])

    # Save the filled data
    np.savetxt(filled_data_path, data, delimiter='\t', fmt='%.15f')

