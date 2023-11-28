import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

from DetectDelimiter import detect_delimiter

def KNNimpute(train_data_path, test_data_path):
    # Detect the file delimiter
    train_delimiter = detect_delimiter(train_data_path)
    test_delimiter = detect_delimiter(test_data_path)

    # Read and preprocess train data
    train_df = pd.read_csv(train_data_path, delimiter=train_delimiter, header=None)
    train_df.columns = ['Column_' + str(i) for i in range(1, train_df.shape[1] + 1)]
    train_df.replace(1.00000000000000e+99, np.nan, inplace=True)

    # Read and preprocess test data
    test_df = pd.read_csv(test_data_path, delimiter=test_delimiter, header=None)
    test_df.columns = ['Column_' + str(i) for i in range(1, test_df.shape[1] + 1)]
    test_df.replace(1.00000000000000e+99, np.nan, inplace=True)

    # Combine train and test data for imputation
    combined_df = pd.concat([train_df, test_df])

    best_k = find_best_k(train_df)
    print(f"Best K: {best_k}, replacing missing values with best K...")

    # Create and fit the imputer using combined data
    imputer = KNNImputer(n_neighbors=best_k)
    imputed_data = imputer.fit_transform(combined_df)

    # Separate imputed data back into train and test
    imputed_train_df = pd.DataFrame(imputed_data[:len(train_df)], columns=train_df.columns)
    imputed_test_df = pd.DataFrame(imputed_data[len(train_df):], columns=test_df.columns)

    # Save imputed data to files
    imputed_train_df.to_csv(train_data_path, sep=train_delimiter, index=False, header=False)
    imputed_test_df.to_csv(test_data_path, sep=test_delimiter, index=False, header=False)


def find_best_k(train_df):
    # Create a range of K values to test
    k_values = [3, 5, 7, 10, 12, 15, 20]
    
    best_k = None
    best_score = float('inf')  # Initialize with a large value
    
    for k in k_values:
        imputer = KNNImputer(n_neighbors=k)
        
        # Perform cross-validation to evaluate imputation quality
        imputed_data = imputer.fit_transform(train_df)
        imputed_train_df = pd.DataFrame(imputed_data, columns=train_df.columns)
        
        # Calculate mean squared error for imputed data
        mse = ((train_df - imputed_train_df) ** 2).mean().mean()
        
        # Check if this K gives a better MSE score
        if mse < best_score:
            best_score = mse
            best_k = k
    
    return best_k
