import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

def load_and_format_csv(file_path):

    df = pd.read_csv(file_path)

    # Drops any rows where all values are missing.
    df.dropna(how='all', inplace=True)
    # Drops any columns where all values are missing.
    df.dropna(axis=1, how='all', inplace=True)
    # Removes any row that has x >= 1 missing value.
    df.dropna(inplace=True)
    # Keeps only numeric columns | deletes text or categorical columns.
    df = df.select_dtypes(include=['number'])

    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and target column.")

    print(f"Loaded CSV with shape: {df.shape}")
    
    return df

def scale_dataset(dataframe, oversample=False):

    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (len(y), 1))))

    return data, X, y
