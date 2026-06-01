import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


def read_csv(filename='../data/AppML_ZTF_table.csv'):
    df = pd.read_csv(filename)
    return df

def read_csv_numpy(filename='../data/AppML_ZTF_table.csv'):
    data = np.genfromtxt(filename, delimiter=',', names=True)
    return data

def read_get_classification_labels(filename='../data/AppML_ZTF_table.csv', type="classification"):
    """Get just the classification labels as a numpy array. 
    Useful for external validation.
    Can specify other label types (e.g. tns_type) if needed.
    """
    df = pd.read_csv(filename)
    return df[type].values

def get_object_ids(filename='../data/AppML_ZTF_table.csv', idxs=None):
    """Get object IDs for given indices. Useful for cross-referencing with TNS."""
    df = pd.read_csv(filename)
    if idxs is not None:
        return df.loc[idxs, 'objectId'].values
    else:
        return df['objectId'].values

def remove_columns(df, columns_to_remove=None):
    """
    Remove specified columns from the DataFrame. 
    If no columns are specified, remove a default set of metadata columns.
    """
    if columns_to_remove is None:
        drop_cols = ["id", "objectId", 'decstd', 'rastd', "htm16", "ssnamenr", "tns_name", "tns_type",
                 "host_name", "classification", "classificationReliability"]
    else:
        drop_cols = columns_to_remove
    df = df.drop(columns=drop_cols)
    return df

def impute_and_scale(df):
    # Impute (median is robust to outlier-heavy astro data)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df)

    # Scale — RobustScaler handles the heavy-tailed distributions typical in ZTF
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled