# imports
import pandas as pd
import numpy as np
import os
import sys
from urllib.request import urlretrieve
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# helper to get data
# get data, downloads from url if not in relative path
def get_data(data_path):
    # checkes if file exists, if not downloaded to that path
    url = "https://raw.githubusercontent.com/tonyliang19/econ323_final_project/main/data/boston_housing_data.csv"
    if not os.path.isfile(data_path):
        print(f"You don't have the file yet, and it will be downloaded to: {os.path.abspath(data_path)}")
        print("Downloading now, wait a few secs")
        urlretrieve(url, data_path)
        print("Done!")
    else:
        pass
        
    data = pd.read_csv(data_path)
    return data

# function to load and split data

# loads data from path, and specifies proportion of train data, with 1 - proportion of test data
# and target is the variable of interest (your y), then returns train and test data
# proportion is default to 0.5, and target default to None 
def split_data(data_path, proportion=0.5, target=None, random_state=123, drop_na = True, **kwargs):
    """
    Loads data from path, and specifies proportion of train data, with 1 - proportion of test data
    and target is the variable of interest (your y), then returns train and test data
    proportion is default to 0.5, and target default to None.
    
    Optional argument:
    random_state = 123 (default), change to other number of your choice to assert reproducibility
    """
    # load the data
    data = pd.read_csv(data_path)
    # drop nas
    if drop_na is True:
        data = data.dropna()
    # inner function to split data into train and test portion
    def train_test_split(data, proportion, plain=True, chosen_cols=None):
        train = data.sample(frac = proportion, random_state=random_state)
        test = data.drop(train.index)
        # rest and remove index of both
        train = train.reset_index().drop(columns=["index"])
        test = test.reset_index().drop(columns=["index"])
        # asserting dimension matches (i.e. number of rows)
        assert train.shape[0] + test.shape[0] == data.shape[0]
        return train, test
    # split the data into train and test
    train, test = train_test_split(data, proportion, **kwargs)
    # further split train data to X and y
    def split_X_y(data, plain=True, chosen_cols=None):
        if plain is not True:
            X = data[chosen_cols]
        else:
            X = data.drop(columns=[target])
        y = data[target]
        return X, y
    X_train, y_train = split_X_y(train, **kwargs)
    # split test data to X and y
    X_test, y_test = split_X_y(test, **kwargs)
    # check dimension again
    assert X_train.shape[0] + X_test.shape[0] == data.shape[0]
    assert y_train.shape[0] == X_train.shape[0] and y_test.shape[0] == X_test.shape[0]
    # return the objects needed
    return X_train, X_test, y_train, y_test


# Preprocessing and Feature engineering
# common preprocessor for data
def preprocess_data(df, drop="RAD"):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    try:
        numeric_cols.remove(drop)
    except:
        pass
    # transformers
    
    # impute NA values by median
    numeric_transformer = make_pipeline(SimpleImputer(strategy="median"),
                                        StandardScaler())
    try:
        preprocessor = make_column_transformer(
            (numeric_transformer, numeric_cols),# scaling on numeric features
        )
    except:
        preprocessor = make_column_transformer(
            (numeric_transformer, numeric_cols),
            ("drop", [drop]) # drop RAD, since it is index-like obj
        )
    return preprocessor

# simple helper to merge dataframe with same columns
def merge_data(*df):
    merged = pd.concat([*df])
    return merged