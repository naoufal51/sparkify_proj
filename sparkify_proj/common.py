from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import os
import pandas as pd
from pandas import DataFrame as pd_DataFrame
import subprocess


def get_stratified_train_test_split(df: DataFrame, target_col: str, train_prop: float, seed: int):
    """
    Perform stratified sampling to split the data into train and test sets.
    It is used to mitigate the effects of class imbalance when splitting the data.

    Args:
        df (DataFrame): Dataframe containing the data
        target_col (str): Name of the target column
        train_prop (float): Proportion of the data to use for training
        seed (int): Random seed for reproducibility

    Returns: 
        train (DataFrame): Dataframe containing the training data
        test (DataFrame): Dataframe containing the test data
    """
    # Calculate the proportions of each class in the target column
    majority_class = df.filter(col(target_col) == 0)
    minority_class = df.filter(col(target_col) == 1)

    # Split the majority class into train and test sets
    train_majority, test_majority = majority_class.randomSplit(
        [train_prop, 1 - train_prop], seed)

    # Split the minority class into train and test sets
    train_minority, test_minority = minority_class.randomSplit(
        [train_prop, 1 - train_prop], seed)

    # Combine the train and test sets
    train = train_majority.union(train_minority)
    test = test_majority.union(test_minority)

    return train, test


def download_from_s3(s3_path, local_path):
    """
    Download a file from an S3 bucket to the local machine.

    Args:
        s3_path (str): Path to the source dataset for churn prediction
        local_path (str): Path to the local directory

    """
    try:
        subprocess.check_output(
            ['aws', 's3', 'cp', s3_path, local_path, '--no-sign-request'])
        print(f"File downloaded successfully from {s3_path} to {local_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading file from {s3_path} to {local_path}: {e}")


def remove_file(file_path):
    """
    Remove a file from the local machine.
    Args:
        file_path (str): Path to the file to be removed

    """
    try:
        os.remove(file_path)
        print(f"File removed successfully from {file_path}")
    except OSError as e:
        print(f"Error removing file from {file_path}: {e}")


def save_data(df: DataFrame, path: str, file_name: str):
    """
    Save the data to parquet files.

    Args:
        df (DataFrame): Dataframe containing the data
        path (str): Path to the local directory
        file_name (str): Name of the file to be saved

    """
    df.write.mode('overwrite').parquet(path + file_name)
    print(f"File saved successfully to {path + file_name}")


def save_pd_parquet(df: pd_DataFrame, path: str, file_name: str):
    """
    Save the data to parquet files. We use gzip compression.

    Args:
        df (DataFrame): Dataframe containing the data
        path (str): Path to the local directory
        file_name (str): Name of the file to be saved

    """
    df.to_parquet(path + file_name, compression='gzip')
    print(f"File saved successfully to {path + file_name}")
