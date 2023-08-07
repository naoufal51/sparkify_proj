import toml
from prefect import task, flow
import importlib
from sparkify_proj.model_experimenter import ModelExperiment
from sparkify_proj.preprocessing import Preprocessing
from sparkify_proj.feature_engineering import FeatureEngineering
from sparkify_proj.pipeline import DataPipeline
from sparkify_proj.model_explainer import ModelExplainer

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from hyperopt import hp
import os
import shutil
from datetime import datetime
from haikunator import Haikunator
import wandb
import boto3
import botocore

import glob
import json


# Machine Learning libraries
from sklearn.ensemble import (GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from pyspark.sql import SparkSession
import numpy as np
import pickle as pkl


def init_and_create_directory(path: str, fixed_name: Optional[str] = None, clear: bool = False) -> Optional[str]:
    """
    Initialize a base directory and create a subdirectory.
    If clear is True, the base directory is cleared if it exists.

    Args:
        path (str): Path to the base directory.
        fixed_name (Optional[str]): Fixed name for the subdirectory.
        clear (bool): Whether to clear the base directory if it exists.

    Returns:
        session_directory (Optional[str]): Path to the session subdirectory.

    """

    if os.path.exists(path):
        if clear:
            # Remove all files and directories
            shutil.rmtree(path)
            # Create the base directory
            os.makedirs(path, exist_ok=True)
    else:
        print("We are creating the base directory")
        # Create the base directory
        os.makedirs(path, exist_ok=True)

    dir_name = fixed_name if fixed_name else f'{Haikunator().haikunate()}-{datetime.now().strftime("%Y%m%d%H%M%S")}'

    # Create a new directory
    session_directory = os.path.join(path, dir_name)

    try:
        os.makedirs(session_directory, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory. Error: {e}")
        return None

    return session_directory


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = toml.load(f)
    return config


def load_model(path: str):
    """
    Load the model from the given path

    Args:
        path (str): path to the model

    Returns:
        model (sklearn model): sklearn model
    """
    try:
        with open(path, 'rb') as f:
            model = pkl.load(f)
        return model
    except Exception as e:
        print(e)


@task
def fetch_data(bucket_name: str, file_name: str, raw_data_path: str, run) -> None:
    """
    Fetch the data from S3 and return a spark dataframe

    Args:
        bucket_name (str): Name of the S3 bucket
        file_name (str): Name of the file in the S3 bucket
        raw_data_path (str): Path to save the raw data
        run (wandb run): wandb run object

    Returns:
        None
    """
    if not os.path.exists(raw_data_path):
        os.mkdir(raw_data_path)
    local_file_name = file_name.split('/')[-1]
    s3_client = boto3.client('s3', config=botocore.client.Config(
        signature_version=botocore.UNSIGNED))
    s3_client.download_file(bucket_name, file_name,
                            f'{raw_data_path}/{local_file_name}')

    # create Artifact wandb called raw_data
    artifact = wandb.Artifact(
        name='raw_data',
        type='raw_data',
        description='raw data from s3 for sparkify project',
        metadata=dict(file_name=file_name),

    )
    artifact.add_dir(raw_data_path)
    run.log_artifact(artifact)


@task
def preprocess_data(raw_data_path: str, save_path: str, spark_master: str, spark_app_name: str, run) -> None:
    """
    Preprocess the data

    Args:
        path (str): path to the raw data
        save_path (str): path to save the preprocessed data
        spark_master (str): spark master url / local
        spark_app_name (str): spark app name

    Returns:
        None

    """
    # return spark session
    spark = SparkSession \
        .builder \
        .master(spark_master) \
        .appName(spark_app_name) \
        .getOrCreate()

    # fetch the raw data
    df = spark.read.json(raw_data_path)

    # repartition the data to avoid out of memory errors
    df = df.repartition(100)

    # Instantiate the preprocessing class and preprocess the data
    preprocessor = Preprocessing(df)
    preprocessed_data = preprocessor.preprocess_data()

    # save the data coallesce(1) is used to save the data as a single file
    path_preprocessed_data = save_path
    preprocessed_data.write.parquet(
        path_preprocessed_data, mode='overwrite')

    # create Artifact wandb called preprocessed_data
    artifact = wandb.Artifact(
        name='preprocessed_data',
        type='preprocessed_data',
        description='preprocessed data for sparkify project',
        metadata=dict(file_name=raw_data_path),
    )
    artifact.add_dir(path_preprocessed_data)
    run.log_artifact(artifact)

    spark.stop()


@task
def feature_engineering(path_preprocessed_data: str, save_path: str, spark_master: str, spark_app_name: str, run) -> None:
    """
    Feature engineering

    Args:
        path_preprocessed_data (str): path to preprocessed data
        save_path (str): path to save the feature engineered data
        spark_master (str): spark master url/local
        spark_app_name (str): spark app name
        run (wandb run): wandb run object

    Returns:
        None
    """
    # create spark session
    spark = SparkSession \
        .builder \
        .master(spark_master) \
        .appName(spark_app_name) \
        .getOrCreate()

    df = spark.read.parquet(path_preprocessed_data)

    # df repartitioned to 20 partitions to out of memory error
    df = df.repartition(100)

    # Instantiate the feature engineering class and perform feature engineering
    feature_engineering = FeatureEngineering(df)
    feature_engineered_data = feature_engineering.feature_engineering()

    path_feature_engineered_data = save_path
    feature_engineered_data.coalesce(1).write.parquet(
        path_feature_engineered_data, mode='overwrite')

    # create Artifact wandb called feature_engineered_data
    artifact = wandb.Artifact(
        name='feature_engineered_data',
        type='feature_engineered_data',
        description='feature engineered data for sparkify project',
        metadata=dict(file_name=path_preprocessed_data),
    )

    artifact.add_dir(path_feature_engineered_data)
    run.log_artifact(artifact)

    spark.stop()


@task
def data_pipeline(feature_engineered_data_path: str, path_proc_data_path: str, path_pipeline_model: str, spark_master: str, spark_app_name: str, run) -> None:
    """
    Run the data pipeline

    Args:
        feature_engineered_data_path: path to the feature engineered data
        path_proc_data_path: path to save the processed data
        path_pipeline_model: path to save the pipeline model
        spark_master: spark master url/local
        spark_app_name: spark app name
        run: wandb run
    Returns:
        None
    """
    # create spark session
    spark = SparkSession \
        .builder \
        .master(spark_master) \
        .appName(spark_app_name) \
        .getOrCreate()

    # read the feature engineered data
    feature_engineered_data = spark.read.parquet(feature_engineered_data_path)

    # Instantiate the data pipeline class and run it
    data_pipeline = DataPipeline(feature_engineered_data, path_proc_data_path)
    X_train, y_train, X_test, y_test = data_pipeline.run()

    # create Artifact wandb called data_pipeline
    artifact = wandb.Artifact(
        name='data_pipeline',
        type='data_pipeline',
        description='data pipeline for sparkify project',
        metadata=dict(file_name=feature_engineered_data_path),
    )

    artifact.add_dir(path_proc_data_path)
    run.log_artifact(artifact)

    # create Artifact wandb pipeline_model
    artifact = wandb.Artifact(
        name='pipeline_model',
        type='pipeline_model',
        description='pipeline model for sparkify project, contains the fitted pipeline (preprocessing)',
        metadata=dict(file_name=feature_engineered_data_path),
    )
    artifact.add_dir(path_pipeline_model)
    run.log_artifact(artifact)

    spark.stop()


@task
def init_model_experiment(config_file: str, max_evals: int, seed: int) -> ModelExperiment:

    def calculate_weights(y):
        flat_y = y.values.ravel()
        class_weight_current = len(flat_y) / (2.0 * np.bincount(flat_y))
        return {0: class_weight_current[0], 1: class_weight_current[1]}

    y_train = pd.read_parquet('data/sparkify/proc_data/y_train.gzip')
    weights = calculate_weights(y_train)

    # Load the configuration file
    config = load_config(config_file)

    # Define the classifiers
    classifiers = {}
    base_class = {}

    for clf_name, clf_info in config["models"].items():
        module_name, class_name = clf_info["class"].rsplit(".", 1)
        MyClass = getattr(importlib.import_module(module_name), class_name)
        args = clf_info.get("args", {})

        if clf_name == 'XGBClassifier':
            args["scale_pos_weight"] = weights[0]/weights[1]
        clf = MyClass(**args)

        classifiers[clf_name] = clf
        base_class[clf_name] = clf

    base_classifiers = [(name, deepcopy(clf))
                        for name, clf in base_class.items()]

    meta_learner = LogisticRegression(max_iter=5000)
    # Define stacking classifier
    stacking_classifier = StackingClassifier(
        estimators=base_classifiers, final_estimator=meta_learner)

    # Include the stacking classifier in the classifiers dictionary
    classifiers['StackingClassifier'] = stacking_classifier

    # Define spaces for hyperparameters
    hyperparameters = {}

    for model, params in config["hyperparameters"].items():
        hyperparameters[model] = {}
        for param, values in params.items():
            print(f'Processing: {model} {param} {values}')
            hyperparameters[model][param] = getattr(
                hp, values["type"])(param, *map(float, values["args"]))

    # Initialize the model experimenter
    model_experimenter = ModelExperiment(classifiers=classifiers, spaces=hyperparameters,
                                         max_evals=max_evals, seed=seed, workdir=config["dirs"]["session_dir"])
    return model_experimenter


@task
def run_model_experiment(model_experimenter: ModelExperiment, proc_data: str):
    """
    Task to run the model experiment

    Args:
        model_experimenter (ModelExperiment): model experimenter

    Returns:
        results (dict): results of the model experiment
    """

    # Load the data
    X_train = pd.read_parquet(f'{proc_data}/X_train.gzip')
    y_train = pd.read_parquet(f'{proc_data}/y_train.gzip')

    X_test = pd.read_parquet(f'{proc_data}/X_test.gzip')
    y_test = pd.read_parquet(f'{proc_data}/y_test.gzip')

    # run the model experiment
    model_experimenter.run(X_train, y_train, X_test, y_test)


@task
def run_model_calibration(model_experimenter: ModelExperiment, proc_data: str):
    """
    Task to run the model calibration experiment

    Args:
        model_experimenter (ModelExperiment): model experimenter
        X_train (pd.DataFrame): training features
        y_train (pd.DataFrame): training labels
        X_test (pd.DataFrame): test features
        y_test (pd.DataFrame): test labels

    Returns:
        results (dict): results of the model experiment
    """
    # Load the data
    X_train = pd.read_parquet(f'{proc_data}/X_train.gzip')
    y_train = pd.read_parquet(f'{proc_data}/y_train.gzip')

    X_test = pd.read_parquet(f'{proc_data}/X_test.gzip')
    y_test = pd.read_parquet(f'{proc_data}/y_test.gzip')

    # run the model calculator
    results = model_experimenter.run_calibration_exp(
        X_train, y_train, X_test, y_test)
    # Visualize calibration results
    model_experimenter.vizualize_calibration(X_test, y_test)


@task
def model_explainability(session_dir: str, n_samples:int):
    """
    Task to perform model explainability using ModelExplainer class (SHAP under the hood)

    Args:
        session_dir (str): path to the session directory

    Returns:
        None
    """
    # load the data
    # artifact = run.use_artifact('data_pipeline:latest')
    # path = artifact.download()
    path = f'{session_dir}/proc_data'

    X_train = pd.read_parquet(f'{path}/X_train.gzip')
    X_test = pd.read_parquet(f'{path}/X_test.gzip')
    y_train = pd.read_parquet(f'{path}/y_train.gzip')
    y_test = pd.read_parquet(f'{path}/y_test.gzip')

    # convert y_train and y_test to pandas series
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # get feature names
    feature_names = X_train.columns.tolist()

    # loop over all the saved models (check the path and experiment_session_id)
    models_path = f'{session_dir}/models/'
    artifacts_path = f'{session_dir}/artifacts/'
    for model_name in [m for m in os.listdir(models_path) if not m.startswith('calibrated')]:
        print(model_name)
        model = load_model(f'{models_path}{model_name}')
        # extract model name
        model_name = model_name.split('.')[0]
        # initialize ModelExplainer
        print('Initializing ModelExplainer...')
        print('model_name: ', model_name)
        explainer = ModelExplainer(model=model, model_name=model_name,
                                   X_train=X_train, X_test=X_test, feature_names=feature_names)
        # check if explainer object has been instantiated
        print(explainer)
        # compute SHAP values
        explainer.compute_shap_values(n_samples=n_samples)
        # plot SHAP values
        explainer.plot_shap_values(path=artifacts_path, n_samples=n_samples)
        # plot SHAP force for a specific sample and a specific output
        explainer.plot_shap_force(
            sample_index=0, output_index=0, path=artifacts_path)


@task
def log_wandb_artifact(artifact_name: str, dir_path: str, models_path: str, run):
    """
    Task to log the artifacts to W&B

    Args:
        artifact_name (str): name of the artifact
        dir_path (str): path to the directory containing the artifacts
        models_path (str): path to the directory containing the models
        run (wandb.Run): W&B run object

    Returns:
        None
    """

    def extract_model_name(file_name: str) -> str:
        """
        Extracts the model name from the file name.

        Args:
            file_name (str): The file name.

        Returns:
            str: The model name.
        """
        prefixes = ["confusion_matrix",
                    "calibration_comparison", "pr_roc_curves"]
        suffixes = ["_shap_force_0_0", "_shap_values", "_shap_force", ""]

        # Remove extension from the file name
        file_name = file_name.rsplit('.', 1)[0]

        # Remove the prefix from the file name
        for prefix in prefixes:
            if file_name.startswith(prefix + "_"):
                file_name = file_name[len(prefix) + 1:]
                break

        # Remove the suffix from the file name
        for suffix in suffixes:
            if file_name.endswith(suffix):
                file_name = file_name[: -len(suffix) if suffix else None]
                break

        return file_name

    artifact = wandb.Artifact(artifact_name, type='result')
    file_paths = glob.glob(f'{dir_path}/*')
    for file_path in file_paths:
        # Check if the file path is a json file containing metrics
        if file_path.endswith('.json'):
            # Load the json file
            with open(file_path, 'r') as f:
                metrics_dict = json.load(f)

            # Log metrics to W&B
            for model, metrics in metrics_dict.items():
                for metric_name, metric_value in metrics.items():
                    run.log({f'{model}/{metric_name}': metric_value})
        elif file_path.endswith('.png'):
            # Log the the figures to W&B as images
            image = wandb.Image(file_path)
            file_name = os.path.basename(file_path)
            model_name = extract_model_name(file_name)
            run.log({f'{model_name}/{file_name}': image})
        else:
            # add all other files to the artifact
            artifact.add_file(file_path, name=os.path.basename(file_path))
    run.log_artifact(artifact)

    # log models to wandb
    for model_name in os.listdir(models_path):
        model_path = f'{models_path}/{model_name}'
        model_artifact = wandb.Artifact(
            model_name, type='model', description='Trained model', metadata=dict(model_name=model_name))
        model_artifact.add_file(model_path, name=model_name)
        run.log_artifact(model_artifact)


@flow(log_prints=True)
def sparkify_flow():

    # load config file
    config_file_path = "./config.toml"
    config = load_config(config_file_path)

    # login to wandb
    wandb.login(key=config['wandb']['key'])
    # wandb.login(key=os.environ.get('WANDB'))

    # login to prefect cloud
    os.system(f'prefect cloud login -k {config["prefect"]["key"]}')
    # os.system(f'prefect cloud login -k {os.environ.get("PREFECT")}')

    # initialize wandb run for the project
    run = wandb.init(project=config['wandb']['project'])

    # init dir for data
    base_dir = config['dirs']['init_path']
    session_name = config['dirs']['session_name']
    init_and_create_directory(
        path=base_dir, fixed_name=session_name, clear=True)

    # fetch data
    bucket_name = config['raw_data_s3']['bucket_name']
    file_name = config['raw_data_s3']['file_name']
    raw_data_dir = config['dirs']['raw_data_dir']
    fetch_data(bucket_name=bucket_name, file_name=file_name,
               raw_data_path=raw_data_dir, run=run)

    # get spark session config
    spark_master = config['spark']['master']
    spark_app_name = config['spark']['app_name']

    # preprocess data
    preprocessed_data = config['generated_data']['preprocessed']
    preprocess_data(raw_data_path=raw_data_dir, save_path=preprocessed_data,
                    spark_master=spark_master, spark_app_name=spark_app_name, run=run)

    # feature engineering
    feature_eng_data = config['generated_data']['features']
    feature_engineering(path_preprocessed_data=preprocessed_data, save_path=feature_eng_data,
                        spark_master=spark_master, spark_app_name=spark_app_name, run=run)

    # data pipeline
    path_proc_data_path = config['dirs']['session_dir']
    path_pipeline_model = config['generated_data']['pipeline_model']
    data_pipeline(feature_engineered_data_path=feature_eng_data, path_proc_data_path=path_proc_data_path,
                  path_pipeline_model=path_pipeline_model, spark_master=spark_master, spark_app_name=spark_app_name, run=run)

    # model experiment
    max_evals = config['model_experiment']['max_evals']
    seed = config['model_experiment']['seed']
    processed_data_dir = config['generated_data']['proc_data']
    model_experiment = init_model_experiment(
        config_file=config_file_path, max_evals=max_evals, seed=seed)
    run_model_experiment(model_experimenter=model_experiment,
                         proc_data=processed_data_dir)
    run_model_calibration(model_experimenter=model_experiment,
                          proc_data=processed_data_dir)

    # model explainability
    n_samples = config['model_explainability']['n_samples']
    model_explainability(session_dir=config['dirs']['session_dir'],n_samples=n_samples)

    # log artifacts to wandb
    artifacts_path = config['generated_data']['artifacts_dir']
    models_path = config['generated_data']['models_dir']
    log_wandb_artifact('model_explainability',
                       artifacts_path, models_path, run)

    # finish wandb run
    run.finish()


if __name__ == "__main__":
    sparkify_flow()
