# Code verfied: OK! PASS!
# todo : review descriptions
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StandardScaler
from pyspark.sql.types import IntegerType
from .common import get_stratified_train_test_split, download_from_s3, remove_file, save_data, save_pd_parquet
import pandas as pd
import os
from typing import Tuple, List


class DataPipeline:
    """
    Class to create and run the data pipeline.

    Attributes:
        agg_features (DataFrame): spark dataframe containing aggregated features
        path (str): path to the project root
    """

    def __init__(self, agg_features: DataFrame, path: str):
        self.agg_features = agg_features
        self.localpath = path + '/proc_data/'
        self.modelpath = path + '/pipeline_model/'
        if not os.path.exists(self.localpath):
            os.makedirs(self.localpath)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the pipeline.
        Saves to parquet the transformed data and the pipeline model.

        Returns:
            X_train (pd.Dataframe): Train data
            y_train (pd.Dataframe): Train labels
            X_test (pd.Dataframe): Test data
            y_test (pd.Dataframe): Test labels        
        """
        train_data, test_data = get_stratified_train_test_split(
            self.agg_features, 'churn', 0.8, 42)
        # create pipeline
        pipeline = self.create_pipeline()
        # fit the pipeline
        pipeline_model = pipeline.fit(train_data)
        # transform the data
        train_data = pipeline_model.transform(train_data)
        test_data = pipeline_model.transform(test_data)
        # save the data
        save_data(train_data, self.localpath, 'train_data.parquet')
        save_data(test_data, self.localpath, 'test_data.parquet')

        # transform the data to pandas
        X_train, y_train, X_test, y_test = self.transform_to_pandas(
            train_data, test_data)

        # save X_train, y_train, X_test, y_test to parquet files
        save_pd_parquet(X_train, self.localpath, 'X_train.gzip')
        save_pd_parquet(y_train, self.localpath, 'y_train.gzip')
        save_pd_parquet(X_test, self.localpath, 'X_test.gzip')
        save_pd_parquet(y_test, self.localpath, 'y_test.gzip')

        # save the pipeline model
        pipeline_model.save(self.modelpath)
        return X_train, y_train, X_test, y_test

    def run_inference_pipeline(self, pipeline_model: PipelineModel) -> pd.DataFrame:
        """
        Run the inference pipeline on the given data.

        Args:
            data (DataFrame): Dataframe containing the data to be transformed
            pipeline_model (PipelineModel): Pipeline model to be used for transforming the data

        Returns:
            X : pandas dataframe containing the transformed data
        """
        # transform the data
        data = pipeline_model.transform(self.agg_features)
        # transform the data to pandas
        X = self.transform_to_pandas_inference(data)
        return X

    def create_pipeline(self) -> Pipeline:
        """ 
        Create a pipeline for the given model. 
        The pipeline performs the following steps:
        - Index categorical columns
        - One-hot encode categorical columns
        - Assemble the numerical and categorical features into a single vector
        - Scale the features

        Returns:
            pipeline (Pipeline): Pipeline containing the steps to be performed
        """

        feature_columns = [
            col for col in self.agg_features.columns if col not in ['userId', 'churn']]
        categorical_columns = ['gender', 'level', 'platform', 'browser']
        numerical_columns = [
            col for col in feature_columns if col not in categorical_columns]
        encoded_columns = [col + '_vec' for col in categorical_columns]

        indexers = [StringIndexer(inputCol=column, outputCol=column+"_index")
                    for column in categorical_columns]
        encoders = [OneHotEncoder(
            inputCol=column+"_index", outputCol=column+"_vec") for column in categorical_columns]
        num_assembler = VectorAssembler(
            inputCols=numerical_columns, outputCol="numerical_features")
        assembler = VectorAssembler(
            inputCols=["numerical_features"] + encoded_columns, outputCol="features")
        scaler = StandardScaler(
            inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        pipeline = Pipeline(stages=indexers + encoders +
                            [num_assembler, assembler, scaler])
        return pipeline

    def get_feature_names(self) -> List[str]:
        """
        Generates list of feature names used to train the model.

        Returns:
            feature_names (List[str]): List of feature names used to train the model. 
        """
        feature_names = ['num_sessions', 'num_unique_artists', 'num_nextsong', 'num_thumbs_up',
                         'num_thumbs_down', 'num_add_to_playlist', 'num_add_friend', 'num_error',
                         'num_help', 'num_submit_upgrade', 'num_submit_downgrade',
                         'num_roll_advert', 'num_logout', 'num_downgrade', 'num_upgrade',
                         'num_settings', 'num_about', 'num_save_settings', 'tenure',
                         'num_ads_per_session', 'avg_unique_artists_per_day',
                         'avg_songs_per_session', 'avg_time_between_sessions', 'prop_thumbs_up',
                         'prop_add_to_playlist', 'prop_thumbs_down', 'unique_artist_ratio',
                         'avg_listening_time_per_session', 'avg_artists_per_session',
                         'stddev_songs_per_session', 'stddev_time_between_sessions', 'gender_vec', 'level_vec', 'platform_vec_0', 'platform_vec_1', 'platform_vec_2', 'browser_vec_0', 'browser_vec_1', 'browser_vec_2']
        return feature_names

    def transform_to_pandas_inference(self, data: DataFrame) -> pd.DataFrame:
        """
        Converts spark dataframe to pandas dataframe for inference pipeline.

        Args:
            data (DataFrame): Spark dataframe to be converted

        Returns:
            data_features_df (pd.DataFrame): Pandas dataframe containing the transformed data
        """
        # Convert the Spark DataFrame to a Pandas DataFrame
        data_pd = data.select('scaled_features', 'userId').toPandas()

        # Get the feature names
        feature_names = self.get_feature_names()

        # Transform the 'scaled_features' column from a vector into a list
        data_pd['scaled_features'] = data_pd['scaled_features'].apply(
            lambda x: x.toArray().tolist())

        # Convert PySpark DataFrame to Pandas DataFrame
        data_features_df = pd.DataFrame(
            data_pd['scaled_features'].to_list(), columns=feature_names)

        # Add the userId column
        data_features_df['userId'] = data_pd['userId']
        return data_features_df

    def transform_to_pandas(self, train_tx: DataFrame, test_tx: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Converts spark dataframes to pandas dataframes for training and testing.

        Args:
            train_tx (DataFrame): Spark dataframe containing training data to be converted
            test_tx (DataFrame): Spark dataframe containin test data to be converted

        Returns:
            X_train (pd.DataFrame): Pandas dataframe containing the transformed training data
            y_train (pd.DataFrame): Pandas dataframe containing the training labels
            X_test (pd.DataFrame): Pandas dataframe containing the transformed test data
            y_test (pd.DataFrame): Pandas dataframe containing the test labels  

        """
        # Convert the Spark DataFrame to a Pandas DataFrame
        train_pd = train_tx.select('scaled_features', 'churn').toPandas()
        test_pd = test_tx.select('scaled_features', 'churn').toPandas()

        # Get the feature names
        feature_names = self.get_feature_names()

        # Transform the 'scaled_features' column from a vector into a list
        train_pd['scaled_features'] = train_pd['scaled_features'].apply(
            lambda x: x.toArray().tolist())
        test_pd['scaled_features'] = test_pd['scaled_features'].apply(
            lambda x: x.toArray().tolist())

        # Convert PySpark DataFrame to Pandas DataFrame
        train_features_df = pd.DataFrame(
            train_pd['scaled_features'].to_list(), columns=feature_names)
        test_features_df = pd.DataFrame(
            test_pd['scaled_features'].to_list(), columns=feature_names)

        # Separate features and target variable
        X_train = train_features_df
        y_train = train_pd['churn']
        X_test = test_features_df
        y_test = test_pd['churn']

        # convert y_train and y_test to pandas DataFrame
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)

        return X_train, y_train, X_test, y_test
