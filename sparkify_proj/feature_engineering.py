# Code verfied: OK! PASS!
# todo : review descriptions
# Description: This script contains the feature engineering steps for the Sparkify churn prediction model.
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, when, col, max, min, count, countDistinct, avg, coalesce, first, lag, sum
# from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql.window import Window
import pyspark.sql.functions as F


# Define constants
PAGES_TO_COUNT = ["NextSong", "Thumbs Up", "Thumbs Down", "Add to Playlist", "Add Friend", "Error", "Help", "Submit Upgrade",
                  "Submit Downgrade", "Roll Advert", "Logout", "Downgrade", "Upgrade", "Settings", "About", "Save Settings"]


class FeatureEngineering:
    """
    Feature engineering class.
    Provides feature engineering steps for the Sparkify churn prediction.

    Attributes:
        df (pd.Dataframe): Dataframe containg preprocessed sparkify data.

    """

    def __init__(self, df: DataFrame):
        self.df = df

    def feature_engineering(self) -> DataFrame:
        """
        Perform feature engineering to create features for the model training.
        We aggregate the data by userId and create relevant features given we are predicting churn.

        Returns:
            df (DataFrame): Dataframe containing the engineered features

        """
        agg_features = self.aggregate_features()
        agg_features = self.calculate_avg_songs_per_session(agg_features)
        agg_features = self.calculate_time_between_sessions(agg_features)
        agg_features = self.calculate_action_proportions(agg_features)
        agg_features = self.calculate_listening_time_per_session(agg_features)
        agg_features = self.calculate_artists_per_session(agg_features)
        agg_features = self.calculate_stddev_songs_per_session(agg_features)
        agg_features = self.calculate_stddev_time_between_sessions(
            agg_features)

        return agg_features

    def aggregate_features(self) -> DataFrame:
        """
        Aggregate the data by userId.
        The aggregation is done the page types that the user interacted with.
        Here are the features that we create:
        - gender
        - level
        - platform
        - browser
        - churn
        - num (page): number of times the user visited the page (e.g. numNextSong, numThumbsUp, etc.)
        - tenure: number of days since registration 
        - num_ads_per_session: number of ads per session
        - avg_unique_artists_per_day: average number of unique artists per day
        - num_sessions: number of sessions
        - num_unique_artists: number of unique artists

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features
        """
        agg_dict = {
            "gender": first("gender"),
            "level": first("level"),
            "platform": first("platform"),
            "browser": first("browser"),
            "churn": first("churn"),
            "num_sessions": countDistinct("sessionId"),
            "num_unique_artists": countDistinct("artist")
        }
        # Add page counts to the dictionary using list comprehension
        agg_dict.update({f"num_{page.replace(' ', '_').lower()}": count(
            when(col("page") == page, 1)) for page in PAGES_TO_COUNT})

        # Update the dictionary for aggregation with new features
        agg_dict.update({
            "tenure": (F.max(col("ts") / 1000) - first(col("registration") / 1000)) / (60 * 60 * 24),
            "num_ads_per_session": count(when(col("page") == "Roll Advert", 1)) / countDistinct("sessionId"),
            "avg_unique_artists_per_day": countDistinct("artist") / ((max("ts") - first("registration")) / (1000 * 60 * 60 * 24) + 1e-6),
        })

        # Aggregating features
        agg_features = self.df.groupBy("userId").agg(
            *agg_dict.values()).toDF("userId", *agg_dict.keys())
        return agg_features

    def calculate_avg_songs_per_session(self, agg_features: DataFrame) -> DataFrame:
        """
        Calculate the average number of songs per session for each user.
        This feature is created to capture the user's listening behavior also we want to gouge his/her degree of engagement with the platform.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with avg_songs_per_session added
        """

        songs_per_session = self.df.filter(col("page") == "NextSong").groupBy(
            "userId", "sessionId").count().groupBy("userId").agg(avg("count").alias("avg_songs_per_session"))
        agg_features = agg_features.join(songs_per_session, on="userId")
        return agg_features

    def calculate_time_between_sessions(self, agg_features: DataFrame) -> DataFrame:
        """
        Calculate the average time between sessions for each user.
        We expect that users who are more engaged with the platform will have shorter time between sessions.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with avg_time_between_sessions added
        """

        time_between_sessions = self.df.groupBy("userId", "sessionId") \
            .agg((min("ts")/1000).alias("session_start"), first("registration").alias("registration_ts")) \
            .withColumn("prev_session_start", lag("session_start").over(Window.partitionBy("userId").orderBy("session_start"))) \
            .withColumn("prev_session_start", coalesce(col("prev_session_start"), col("registration_ts")/1000)) \
            .withColumn("time_between_sessions", (col("session_start") - col("prev_session_start"))/3600) \
            .groupBy("userId") \
            .agg(avg("time_between_sessions").alias("avg_time_between_sessions"))

        agg_features = agg_features.join(time_between_sessions, on="userId")
        return agg_features

    def calculate_action_proportions(self, agg_features: DataFrame) -> DataFrame:
        """
        We exploit the previous aggregation of the page types to calculate the proportion of each page type for each user.
        We create the following features:
        - prop_thumbs_up: proportion of thumbs up
        - prop_add_to_playlist: proportion of add to playlist
        - prop_thumbs_down: proportion of thumbs down
        - unique_artist_ratio: ratio of unique artists to total number of songs

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with the new features added

        """

        agg_features = agg_features.withColumn(
            "prop_thumbs_up", col("num_thumbs_up") / col("num_nextsong"))
        agg_features = agg_features.withColumn(
            "prop_add_to_playlist", col("num_add_to_playlist") / col("num_nextsong"))
        agg_features = agg_features.withColumn(
            "prop_thumbs_down", col("num_thumbs_down") / col("num_nextsong"))
        agg_features = agg_features.withColumn(
            "unique_artist_ratio", col("num_unique_artists") / col("num_nextsong"))
        return agg_features

    def calculate_listening_time_per_session(self, agg_features: DataFrame) -> DataFrame:
        """
        Calculate the average listening time per session for each user.
        We want to use listening time per session as a proxy for user engagement.
        We expect that users who are more engaged with the platform will have longer listening time per session.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with avg_listening_time_per_session added
        """

        listening_time_per_session = self.df.filter(col("page") == "NextSong").groupBy("userId", "sessionId")\
            .agg(sum("length").alias("session_listening_time"))\
            .groupBy("userId")\
            .agg(avg("session_listening_time").alias("avg_listening_time_per_session"))
        agg_features = agg_features.join(
            listening_time_per_session, on="userId")
        return agg_features

    def calculate_artists_per_session(self, agg_features: DataFrame) -> DataFrame:
        """
        Calculate the average number of artists per session for each user.
        A high number of different artistis per session could indicated that either the user is exploring the platform or that he/she is not satisfied with the current artist.
        Maybe the user is using the service to listen to compilations or playlists, which would lead to a high number of artists per session.
        We expect that users who are more engaged with the platform will have a lower number of artists per session.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with avg_artists_per_session added

        """

        artists_per_session = self.df.filter(col("page") == "NextSong").groupBy("userId", "sessionId").agg(countDistinct(
            "artist").alias("artists_in_session")).groupBy("userId").agg(avg("artists_in_session").alias("avg_artists_per_session"))
        agg_features = agg_features.join(artists_per_session, on="userId")
        return agg_features

    def calculate_stddev_songs_per_session(self, agg_features: DataFrame) -> DataFrame:
        """
        Cpmputes the standard deviation of the number of songs per session for each user.
        Which means that a higher stdev in songs per session indicates that the user may have sessions where he/she explores different artists and genres and other where he exploits.
        If there is too much exploration, this could indicated that he/she is not able to find the music he/she likes.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with stddev_songs_per_session added
        """

        stddev_songs_per_session = self.df.filter(col("page") == "NextSong").groupBy("userId", "sessionId").count(
        ).groupBy("userId").agg(F.stddev("count").alias("stddev_songs_per_session")).fillna(0)
        agg_features = agg_features.join(stddev_songs_per_session, on="userId")
        return agg_features

    def calculate_stddev_time_between_sessions(self, agg_features: DataFrame) -> DataFrame:
        """
        Computes the standard deviation of the time between sessions for each user.
        A high standard deviation in time between sessions could indicate that the user is not using the service regularly.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            agg_features (DataFrame): Dataframe containing the aggregated features with stddev_time_between_sessions added
        """
        stddev_time_between_sessions = self.df.groupBy("userId", "sessionId") \
            .agg((min("ts")/1000).alias("session_start"), first("registration").alias("registration_ts")) \
            .withColumn("prev_session_start", lag("session_start").over(Window.partitionBy("userId").orderBy("session_start"))) \
            .withColumn("prev_session_start", coalesce(col("prev_session_start"), col("registration_ts")/1000)) \
            .withColumn("time_between_sessions", (col("session_start") - col("prev_session_start"))/3600) \
            .groupBy("userId") \
            .agg(F.stddev("time_between_sessions").alias("stddev_time_between_sessions")).fillna(0)
        agg_features = agg_features.join(
            stddev_time_between_sessions, on="userId")
        return agg_features

    def get_feature_columns(self, agg_features: DataFrame) -> list:
        """
        Get the list of feature columns.

        Args:
            agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            feature_columns (list): List of feature columns
        """
        return [col for col in agg_features.columns if col not in ['userId', 'churn']]

    def get_categorical_columns(self) -> list:
        """
        Get the list of categorical columns.

        Returns:
            list : List of categorical columns to be encoded.
        """
        return ['gender', 'level', 'platform', 'browser']

    def get_numerical_columns(self, agg_features: DataFrame) -> list:
        """
        Get the list of numerical columns.
        This will be used during numerical feature scaling.

        Args:
             agg_features (DataFrame): Dataframe containing the aggregated features

        Returns:
            list : List of numerical columns        
        """
        return [col for col in agg_features.columns if col not in self.get_categorical_columns()]
