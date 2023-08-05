import httpagentparser

from pyspark.sql.functions import udf, when, col, max
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.sql.window import Window
from typing import Tuple, List, Dict


class Preprocessing:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def parse_agent(userAgent: str) -> Tuple[str, str]:
        """
        Get the platform (e.g. Windows, Mac) and the browser (e.g. Chrome, Firefox) from the user agent string

        Args:
            userAgent (str): User agent string

        Returns:
            platform (str): Platform
            browser (str): Browser
        """
        try:
            fields = httpagentparser.detect(userAgent)
            platform = fields['platform']['name']
            browser = fields['browser']['name']
            return platform, browser
        except Exception as e:
            print(f"Error parsing user agent: {e}")

    def drop_na_rows(self):
        """
        Given that we are dealing with user data, we drop rows with missing userId.
        In fact without user information we cannot identify the sessions allocated to a specific user.
        Therefore, we only keep entries with a valid userId.
        """

        self.df = self.df.dropna(subset=['userId', 'firstName', 'lastName',
                                 'location', 'userAgent', 'registration', 'gender'], how='any')
        return self

    def add_listening_event(self):
        """
        Add a column for listening events.
        This is motivated by the fact that we have entries where the columns song and artist are missing among others.
        This is because the user is not listening to music but is doing something else on the platform.
        So instead of dropping these entries, which could be crucial for the prediction, we add a column to indicate if the user is listening to music or not.
        """
        self.df = self.df.withColumn('Listening', when(
            col('song').isNotNull(), 1).otherwise(0))
        return self

    def define_churn(self):
        """
        We define churn as a our target label.
        We choose to define chrun as a user visiting the page "Cancellation Confirmation".
        """

        churn_udf = udf(lambda x: 1 if x ==
                        "Cancellation Confirmation" else 0, IntegerType())
        self.df = self.df.withColumn('churn_f', churn_udf('page'))
        user_window = Window.partitionBy('userId').rangeBetween(
            Window.unboundedPreceding, Window.unboundedFollowing)
        self.df = self.df.withColumn('churn', max('churn_f').over(user_window))
        self.df = self.df.drop('churn_f')
        return self

    def parse_user_agent(self):
        """
        Parse the user agent string to get the platform (e.g. Windows, Mac) and the browser (e.g. Chrome, Firefox)
        We use the httpagentparser library exploited in the function parse_agent
        """
        parse_agent_udf = udf(self.parse_agent, StructType(
            [StructField('platform', StringType()), StructField('browser', StringType())]))
        self.df = self.df.withColumn(
            'Agent_extact', parse_agent_udf(self.df['userAgent']))
        self.df = self.df.withColumn(
            'platform', self.df['Agent_extact']['platform'])
        self.df = self.df.withColumn(
            'browser', self.df['Agent_extact']['browser'])
        self.df = self.df.drop('Agent_extact')
        return self

    def preprocess_data(self):
        """
        Pipeline to preprocess the data.
        It is composed of the following steps:
        - drop rows with missing userId
        - add a column for listening events
        - define churn as a label
        - parse the user agent

        Returns:
            df (DataFrame): Dataframe containing the preprocessed data
        """
        self.drop_na_rows()
        self.add_listening_event()
        self.define_churn()
        self.parse_user_agent()

        return self.df
