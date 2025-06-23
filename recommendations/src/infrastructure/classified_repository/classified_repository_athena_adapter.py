import time
import boto3
import pandas as pd
from logging import Logger

from recommendations.src.domain.interfaces.classified_repository_interface import (
    ClassifiedRepositoryInterface,
)


class ClassifiedRepositoryAthenaAdapter(ClassifiedRepositoryInterface):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.datalake_name = "classified_training_data"
        self.table_name = "classified_training_data"
        self.athena_client = boto3.client("athena")

    def get_classified_training_data(self, online_date: str) -> pd.DataFrame:
        self.logger.info(f"Getting classified training data for {online_date}, from {self.datalake_name}.{self.table_name}")

        query = f"""
        SELECT * FROM {self.datalake_name}.{self.table_name}
        WHERE online_date = '{online_date}'
        """
        response = self.athena_client.start_query_execution(
            QueryString=query,
        )

        response = self.athena_client.get_query_results(QueryExecutionId=response["QueryExecutionId"])

        while response["QueryExecutionStatus"]["State"] == "RUNNING":
            time.sleep(5)
            response = self.athena_client.get_query_results(QueryExecutionId=response["QueryExecutionId"])

        if response["QueryExecutionStatus"]["State"] == "FAILED":
            self.logger.error(f"Query execution failed: {response['QueryExecutionStatus']['StateChangeReason']}")
            raise Exception(f"Query execution failed: {response['QueryExecutionStatus']['StateChangeReason']}")

        return pd.DataFrame(response["ResultSet"]["Rows"])
