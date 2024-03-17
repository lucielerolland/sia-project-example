import pandas as pd
import os
import zipfile


def load_data_from_kaggle(dataset_producer_name: str, dataset_name: str):
    """
    Loads some CSVs from a kaggle dataset. We use the kaggle package for this
    More info on the kaggle API : https://www.kaggle.com/docs/api

    :param dataset_producer_name: str
        name of the person who hosts the dataset
    :param dataset_name: str
        name of the dataset
    :return: None
    """
    download_command = f'kaggle datasets download "{dataset_producer_name}/{dataset_name}"'
    os.system(download_command)
    with zipfile.ZipFile(f"{dataset_name}.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    os.remove(f"{dataset_name}.zip")


def process_loaded_data_from_kaggle(file_name: str) -> pd.DataFrame:
    # Here you would do some processing (reading and merging dataframes, choosing relevant
    # columns, dealing with missing values and so on), but I'm not the one getting a mark
    # so I'm just reading an already-cleaned data frame and dropping the race_ethnicity column
    # as ethnic statistics are forbidden by French law.
    return pd.read_csv(f"data/{file_name}").drop(columns=['race_ethnicity'])


def get_student_data(
        dataset_producer_name: str, dataset_name: str, file_name: str
        ) -> pd.DataFrame:
    load_data_from_kaggle(dataset_producer_name, dataset_name)

    return process_loaded_data_from_kaggle(file_name)

