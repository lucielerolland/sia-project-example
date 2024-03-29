import pandas as pd
import os
import zipfile
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


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


def process_loaded_data_from_kaggle(file_name: str, categorical_variable_list: list) -> Tuple[pd.DataFrame, list]:
    """
    Reads and processes the kaggle data: drops the race_ethnicity column as ethnic statistics
    are forbidden by French law and prepares my one-hot-encoded variables for regression.
    In your case, this kind of function would do some extra processing (reading and
    merging dataframes, choosing relevant columns, dealing with missing values and so on)
    :param file_name: str
        path to the file containing the data
    :param categorical_variable_list: list
        list of variables that should be one-hot encoded
    :return: Tuple[pd.DataFrame, list]
        a dataframe containing the processed data and a list containing the names of the one-hot encoded
        variables
    """
    data = pd.read_csv(f"data/{file_name}").drop(columns=['race_ethnicity'])
    encoded_data, encoded_variables_names = turn_categorical_variables_to_oh(data, categorical_variable_list)

    for regressor in categorical_variable_list:
        reference_variable = [
            value for value in data[regressor].unique()
            if f'{regressor}_{value}' not in encoded_data.columns
        ][0]
        print(f"Reference value for categorical regressor {regressor}: {reference_variable}")

    return data.join(encoded_data), encoded_variables_names


def get_student_data(
        dataset_producer_name: str, dataset_name: str, file_name: str, regressors: list
        ) -> Tuple[pd.DataFrame, list]:
    load_data_from_kaggle(dataset_producer_name, dataset_name)
    loaded_data, oh_encoded_variable_names = process_loaded_data_from_kaggle(file_name, regressors)

    return loaded_data, oh_encoded_variable_names


def turn_categorical_variables_to_oh(
        data: pd.DataFrame, categorical_variables: list
    ) -> Tuple[pd.DataFrame, list]:
    # Drop the first one-hot encoded variable to avoid multicolinearity
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = one_hot_encoder.set_output(transform="pandas").fit_transform(data[categorical_variables])

    one_hot_encoder_names = encoded_data.columns

    return encoded_data, one_hot_encoder_names


def random_function_for_show(x1,x2):
    """
    This is
    :param x1:
    :param x2:
    :return:
    """

