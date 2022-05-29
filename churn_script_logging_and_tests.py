"""
Testing module that will check the churn_library.py procedure.
Pylint is automatically launched by main procedure so simply launching this script will be enough.
Artifact produced will be in images, logs and models folders.

Author: Adam Barrett
Creation Date: 5/29/2022
"""
import os
import logging
import joblib
import pandas as pd
import churn_library

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(dataframe_raw):
    '''
    test perform eda function
    '''
    
    # To test that a dataframe is not empty
    assert dataframe_raw.shape[0] > 0
    assert dataframe_raw.shape[1] > 0
    
    churn_library.perform_eda(dataframe_raw)

    for image_name in [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
            "corr_heatmap"]:
        try:
            with open(f"images/eda/{image_name}.jpg", 'r'):
                logging.info(f"Testing perform_eda -- {image_name}: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform_eda: generated images missing")
            raise err


def test_encoder_helper(dataframe_encoded):
    '''
    test encoder helper
    '''
    try:
        assert dataframe_encoded.shape[0] > 0
        assert dataframe_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err
    try:

        dataframe_encoded = churn_library.encoder_helper(
            dataframe_encoded, [
                "Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"])

        for column in [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
                "Card_Category"]:
            assert column in dataframe_encoded

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")
    return dataframe_encoded


def test_perform_feature_engineering(df):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = churn_library.perform_feature_engineering(
            df)

        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_engineering: Sequences length mismatch")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(feature_sequences):
    """
    test train_models - check result of training process
    """
    churn_library.train_models(
        feature_sequences[0],
        feature_sequences[1],
        feature_sequences[2],
        feature_sequences[3])
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The files waeren't found")
        raise err
    for image_name in [
        "Logistic_Regression",
        "Random_Forest",
            "Feature_Importance"]:
        try:
            with open(f"images/results/{image_name}.jpg", 'r'):
                logging.info(
                    "Testing testing_models (report generation): SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing testing_models (report generation): generated images missing")
            raise err


def test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST):
    '''
    test train_models
    '''
    churn_library.train_models(
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
    try:
        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')
        logging.info("Testing testing_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: The files waeren't found")
        raise err
    for image_name in [
        "Logistic_Regression",
        "Random_Forest",
            "Feature_Importance"]:
        try:
            with open(f"images/results/{image_name}.jpg", 'r'):
                logging.info(
                    "Testing testing_models (report generation): SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing testing_models (report generation): generated images missing")
            raise err


if __name__ == "__main__":
    DATA_FRAME = test_import(churn_library.import_data)
    test_eda(DATA_FRAME)
    DATA_FRAME = test_encoder_helper(DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        DATA_FRAME)
    test_train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
