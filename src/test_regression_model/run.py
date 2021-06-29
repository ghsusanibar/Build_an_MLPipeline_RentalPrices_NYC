#!/usr/bin/env python
"""
Download the trained model, test data to calculate predictions and score
"""
import argparse
import itertools
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_model")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_dataset).file()
    df = pd.read_csv(test_data_path)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("price")

    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.mlflow_model).download()

    pipe = mlflow.sklearn.load_model(model_export_path)
    processed_features = list(itertools.chain.from_iterable([x[2] for x in pipe['preprocessor'].transformers]))
    logger.info("processed_features ---> " + str(processed_features))
    
    y_pred = pipe.predict(X_test[processed_features])
    
    logger.info("Scoring")
    r_squared = pipe.score(X_test[processed_features], y_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")
    # Log MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    ######################


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test the trained model")


    parser.add_argument(
        "--mlflow_model", 
        type=str,## INSERT TYPE HERE: str, float or int,
        help="Input MLFlow model",## INSERT DESCRIPTION HERE,
        required=True
    )

    parser.add_argument(
        "--test_dataset", 
        type=str,## INSERT TYPE HERE: str, float or int,
        help="Test dataset",## INSERT DESCRIPTION HERE,
        required=True
    )


    args = parser.parse_args()

    go(args)
