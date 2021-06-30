#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    """
    This function downloads a raw dataset and apply some basic data cleaning. Then the function uploads the result to a new artifact on W&B
    """

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)
    
    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    ### Changues to new version 1.0.1!!!
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    logger.info("Saving cleaned dataset")
    filename = args.output_artifact#"clean_sample.csv"
    df.to_csv(filename, index=False)
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":
    """
    This function downloads a raw dataset and apply some basic data cleaning. Then the function uploads the result to a new artifact on W&B.
    
    Parameters
    ----------
    input_artifact: str
        Fully qualified name for the artifact
    output_artifact: str
        Name for the output artifact
    output_type: str
        Type for the output artifact
    output_description: str
        Description for the output artifact
    min_price: float
        Minimum price to be considered
    max_price: float
        Maximum price to be considered
    
    Returns
    -------
    This function does not return a value, it uploads the cleaned dataset to Weights & Biases.
    
    Examples
    --------
    >>> mlflow.run(
                 <<file_path_basic_clieaning_directory>>,
                 <<entrypoint>>,
                 parameters={
                     "input_artifact": "sample.csv:latest",
                     "output_artifact": "clean_sample.csv",
                     "output_type": "clean_sample",
                     "output_description": "Data with outliers and null values removed",
                     "min_price": 50,
                     "max_price": 350
                 },
             )
    """

    parser = argparse.ArgumentParser(description="A very basic data cleaning", fromfile_prefix_chars="@",)

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to be considered",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to be considered",
        required=True
    )


    args = parser.parse_args()

    go(args)
