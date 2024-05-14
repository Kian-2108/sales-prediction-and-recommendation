#load libraries
import pandas as pd
import numpy as np
import os
import json
import pickle
import dill
from azure.storage.blob import BlobServiceClient
from pyspark.dbutils import DBUtils
dbutils = DBUtils()

def read_parquet(file_path,root="VIRTUSA"):
    """
    Reads a parquet file from the specified file path.

    Args:
        file_path: The path to the parquet file.
        root: The root directory (optional). Defaults to "VIRTUSA".

    Returns:
        The loaded parquet file as a pandas DataFrame.
    """
    if root==None:
        full_path = os.path.join(os.getenv("ADLS_directory"),file_path).replace("abfss://","abfs://")
    else:
        full_path = os.path.join(os.getenv("ADLS_directory"),"VIRTUSA",file_path).replace("abfss://","abfs://")
    storage_options={"account_key": os.getenv("ACCOUNT_KEY")}
    return pd.read_parquet(full_path,storage_options=storage_options)

def read_csv(file_path,root="VIRTUSA"):
    """
    Reads a CSV file from the specified file path.

    Args:
        file_path: The path to the CSV file.
        root: The root directory (optional). Defaults to "VIRTUSA".

    Returns:
        The loaded CSV file as a pandas DataFrame.
    """
    if root==None:
        full_path = os.path.join(os.getenv("ADLS_directory"),file_path).replace("abfss://","abfs://")
    else:
        full_path = os.path.join(os.getenv("ADLS_directory"),"VIRTUSA",file_path).replace("abfss://","abfs://")
    storage_options={"account_key": os.getenv("ACCOUNT_KEY")}
    return pd.read_csv(full_path,storage_options=storage_options,low_memory=False)

def save_parquet(df,file_path):
    """
    Saves a DataFrame as a parquet file to the specified file path.

    Args:
        df: The DataFrame to be saved.
        file_path: The path to save the parquet file.

    Returns:
        None
    """
    full_path = os.path.join(os.getenv("ADLS_directory"),"VIRTUSA",file_path)
    storage_options={"account_key": os.getenv("ACCOUNT_KEY")}
    df.to_parquet(full_path,storage_options=storage_options)

def read_config(file_path="/Workspace/Users/davide@baxter.com/Solution/config.json"):
    """
    Reads a JSON configuration file from the specified file path.

    Args:
        file_path: The path to the JSON configuration file (optional).
                   Defaults to "/Workspace/Users/davide@baxter.com/Solution/config.json".

    Returns:
        The loaded JSON data as a Python dictionary.
    """
    with open(file_path,"r") as f:
        return json.load(f)
    
def save_stats(stats):
    """
    Saves statistics data to a JSON file.

    Args:
        stats: The statistics data to be saved.

    Returns:
        None
    """

    stats_path = "/Workspace/Users/davide@baxter.com/Solution/stats.json"
    try:
        with open(stats_path,"r") as f:
            old_stats = json.load(f)
        with open(stats_path,"w") as f:
            json.dump({**old_stats, **stats},f,indent=3)
    except:
        with open(stats_path,"w") as f:
            json.dump(stats,f,indent=3)

def save_json(save_path,file_path):
    """
    Log the stats file on azure storage

    Args:
        file_path: The path to the stats.json file
    
    Return:
        None
    """
    account_key = os.getenv("ACCOUNT_KEY")
    account_name = "hrcengagedpocstorageml"
    container_name  = "rawdataupload"
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_blob_client(container_name,os.path.join("VIRTUSA",save_path))

    with open(file_path,"rb") as blob:
        container_client.upload_blob(blob,overwrite=True)

    
def save_pkl(model,file_path):
    """
    Saves a Python model as a pickle file to the specified file path.

    Args:
        model: The Python model object to be saved.
        file_path: The path to save the pickle file.

    Returns:
        None
    """
    account_key = os.getenv("ACCOUNT_KEY")
    account_name = "hrcengagedpocstorageml"
    container_name  = "rawdataupload"
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_blob_client(container_name,os.path.join("VIRTUSA",file_path))

    with open("model.pkl", "wb") as pkl:
        dill.dump(model, pkl)
    with open("model.pkl","rb") as blob:
        container_client.upload_blob(blob,overwrite=True)
    dbutils.fs.rm("file:/databricks/driver/model.pkl")

def read_pkl(file_path):
    """
    Reads a pickle file from the specified file path and returns the loaded data.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The loaded data from the pickle file.
    """
    account_key = os.getenv("ACCOUNT_KEY")
    account_name = "hrcengagedpocstorageml"
    container_name  = "rawdataupload"
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_blob_client(container_name,os.path.join("VIRTUSA",file_path))

    with open("model.pkl", "wb") as pkl:
        pkl.write(container_client.download_blob().readall())
    with open("model.pkl","rb") as blob:
        data = dill.load(blob)
    dbutils.fs.rm("file:/databricks/driver/model.pkl")
    return data

def customer_product_stats(stats, data, name):
    """
    Computes statistics related to customer and product data and updates the stats dictionary.

    Args:
        stats: The dictionary to store the statistics.
        data: The customer or product data.
        name: The name of the data (e.g., "customer", "product").

    Returns:
        None
    """

    stats[f"{name} shape"] = {
            "value": data.shape,
            "description": f"Size of {name} data",
        }
    stats[f"{name} features"] = {
            "value": list(data.columns),
            "description": f"Features of {name} data",
    }
    if "customer_num" in data.columns:
        stats[f"{name} customer count"] = {
            "value": len(data[["customer_num"]].drop_duplicates()),
            "description": f"No. of unique customers in {name} data",
        }
    if "srp_2_desc" in data.columns:
        stats[f"{name} product count"] = {
            "value": len(data[["srp_2_desc"]].drop_duplicates()),
            "description": f"No. of unique products in {name} data",
        }
    if ("customer_num" in data.columns) and ("srp_2_desc" in data.columns):
        stats[f"{name} pair count"] = {
            "value": len(data[["customer_num", "srp_2_desc"]].drop_duplicates()),
            "description": f"No. of unique customer-product pairs in {name} data",
        }

def read_filenames(file_path):
    ADLS_directory = f"abfss://rawdataupload@hrcengagedpocstorageml.dfs.core.windows.net/VIRTUSA/{file_path}"
    all_files = dbutils.fs.ls(ADLS_directory)
    file_names = []
    for f in all_files:
        file_names.append(os.path.splitext(os.path.split(f.path)[-1])[0])
    return file_names

def read_matrix(file_path):
    """
    Reads a matrix stored as Parquet files from the specified file path and returns it as a dictionary.

    Args:
        file_path: The path to the matrix files.

    Returns:
        A dictionary containing the matrix data, where the keys are the matrix IDs and the values are the matrix data.
    """
    matrix_names = read_filenames(file_path)
    cf_matrix = dict()
    for f in matrix_names:
        cf_matrix[int(f)] = read_parquet(f"{file_path}/{f}.parquet")
    return cf_matrix