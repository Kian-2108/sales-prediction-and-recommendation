#load libraries
import pandas as pd
import numpy as np
import os
import json
import pickle
import dill

def read_table(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".parquet":
        read_func = pd.read_parquet
    elif ext == ".csv":
        read_func = pd.read_csv
    return read_func(file_path)
    
    
    
def read_json(file_path):
    with open(file_path,"r") as f:
        return json.load(f)
    
def save_stats(stats):
    stats_path = "./stats.json"
    try:
        with open(stats_path,"r") as f:
            old_stats = json.load(f)
        with open(stats_path,"w") as f:
            json.dump({**old_stats, **stats},f,indent=3)
    except:
        with open(stats_path,"w") as f:
            json.dump(stats,f,indent=3)

def save_json(save_path,file_path):
    account_key = os.getenv("ACCOUNT_KEY")
    account_name = "hrcengagedpocstorageml"
    container_name  = "rawdataupload"
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_blob_client(container_name,os.path.join("VIRTUSA",save_path))

    with open(file_path,"rb") as blob:
        container_client.upload_blob(blob,overwrite=True)

    
def save_pkl(model,file_path):
    with open(file_path, "wb") as pkl:
        dill.dump(model, pkl)

def read_pkl(file_path):
    with open(file_path, "rb") as pkl:
        data = dill.load(pkl)
    return data

def customer_product_stats(stats, data, name):
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
    all_files = os.listdir(file_path)
    file_names = []
    for f in all_files:
        file_names.append(os.path.splitext(f)[0])
    return file_names

def read_matrix(file_path):
    matrix_names = read_filenames(file_path)
    cf_matrix = dict()
    for f in matrix_names:
        cf_matrix[int(f)] = pd.read_parquet(f"{file_path}/{f}.parquet")
    return cf_matrix