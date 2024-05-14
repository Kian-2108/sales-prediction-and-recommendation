# Databricks notebook source
# MAGIC %md
# MAGIC ### Compute pair wise distance for customers
# MAGIC This notebook uses the custom dataset **2A1_Create_Data_For_Customer_Clustering** to calculate the pairwise distance between each customer pair.

# COMMAND ----------

# DBTITLE 1,Load Libraries
# utility and cluster are python scripts imported from Utility_fuction folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import cluster
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration file
config = utils.read_config()

# Extract the data version from the configuration file
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file paths
# Path for customer data based on the data version
customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"
# Path for pairwise distance data based on the data version
pwd_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA/PWD"

# COMMAND ----------

# DBTITLE 1,Load files
# Read the custoemer data created for clustering
customers = utils.read_parquet(f"{customer_data_path}/customer_details.parquet")

# COMMAND ----------

# DBTITLE 1,Choosing distance method for each feature
# Sepcifying the distance type for each feature

distance_type = {
    "active_cust_yn"            : "matching",
    "customer_class"            : "matching",
    "customer_price_group"      : "matching",
    "customer_type"             : "matching",
    # "gl_acct_classification"    : "matching",
    "No. of Discharges"         : "euclidean",
    "No. of Staffed Beds"       : "euclidean",
    "340B Classification"       : "matching",
    "sales_list"                : "jaccard",
    "salesforce_list"           : "jaccard",
    "rental_list"               : "jaccard",
    "purchase_per_year"         : "euclidean",
    # "Latitude"                  : "euclidean",
    # "Longitude"                 : "euclidean",
    "Hospital Type"             : "matching",
    # "IDN"                       : "matching",
    # "IDN Parent"                : "matching",
    "Bed Utilization Rate"      : "euclidean",
    "Capital Expenditures"      : "euclidean",
    "Net Income"                : "euclidean",
}

# COMMAND ----------

# DBTITLE 1,Saving the pairwise distances as a parquet file
for col,dist in distance_type.items():
    utils.save_parquet(pd.DataFrame(cluster.pairwise(customers[col],dist),columns=[col]),f"{pwd_data_path}/{col}.parquet")