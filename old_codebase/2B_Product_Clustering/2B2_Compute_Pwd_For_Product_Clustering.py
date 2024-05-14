# Databricks notebook source
# MAGIC %md
# MAGIC ### Compute pair wise distance for products
# MAGIC This notebook uses the custom dataset **2B1_Create_Data_For_Product_Clustering** to calculate the pairwise distance between each product pair.

# COMMAND ----------

# DBTITLE 1,Load Libraries
# utility and cluster are python scripts imported from Utility_fuction folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import cluster
import numpy as np
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration file
config = utils.read_config()

# Extract the data version from the configuration file
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file paths
# Path for product data based on the data version
product_data_path = f"DATA_VERSIONS/{data_version}/PRODUCT_DATA"
# Path for pairwise distance data based on the data version
pwd_data_path = f"DATA_VERSIONS/{data_version}/PRODUCT_DATA/PWD"

# COMMAND ----------

# DBTITLE 1,Load files
# Read the product data created for clustering
products = utils.read_parquet(f"{product_data_path}/product_details.parquet")

# COMMAND ----------

# DBTITLE 1,Choosing distance method for each feature
# Sepcifying the distance type for each feature

distance_type = {
    "product_type"          : "matching",
    "location"              : "matching",
    # "srp_3_desc"          : "jaccard", # matching
    "sales_list"            : "jaccard",
    "salesforce_list"       : "jaccard",
    "rental_list"           : "jaccard",
    "order_list"            : "jaccard",
    "avg_order_amount"      : "euclidean",
    "avg_sales_per_year"    : "euclidean",
    "opp_won_list"          : "jaccard",
    "opp_lost_list"         : "jaccard",
}

# COMMAND ----------

# DBTITLE 1,Saving the pairwise distances as a parquet file
for col,dist in distance_type.items():
    utils.save_parquet(pd.DataFrame(cluster.pairwise(products[col],dist),columns=[col]),f"{pwd_data_path}/{col}.parquet")