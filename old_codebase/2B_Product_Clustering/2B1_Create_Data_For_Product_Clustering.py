# Databricks notebook source
# MAGIC %md 
# MAGIC ### Create Custom Data for Product Clustering
# MAGIC This notebook creates a custom dataset by using Capital Sales, Salesforce, Rental, and Curated Product data . The dataset involves creating new engineered features. 

# COMMAND ----------

# DBTITLE 1,Load Libraries
# utils is python scripts imported from the 0A_Utility_Functions folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration settings from a configuration file
config = utils.read_config()
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file path
#path of data before cutoff based on the data version
raw_data_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"

#path of product cluster info based on the data version
product_data_path = f"DATA_VERSIONS/{data_version}/PRODUCT_DATA"

# COMMAND ----------

# DBTITLE 1,Load files
#load Capital Sales data 
CSD = utils.read_parquet(f"{raw_data_path}/customer_sales_data.parquet")

#load salesforce data 
SFD = utils.read_parquet(f"{raw_data_path}/sales_force_data.parquet")

#load Rental Sales data 
RSD = utils.read_parquet(f"{raw_data_path}/rental_sales_data.parquet")

#load Customer Baseline data 
CBD = utils.read_parquet(f"{raw_data_path}/customer_baseline_data.parquet")

#load Customer cross walk data 
CWD = utils.read_parquet(f"{raw_data_path}/customer_crosswalk_data.parquet")

#load External Definitve Data data 
EDD = utils.read_parquet(f"{raw_data_path}/external_definitive_data.parquet")

#load curated product data 
CPL = utils.read_parquet(f"{raw_data_path}/curated_products_data.parquet")

#load unfiletered products data
UPL = utils.read_parquet(f"{raw_data_path}/unfiltered_products_data.parquet")

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Selecting attributes of customer used for customer clustering
# Selecting features present in curated product data like location and replacement of inactive products
products = CPL[
    [
        "srp_2_code",
        "srp_2_desc",
        "replacement_srp_2_code",
        "updated",
        "product_type",
        "location",
    ]
]

# Renaming columns of added features
products.rename(
    columns={
        # "srp_2_code": "old_srp_2_code",
        # "srp_2_desc": "old_srp_2_desc",
        "replacement_srp_2_code": "new_srp_2_code",
        "updated": "new_srp_2_desc",
    },
    inplace=True,
)

# Selecting features from the unfiltered product data like srp 1 and srp 3 level description
products = products.merge(UPL[["srp_2_code","srp_2_desc","srp_3_desc"]].groupby(["srp_2_code","srp_2_desc"]).first().reset_index(),how="left",on=["srp_2_code","srp_2_desc"])

# Adding the list of customers which purchased the product in customer sales table as a feature
temp_prod1 = (
    CSD[["srp_2_code","srp_2_desc", "customer_num"]]
    .groupby(["srp_2_code","srp_2_desc"])["customer_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod1, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"customer_num": "sales_list"}, inplace=True)

# Adding the list of customers which purchased the product in salesforce table as a feature
temp_prod2 = (
    SFD[["srp_2_desc", "customer_num"]]
    .groupby("srp_2_desc")["customer_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod2, on="srp_2_desc", how="left")
products.rename(columns={"customer_num": "salesforce_list"}, inplace=True)

# Adding the list of customers which rented the product in rental sales table as a feature
temp_prod3 = (
    RSD[["srp_2_code","srp_2_desc", "customer_num"]]
    .groupby(["srp_2_code","srp_2_desc"])["customer_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod3, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"customer_num": "rental_list"}, inplace=True)

# Adding the list of orders which contained the product in customer sales table as a feature
temp_prod4 = (
    CSD[["srp_2_code","srp_2_desc", "dtl_order_num"]]
    .groupby(["srp_2_code","srp_2_desc"])["dtl_order_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod4, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"dtl_order_num": "order_list"}, inplace=True)

# Adding the average order amount of the product in customer sales table as a feature
temp_prod6 = (
    CSD[["srp_2_code","srp_2_desc", "dtl_qty_ordered"]]
    .groupby(["srp_2_code","srp_2_desc"])["dtl_qty_ordered"]
    .mean()
    .reset_index()
)
products = products.merge(temp_prod6, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"dtl_qty_ordered": "avg_order_amount"}, inplace=True)

# Adding the average sales per year of the product in customer sales table as a feature

# Getting the total no. of products purhcased and first and last purchase dates of each customer
temp_prod7 = CSD.groupby(["srp_2_code","srp_2_desc"]).agg({"dtl_qty_ordered":"sum","dtl_order_dt":["min","max"]})

# Computing the number of days between first and last purchases
temp_prod7["delta_days"]=(temp_prod7[("dtl_order_dt","max")]-temp_prod7[("dtl_order_dt","min")]).dt.days

# Dividing the total no. of products purhcased by the number of days between first and last purchases
temp_prod7["avg_sales_per_year"]=temp_prod7[("dtl_qty_ordered","sum")]/temp_prod7["delta_days"]*365

# Merging the nessecary feature with the main table 
temp_prod7 = temp_prod7.reset_index()[["srp_2_code","srp_2_desc","avg_sales_per_year"]]
temp_prod7.columns = ["srp_2_code","srp_2_desc","avg_sales_per_year"]
products = products.merge(temp_prod7,on=["srp_2_code","srp_2_desc"], how="left")

# Adding the list of opportunities won which contained the product in salesforce table as a feature
temp_prod8 = (
    SFD[SFD["iswon"]=='True'][["srp_2_desc", "opp_id"]]
    .groupby("srp_2_desc")["opp_id"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod8, on="srp_2_desc", how="left")
products.rename(columns={"opp_id": "opp_won_list"}, inplace=True)

# Adding the list of opportunities lost which contained the product in salesforce table as a feature
temp_prod9 = (
    SFD[SFD["iswon"]=='False'][["srp_2_desc", "opp_id"]]
    .groupby("srp_2_desc")["opp_id"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod9, on="srp_2_desc", how="left")
products.rename(columns={"opp_id": "opp_lost_list"}, inplace=True)

# COMMAND ----------

# DBTITLE 1,Save the data for product clustering
# Save the products for product clustering
utils.save_parquet(products,f"{product_data_path}/product_details.parquet")

# COMMAND ----------

# DBTITLE 1,Save the results in the stats.json file
stats = {}

utils.customer_product_stats(stats,products,"product clustering")

utils.save_stats(stats)