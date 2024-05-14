# Databricks notebook source
# MAGIC %md 
# MAGIC ### Create Custom Data for Customer Clustering
# MAGIC This notebook creates a custom dataset by using Capital Sales, Salesforce, Rental, and External Definitive data. The dataset involves creating new engineered features. 

# COMMAND ----------

# DBTITLE 1,Load Libraries
# utils is python scripts imported from the 0A_Utility_Functions folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration settings from a configuration file
config = utils.read_config()
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file paths
#path for before cutoff raw data based on the data version
raw_data_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"

#path for customer data based on the data version
customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"

# COMMAND ----------

# DBTITLE 1,Load files
#Read the customer sales data
CSD = utils.read_parquet(f"{raw_data_path}/customer_sales_data.parquet")

#Read the sales force data
SFD = utils.read_parquet(f"{raw_data_path}/sales_force_data.parquet")

#Read the rental sales data
RSD = utils.read_parquet(f"{raw_data_path}/rental_sales_data.parquet")

#Read the customer baseline data
CBD = utils.read_parquet(f"{raw_data_path}/customer_baseline_data.parquet")

#Read the customer crosswalk data
CWD = utils.read_parquet(f"{raw_data_path}/customer_crosswalk_data.parquet")

#Read the external definitive data
EDD = utils.read_parquet(f"{raw_data_path}/external_definitive_data.parquet")

#Read the curated products data
CPL = utils.read_parquet(f"{raw_data_path}/curated_products_data.parquet")

#Read the unfiltered products data
UPL = utils.read_parquet(f"{raw_data_path}/unfiltered_products_data.parquet")

# COMMAND ----------

# DBTITLE 1,Collecting all the customers from all the tables
# Get definitive id and name of all the customers in the external definitive data 
IN_EDD = EDD[["definitive_id", "customer_name"]].groupby("definitive_id").first().reset_index()

# Get definitive id, customer num and name of all customers in the crosswalk data
IN_CWD = CWD[["definitive_id", "customer_num", "customer_name"]].groupby("customer_num").first().reset_index()

# Get customer num and name of all customers in the customer sales data
IN_CSD = CSD[["customer_num", "customer_name"]].groupby("customer_num").first().reset_index()

# Get customer num and name of all customers in the rental sales data
IN_RSD = RSD[["customer_num", "customer_name"]].groupby("customer_num").first().reset_index()

# Get customer num and name of all customers in the salesforce data
IN_SFD = SFD[["customer_num"]].groupby("customer_num").first().reset_index()

# Get customer num and name of all customers in the customer baseline data
IN_CBD = CBD[["customer_num", "customer_name"]].groupby("customer_num").first().reset_index()

# Create a dictionary of all the above tables
all_customer_tables = {"IN_EDD":IN_EDD,"IN_CWD":IN_CWD,"IN_CSD":IN_CSD,"IN_RSD":IN_RSD,"IN_SFD":IN_SFD,"IN_CBD":IN_CBD}

# Add a tag to indicate which tables each customer is present in (after concatinating the null will be filled with False)
for col,table in all_customer_tables.items():
    # Fill empty customer num, definitive id or name with empty string
    table.fillna("",inplace=True)
    table[col] = [True for i in range(len(table))]

# COMMAND ----------

# DBTITLE 1,Getting the customer names from all the tables
# Combining all the customers in a single table
customers = IN_CWD[["definitive_id","customer_num","IN_CWD"]].merge(IN_EDD[["definitive_id","IN_EDD"]],on="definitive_id",how="outer")
customers = customers.merge(IN_CSD[["customer_num","IN_CSD"]],on="customer_num",how="outer")
customers = customers.merge(IN_SFD[["customer_num","IN_SFD"]],on="customer_num",how="outer")
customers = customers.merge(IN_RSD[["customer_num","IN_RSD"]],on="customer_num",how="outer")
customers = customers.merge(IN_CBD[["customer_num","IN_CBD"]],on="customer_num",how="outer")
customers[["definitive_id","customer_num"]] = customers[["definitive_id","customer_num"]].fillna("")

# Filling the empty tags as False
customers.fillna(False,inplace=True)

# Adding the customer names from customer baseline table
customers = customers.merge(IN_CBD.drop("IN_CBD",axis=1),on="customer_num",how="outer")

# Adding the customer names from external definitive data
customers = customers.merge(IN_EDD.drop("IN_EDD",axis=1),on=["definitive_id"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

# Adding the customer names from customer sales data
customers = customers.merge(IN_CSD.drop("IN_CSD",axis=1),on=["customer_num"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

# Adding the customer names from customer crosswalk data
customers = customers.merge(IN_CWD.drop(["IN_CWD","definitive_id"],axis=1),on=["customer_num"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

# Adding the customer names from rental sales data
customers = customers.merge(IN_RSD.drop("IN_RSD",axis=1),on=["customer_num"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

# Converting all customer names to uppercase for uniformity
customers["customer_name"] = customers["customer_name"].str.upper()

# Reordering the columns
customers = customers[["customer_num","definitive_id","customer_name","IN_CBD","IN_CSD","IN_CWD","IN_EDD","IN_RSD","IN_SFD"]]

# COMMAND ----------

# DBTITLE 1,Selecting attributes of customer used for customer clustering
# Filtering all customers from customer sales, salesforce and external definitive data
customer_details = customers[
    (customers["IN_CSD"]) | (customers["IN_SFD"]) | (customers["IN_EDD"])
][["definitive_id", "customer_num", "customer_name"]]

# Merging feature of customers from the customer baseline data
customer_details = customer_details.merge(
    CBD[
        [
            "customer_num",
            "active_cust_yn",
            "customer_class",
            "customer_price_group",
            "customer_type",
            "gl_acct_classification",
        ]
    ],
    on="customer_num",
    how="left",
)

# Merging feature of customers from the external definitive data 
customer_details = customer_details.merge(
    EDD[
        [
            "definitive_id",
            "No. of Discharges",
            "No. of Staffed Beds",
            "340B Classification",
            "Hospital Type",
            "IDN",
            "IDN Parent",
            "Bed Utilization Rate",
            "Capital Expenditures",
            "Net Income",
            "Latitude",
            "Longitude",
        ]
    ],
    on="definitive_id",
    how="left",
)

# Adding the list of products purchased by the customer in customer sales table as a feature
temp_prod1 = (
    CSD[["customer_num", "srp_2_desc"]]
    .groupby("customer_num")["srp_2_desc"]
    .apply(list)
    .reset_index()
)
customer_details = customer_details.merge(temp_prod1, on="customer_num", how="left")
customer_details.rename(columns={"srp_2_desc": "sales_list"}, inplace=True)

# Adding the list of products purchased by the customer in salesforce table as a feature
temp_prod2 = (
    SFD[SFD["iswon"] == "True"][["customer_num", "srp_2_desc"]]
    .dropna()
    .groupby("customer_num")["srp_2_desc"]
    .apply(list)
    .reset_index()
)
customer_details = customer_details.merge(temp_prod2, on="customer_num", how="left")
customer_details.rename(columns={"srp_2_desc": "salesforce_list"}, inplace=True)

# Adding the list of products rented by the customer in rental sales table as a feature
temp_prod3 = (
    RSD[["customer_num", "srp_2_desc"]]
    .groupby("customer_num")["srp_2_desc"]
    .apply(list)
    .reset_index()
)
customer_details = customer_details.merge(temp_prod3, on="customer_num", how="left")
customer_details.rename(columns={"srp_2_desc": "rental_list"}, inplace=True)

# Adding the average number of purchases per year by a customer in customer sales table as a feature

# Getting the total no. of products purhcased and first and last purchase dates of each customer
temp_prod4 = CSD.groupby("customer_num").agg(
    {"dtl_qty_ordered": "sum", "dtl_order_dt": ["min", "max"]}
)

# Computing the number of days between first and last purchases
temp_prod4["delta_days"] = (
    temp_prod4[("dtl_order_dt", "max")] - temp_prod4[("dtl_order_dt", "min")]
).dt.days

# Dividing the total no. of products purhcased by the number of days between first and last purchases
temp_prod4["purchases_per_year"] = (
    temp_prod4[("dtl_qty_ordered", "sum")] / temp_prod4["delta_days"] * 365
)

# Merging the nessecary feature with the main table 
temp_prod4 = temp_prod4.reset_index()[["customer_num", "purchases_per_year"]]
temp_prod4.columns = ["customer_num", "purchase_per_year"]
customer_details = customer_details.merge(temp_prod4, on="customer_num", how="left")

# Creating a unique id (customerNum_definitiveId) for indexing as both definitive id and customer num contain null values
customer_details["unique_id"] = customer_details["customer_num"] + "_" + customer_details["definitive_id"]

# Using the unique id if customer id is missing
customer_details["customer_num"] = customer_details["customer_num"].where(customer_details["customer_num"]!="",customer_details["unique_id"])

customer_details.set_index("unique_id", inplace=True)

# COMMAND ----------

# DBTITLE 1,Save the data for customer clustering
# Save the customers table for refrence and customer_details for customer clustering
utils.save_parquet(customers,f"{customer_data_path}/customer_interactions.parquet")
utils.save_parquet(customer_details,f"{customer_data_path}/customer_details.parquet")

# COMMAND ----------

# DBTITLE 1,Save the results in the stats.json file
stats = {}

utils.customer_product_stats(stats,customer_details,"customer clustering")

utils.save_stats(stats)