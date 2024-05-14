# Databricks notebook source
# MAGIC %md
# MAGIC ### Truncate Data
# MAGIC This notebook splits dataset into two parts (before cutoff and after cutoff), depending on the date given in the config file.

# COMMAND ----------

# DBTITLE 1,Load Libraries
# Utils is a python script imported from Utility functions folder

import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Read config file
# Read the configuration settings from the config file

config = utils.read_config()
data_version = config["data_version"]["value"]
cutoff_date = pd.to_datetime(config["cutoff_date"]["value"])

# COMMAND ----------

# DBTITLE 1,Data Path
#path of raw data
raw_data_path = f"DATA_VERSIONS/{data_version}/CLEAN_RAW_DATA"
#path of the folder where data before cutoff are stored
before_cutoff_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"
#path of the folder where data after cutoff are stored
after_cutoff_path = f"DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

# COMMAND ----------

# DBTITLE 1,Load raw data
#load Capital Sales Data
CSD = utils.read_parquet(f"{raw_data_path}/customer_sales_data.parquet")
#load Salesforce Data
SFD = utils.read_parquet(f"{raw_data_path}/sales_force_data.parquet")
#load Rental Sales Data
RSD = utils.read_parquet(f"{raw_data_path}/rental_sales_data.parquet")
#load Customer Baseline Data
CBD = utils.read_parquet(f"{raw_data_path}/customer_baseline_data.parquet")
#load Customer Crosswalk Data
CWD = utils.read_parquet(f"{raw_data_path}/customer_crosswalk_data.parquet")
#load External Definitve Data
EDD = utils.read_parquet(f"{raw_data_path}/external_definitive_data.parquet")
#load Curated Product List
CPL = utils.read_parquet(f"{raw_data_path}/curated_products_data.parquet")
#load Unfiltered product List
UPL = utils.read_parquet(f"{raw_data_path}/unfiltered_products_data.parquet")

# COMMAND ----------

# DBTITLE 1,Cutoff sales force data
#Cutoff the data on the create_dt feature
SFD_before_cutoff = SFD[SFD["create_dt"] <= cutoff_date]
SFD_after_cutoff = SFD[SFD["create_dt"] > cutoff_date]

#Save the cutoff data
utils.save_parquet(SFD_before_cutoff, f"{before_cutoff_path}/sales_force_data.parquet")
utils.save_parquet(SFD_after_cutoff, f"{after_cutoff_path}/sales_force_data.parquet")

# COMMAND ----------

# DBTITLE 1,Cutoff capital sales data
#Cutoff the data on the invoice date feature
CSD_before_cutoff = CSD[CSD["dtl_invoice_dt"] <= cutoff_date]
CSD_after_cutoff = CSD[CSD["dtl_invoice_dt"] > cutoff_date]

#Save the cutoff data
utils.save_parquet(CSD_before_cutoff, f"{before_cutoff_path}/customer_sales_data.parquet")
utils.save_parquet(CSD_after_cutoff, f"{after_cutoff_path}/customer_sales_data.parquet")

# COMMAND ----------

# DBTITLE 1,Cutoff rental sales data
#Cutoff the data on the billing date feature

RSD_before_cutoff = RSD[RSD["billing_from_dt"] <= cutoff_date]
RSD_after_cutoff = RSD[RSD["billing_from_dt"] > cutoff_date]

#Save the cutoff data
utils.save_parquet(RSD_before_cutoff, f"{before_cutoff_path}/rental_sales_data.parquet")
utils.save_parquet(RSD_after_cutoff, f"{after_cutoff_path}/rental_sales_data.parquet")

# COMMAND ----------

# DBTITLE 1,Save before cutoff data
#The remaining data are saved as is.
utils.save_parquet(CBD,f"{before_cutoff_path}/customer_baseline_data.parquet")
utils.save_parquet(CWD,f"{before_cutoff_path}/customer_crosswalk_data.parquet")
utils.save_parquet(EDD,f"{before_cutoff_path}/external_definitive_data.parquet")
utils.save_parquet(CPL,f"{before_cutoff_path}/curated_products_data.parquet")
utils.save_parquet(UPL,f"{before_cutoff_path}/unfiltered_products_data.parquet")

# COMMAND ----------

# DBTITLE 1,Save after cutoff data
#The remaining data are saved as is.

utils.save_parquet(CBD,f"{after_cutoff_path}/customer_baseline_data.parquet")
utils.save_parquet(CWD,f"{after_cutoff_path}/customer_crosswalk_data.parquet")
utils.save_parquet(EDD,f"{after_cutoff_path}/external_definitive_data.parquet")
utils.save_parquet(CPL,f"{after_cutoff_path}/curated_products_data.parquet")
utils.save_parquet(UPL,f"{after_cutoff_path}/unfiltered_products_data.parquet")

# COMMAND ----------

# DBTITLE 1,Log stats
# Log stats

stats = {}

# Log stats for cleaned raw data
utils.customer_product_stats(stats,CSD,"customer sales")
utils.customer_product_stats(stats,SFD,"sales force")
utils.customer_product_stats(stats,RSD,"rental sales")
utils.customer_product_stats(stats,CBD,"customer baseline")
utils.customer_product_stats(stats,CWD,"customer crosswalk")
utils.customer_product_stats(stats,EDD,"external definitive")
utils.customer_product_stats(stats,CPL,"curated products")
utils.customer_product_stats(stats,UPL,"unfiltered products")

# Log stats for before cutoff data
utils.customer_product_stats(stats,CSD_before_cutoff,"before cutoff customer sales")
utils.customer_product_stats(stats,SFD_before_cutoff,"before cutoff sales force")
utils.customer_product_stats(stats,RSD_before_cutoff,"before cutoff rental sales")

#Log stats for after cutoff data
utils.customer_product_stats(stats,CSD_after_cutoff,"after cutoff customer sales")
utils.customer_product_stats(stats,SFD_after_cutoff,"after cutoff sales force")
utils.customer_product_stats(stats,RSD_after_cutoff,"after cutoff rental sales")

utils.save_stats(stats)