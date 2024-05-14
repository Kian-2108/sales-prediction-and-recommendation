# Databricks notebook source
# MAGIC %md
# MAGIC ### Dataset Creation for Propensity Model
# MAGIC This Notebook consits of building a custom dataset for the propensity model. Later section of the notebook consists of using correlation to drop highly correlated features.

# COMMAND ----------

# DBTITLE 1,Load Libraries
#Utils is a script imported from Utility_functions folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration file
config = utils.read_config()

# Extract the data version from the configuration file
data_version = config["data_version"]["value"]
cutoff_date = pd.to_datetime(config["cutoff_date"]["value"])

# Extract the maximum number of days between purchases from the configuration file
max_days_between_purchase = config["max_days_between_purchase"]["value"]

# Extract the near sale cutoff value from the configuration file
near_sale_cutoff = config["near_sale_cutoff"]["value"]


# COMMAND ----------

# DBTITLE 1,Data paths
#path of data before cutoff
before_cutoff_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"

#path of data after cutoff
after_cutoff_path = f"DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

#path of data related to propensity model
propensity_data_path = f"DATA_VERSIONS/{data_version}/PROPENSITY_DATA"

# COMMAND ----------

# DBTITLE 1,Load data
#load Capital Sales Data before cutoff
BEFORE_CUTOFF_CSD = utils.read_parquet(f"{before_cutoff_path}/customer_sales_data.parquet")

#load Capital Sales Data after cutoff
AFTER_CUTOFF_CSD = utils.read_parquet(f"{after_cutoff_path}/customer_sales_data.parquet")

# COMMAND ----------

# DBTITLE 1,Function to group distinct customer, product & invoice num
def grouped(data):
    """
    Group the input data by "customer_num", "srp_2_desc", and "dtl_invoice_num",
    keep the first occurrence in each group, and filter out columns with non-single values.

    Args:
        data (pandas.DataFrame): The input DataFrame containing the data to be grouped.

    Returns:
        pandas.DataFrame: The grouped DataFrame with only the first occurrence in each group,
                          filtered to include only columns with non-single values,
                          and sorted based on "customer_num", "srp_2_desc", and "dtl_invoice_dt".
    """
    grouped_data = data.groupby(["customer_num","srp_2_desc","dtl_invoice_num"]).first().reset_index()
    data_non_single = []
    for col in grouped_data.columns:
        if len(grouped_data[col].unique())>1:
            data_non_single.append(col)
    grouped_data = grouped_data[data_non_single].sort_values(['customer_num','srp_2_desc','dtl_invoice_dt'])
    return grouped_data

# COMMAND ----------

BeforeCutoffCSD = grouped(BEFORE_CUTOFF_CSD)
AfterCutoffCSD = grouped(AFTER_CUTOFF_CSD)

# COMMAND ----------

# DBTITLE 1,Compute the next purchase date
# For every Customer/Product pair before cut off get the date of next purchase after cut off
NextPurchase = AfterCutoffCSD.groupby(['customer_num','srp_2_desc']).dtl_invoice_dt.min().reset_index()
NextPurchase.columns = ['customer_num','srp_2_desc','NextPurchaseAfterCutoff']
NextPurchase

# COMMAND ----------

# DBTITLE 1,Compute the last purchase date
# For every Customer/Product pair before cut off get the date of last purchase before cut off
LastPurchase = BeforeCutoffCSD.groupby(['customer_num','srp_2_desc']).dtl_invoice_dt.max().reset_index()
LastPurchase.columns = ['customer_num','srp_2_desc','LastPurchaseBeforeCutoff']
LastPurchase

# COMMAND ----------

# DBTITLE 1,Compute cutoff to next purchase
# Computing the no. of days till the next purchase
PurchaseDatesDF = pd.merge(LastPurchase,NextPurchase,on=['customer_num','srp_2_desc'], how='left')
PurchaseDatesDF['DaysBetweenLastTwoPurchases'] = (PurchaseDatesDF['NextPurchaseAfterCutoff']-PurchaseDatesDF['LastPurchaseBeforeCutoff']).dt.days
PurchaseDatesDF['LastPurchaseToCutoff'] = (cutoff_date-PurchaseDatesDF['LastPurchaseBeforeCutoff']).dt.days

# Assuming the next purchase will happen after 5000 days if Customer/Product pair does not apprear after cut off date
PurchaseDatesDF['DaysBetweenLastTwoPurchases'].fillna(max_days_between_purchase,inplace=True)
PurchaseDatesDF['CutoffToNextPurchase'] = (PurchaseDatesDF['DaysBetweenLastTwoPurchases']-PurchaseDatesDF['LastPurchaseToCutoff'])
PurchaseDatesDF

# COMMAND ----------

# DBTITLE 1,Get the invoice date 
no_of_invoice = 5
for i in range(1,no_of_invoice+1):
    # Create new columns in the BeforeCutoffCSD DataFrame for each invoice date. Shift the 'dtl_invoice_dt' column by 'i' positions within each group of 'customer_num' and 'srp_2_desc'
    BeforeCutoffCSD[f"#{i}InvoiceDate"] = BeforeCutoffCSD.groupby(['customer_num','srp_2_desc'],as_index=False)['dtl_invoice_dt'].shift(i)

#Display new DataFrame containing columns 'customer_num', 'srp_2_desc', 'dtl_invoice_dt', and all the generated invoice date columns using list comprehension
BeforeCutoffCSD[['customer_num','srp_2_desc','dtl_invoice_dt']+[f"#{i}InvoiceDate" for i in range(1,no_of_invoice+1)]]

# COMMAND ----------

# DBTITLE 1,Calculate the no. of days between k-1 and kth purchase
BeforeCutoffCSD[f"DayDiff1"] = (BeforeCutoffCSD[f"dtl_invoice_dt"] - BeforeCutoffCSD[f"#1InvoiceDate"]).dt.days
for i in range(2,no_of_invoice+1):
    # Calculate the day differences between the shifted invoice dates
    BeforeCutoffCSD[f"DayDiff{i}"] = (BeforeCutoffCSD[f"#{i-1}InvoiceDate"] - BeforeCutoffCSD[f"#{i}InvoiceDate"]).dt.days
BeforeCutoffCSD[
    ["customer_num", "srp_2_desc", "dtl_invoice_dt"]
    + [f"#{i}InvoiceDate" for i in range(1, no_of_invoice + 1)]
    + [f"DayDiff{i}" for i in range(1, no_of_invoice)]
]

# COMMAND ----------

# DBTITLE 1,Calculate mean and std deviation
# Calculating the Mean and Standard deviation of no. of days between consecutive purchases
sales_day_diff = (
    BeforeCutoffCSD.groupby(["customer_num", "srp_2_desc"])
    .agg({"DayDiff1": ["mean", "std"]})
    .reset_index()
)
sales_day_diff.columns = ["customer_num", "srp_2_desc", "DayDiffMean", "DayDiffStd"]
sales_day_diff

# COMMAND ----------

# Droping all but the last entry for every Customer/Product Pair
BeforeCutoffCSD2 = BeforeCutoffCSD.drop_duplicates(subset=['customer_num','srp_2_desc'],keep='last')

# COMMAND ----------

# DBTITLE 1,Merging the the features extracted into a single table
BeforeCutoffCSD4ML = BeforeCutoffCSD2.copy()

# Merge BeforeCutoffCSD4ML with sales_day_diff DataFrame based on 'customer_num' and 'srp_2_desc'
BeforeCutoffCSD4ML = pd.merge(BeforeCutoffCSD4ML, sales_day_diff, on=['customer_num','srp_2_desc'])

# Merge BeforeCutoffCSD4ML with PurchaseDatesDF DataFrame based on 'customer_num' and 'srp_2_desc'
BeforeCutoffCSD4ML = pd.merge(BeforeCutoffCSD4ML, PurchaseDatesDF, on=['customer_num','srp_2_desc'])
BeforeCutoffCSD4ML

# COMMAND ----------

# DBTITLE 1,Binning the output feature 
# Binning the Feature to get the output feature (CutoffToNextPurchase)
BeforeCutoffCSD4ML["outcome"] = BeforeCutoffCSD4ML["CutoffToNextPurchase"].apply(lambda x: 1 if x <= near_sale_cutoff else 0)

# COMMAND ----------

# Checking how many entries have next purchase within an year of cut off
BeforeCutoffCSD4ML[BeforeCutoffCSD4ML["CutoffToNextPurchase"]<= near_sale_cutoff]

# COMMAND ----------

# DBTITLE 1,Correlation of features
# Computing the correlation between all features

Corr_data = BeforeCutoffCSD4ML.copy()
for col in Corr_data.select_dtypes(["object","datetime64[ns]"]).columns:
    Corr_data[col] = pd.factorize(Corr_data[col])[0]
# abs(Corr_data.corr()).style.background_gradient()

plt.figure(figsize=(50,40))
plt.rcParams.update({'font.size': 20})
sns.heatmap(abs(Corr_data.corr()),annot=False)

# COMMAND ----------

# DBTITLE 1,Feature list of the new custom dataset
# Dropping highlity correlated features
LimitedCols = [
    "customer_num",
    "srp_2_desc",
    # "dtl_invoice_num",
    # "dtl_actual_ship_dt",
    # "config_rollup_cost",
    # "dtl_crtd_dt",
    # "dtl_doc_type",
    # "dtl_extended_cost",
    # "dtl_extended_price",
    # "dtl_gl_dt",
    "dtl_invoice_dt",
    # "dtl_item_cd",
    # "dtl_item_desc_line_1",
    # "dtl_order_num",
    # "dtl_line_num",
    # "dtl_line_type",
    "dtl_line_type_desc",
    # "dtl_order_co",
    # "dtl_order_dt",
    # "dtl_orig_promised_deliv_dt",
    # "dtl_promised_deliv_dt",
    # "dtl_promished_ship_dt",
    "dtl_qty_ordered",
    # "dtl_qty_shipped",
    # "dtl_qty_shipped_to_date",
    # "dtl_ref",
    # "dtl_requested_dt",
    "dtl_unit_cost",
    "dtl_unit_price",
    # "dtl_value_package",
    "dtl_value_package_desc",
    # "hdr_supergroup",
    "hdr_supergroup_desc",
    # "line_enter_date",
    # "line_open_date",
    # "tot_contract_discount",
    # "tot_dscrtnry_disc",
    "total_discount",
    # "item_cd",
    "item_desc",
    # "rental_cd",
    # "rental_cd_desc",
    # "csms_prod_family",
    "csms_prod_family_desc",
    # "srp_1_code",
    "srp_1_desc",
    # "srp_2_code",
    # "srp_3_code",
    "srp_3_desc",
    # "srp_4_code",
    # "srp_4_desc",
    "active_cust_yn",
    # "addr_ln_1",
    "bill_to_yn",
    # "ship_to_yn",
    "city",
    # "country",
    "county",
    # "customer_class",
    # "customer_class_desc",
    # "customer_name",
    # "customer_price_group",
    "customer_price_group_desc",
    # "customer_type",
    "customer_type_desc",
    "primary_physical_loc_yn",
    # "search_type",
    # "search_type_desc",
    # "ship_to_yn.1",
    # "state",
    # "state_desc",
    # "zip",
    # "active_cust_yn.1",
    # "#1InvoiceDate",
    # "#2InvoiceDate",
    # "#3InvoiceDate",
    # "#4InvoiceDate",
    # "DayDiff1",
    # "DayDiff2",
    # "DayDiff3",
    # "DayDiff4",
    "DayDiffMean",
    "DayDiffStd",
    "LastPurchaseToCutoff",
    # "CutoffToNextPurchase",
    "outcome",
]

# COMMAND ----------

# DBTITLE 1,Correlation after dropping features
# Computing the correlation between all features after dropping highly correlated features

Corr_data = BeforeCutoffCSD4ML[LimitedCols].copy()
for col in Corr_data.select_dtypes(["object","datetime64[ns]"]).columns:
    Corr_data[col] = pd.factorize(Corr_data[col])[0]
# abs(Corr_data.corr()).style.background_gradient()

plt.figure(figsize=(50,40))
plt.rcParams.update({'font.size': 30})
sns.heatmap(abs(Corr_data.corr()),annot=False)

# COMMAND ----------

# Dropping highly correlated features
BeforeCutoffCSD4ML = BeforeCutoffCSD4ML[LimitedCols].dropna()

AfterCutoffCSD4ML = AfterCutoffCSD.drop_duplicates(subset=["customer_num","srp_2_desc"],keep="first")
AfterCutoffCSD4ML["outcome"] = AfterCutoffCSD4ML["dtl_invoice_dt"].apply(lambda x: 1 if x<=cutoff_date+pd.Timedelta(near_sale_cutoff,unit="d") else 0)
AfterCutoffCSD4ML = AfterCutoffCSD4ML[["customer_num","srp_2_desc","outcome"]]

# COMMAND ----------

# DBTITLE 1,Save the new custom dataset
#save the new dataset so that it can be easilty used while building the propensity model
utils.save_parquet(BeforeCutoffCSD4ML,f"{propensity_data_path}/propensity_data_train.parquet")
utils.save_parquet(AfterCutoffCSD4ML,f"{propensity_data_path}/propensity_data_test.parquet")