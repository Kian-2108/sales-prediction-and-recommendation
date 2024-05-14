# Databricks notebook source
# MAGIC %md
# MAGIC ### Create custom dataset for Prediction
# MAGIC In this notebook the custom dataset is created using the Capital Sales and Salesforce data.

# COMMAND ----------

# DBTITLE 1,Import Libraries
#Utils is a python script imported from Utility_function folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import mlflow
import numpy as np
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Load Config file
# Read the configuration settings from the configuration file
config = utils.read_config()
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Set Data path
#path of before cutoff folder based on the data version
before_cutoff_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"

#path of after cutoff folder based on the data version
after_cutoff_path = f"DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

#path of prediction folder based on the data version
output_data_path = f"DATA_VERSIONS/{data_version}/PREDICTION_DATA"

# COMMAND ----------

# DBTITLE 1,Load files
# Load the before cutoff customer sales data
BEFORE_CUTOFF_CSD = utils.read_parquet(f"{before_cutoff_path}/customer_sales_data.parquet")

# Load the before cutoff salesforce data
BEFORE_CUTOFF_SFD = utils.read_parquet(f"{before_cutoff_path}/sales_force_data.parquet")

# Load the after cutoff customer sales data
AFTER_CUTOFF_CSD = utils.read_parquet(f"{after_cutoff_path}/customer_sales_data.parquet")

# Load the after cutoff salesforce data
AFTER_CUTOFF_SFD = utils.read_parquet(f"{after_cutoff_path}/sales_force_data.parquet")

# COMMAND ----------

# DBTITLE 1,Create Custom Dataset
def fill_quadrants(input_CSD, input_SFD):
    """
    The approach of filling the dataset is done by appending the Capital Sales and Salesforce data. Appending these two data creates a new data (table) with four quadrant, with capital sales in the 2nd quadrant and Salesforce in the 4th quadrant. Whereby the 1st and 3rd quadrant are null values. The objective of this function is to impute the 1st and 3rd quadrant. (Ref Approach doc for a visualization of this approach)

    Parameters:
        input_data_path (str): Path to the input data (Capital Sales and Salesforce)
        output_data_path (str): Path to save the output data

    Returns:
        None
    """
    grouped_SFD = (
        input_SFD.groupby(["customer_num", "srp_2_desc", "opp_id"])
        .first()
        .reset_index()
    )
    # Figuring out columns which has atleast more than one unique value in Salesforce Data
    SFD_non_single = []
    for col in grouped_SFD.columns:
        if len(grouped_SFD[col].unique()) > 1:
            SFD_non_single.append(col)

    grouped_SFD = grouped_SFD[set(SFD_non_single + ["cutoff_tag"])]

    grouped_CSD = (
        input_CSD.groupby(["customer_num", "srp_2_desc", "dtl_invoice_num"])
        .first()
        .reset_index()
    )
    # Figuring out columns which has atleast more than one unique value in Capital Sales Data
    CSD_non_single = []
    for col in grouped_CSD.columns:
        if len(grouped_CSD[col].unique()) > 1:
            CSD_non_single.append(col)

    grouped_CSD = grouped_CSD[set(CSD_non_single + ["cutoff_tag"])]

    top_L_data = grouped_CSD
    # Pick only 'Lost' opportunities from Salesforce that can later be appended to the final dataset
    bot_R_data = grouped_SFD[grouped_SFD["iswon"] == "False"]
    tem_R_data = grouped_SFD[grouped_SFD["iswon"] == "True"]

    # Create top portion of the dataset with columns with values from Capital Sales Data & null values for rest of the columns
    top_data = pd.concat(
        [
            top_L_data,
            pd.DataFrame(
                index=top_L_data.index,
                columns=(bot_R_data.columns.difference(top_L_data.columns)),
            ),
        ],
        axis=1,
    )
    # Create bottom portion of the dataset with columns with values from Salesforce Data & null values for rest of the columns
    bot_data = pd.concat(
        [
            pd.DataFrame(
                index=bot_R_data.index,
                columns=(top_L_data.columns.difference(bot_R_data.columns)),
            ),
            bot_R_data,
        ],
        axis=1,
    )

    # Join with Captial Sales Data at the top portion with Salesforce data that were 'won' to fill the top right(empty columns)
    df = top_L_data.merge(
        tem_R_data.drop(
            (top_L_data.columns.intersection(tem_R_data.columns)).difference(
                set(["customer_num", "srp_2_desc"])
            ),
            axis=1,
        ),
        on=["customer_num", "srp_2_desc"],
        how="left",
    )
    df["delta"] = abs(df["close_dt"] - df["dtl_order_dt"]).dt.days
    df = df[df["delta"].notna()]
    df.sort_values("delta", inplace=True)
    df = (
        df.groupby(["customer_num", "srp_2_desc", "dtl_invoice_num"])
        .first()
        .reset_index()
    )
    # Extract rows where the difference between 'close_dt' from Salesforce and 'dtl_order_dt' from Capital Sales is within quarter
    df = df[df["delta"] <= 90].reset_index(drop=True)
    df.drop("delta", axis=1, inplace=True)

    # Include the above extract into the top portion of the dataset
    top_data = pd.concat([top_data, df], axis=0).drop_duplicates(
        subset=["customer_num", "srp_2_desc", "dtl_invoice_num"], keep="last"
    )
    # Merge top & bottom portion created thus far to complete the dataset
    total_data = pd.concat([top_data, bot_data], axis=0)

    match_on_customers = [
        "addr_ln_1",
        "bill_to_yn",
        "city",
        "county",
        "customer_class",
        "customer_class_desc",
        "customer_name",
        "customer_price_group",
        "customer_price_group_desc",
        "hdr_supergroup",
        "hdr_supergroup_desc",
        "primary_physical_loc_yn",
        "rental_cd",
        "search_type",
        "search_type_desc",
        "ship_to_yn",
        "ship_to_yn.1",
    ]

    match_on_products = [
        "csms_prod_family",
        "csms_prod_family_desc",
        "dtl_item_cd",
        "dtl_item_desc_line_1",
        "dtl_line_type",
        "dtl_line_type_desc",
        "dtl_value_package",
        "dtl_value_package_desc",
        "item_cd",
        "item_desc",
        "dtl_qty_ordered",
        "dtl_qty_shipped",
        "dtl_qty_shipped_to_date",
        "dtl_unit_cost",
        "dtl_unit_price",
        "family",
    ]

    def match_fillna(match_col, fill_col):
        """
        Match same customer/product data and fill missing values in the specified column

        Parameters:
            match_col (str): Name of the column used for matching.
            fill_col (str): Name of the column to fill missing values.

        Returns:
            numpy.ndarray: Array containing the filled values.
        """
        temp_df = (
            total_data[[match_col, fill_col]]
            .groupby(match_col)
            .first()
            .reset_index()
            .dropna()
            .rename(columns={fill_col: "drop_col"})
        )
        test = total_data[[match_col, fill_col]].merge(
            temp_df, on=[match_col], how="left"
        )
        filled = test[fill_col].fillna(test["drop_col"])
        return filled.values

    match_col = "customer_num"
    for fill_col in match_on_customers:
        filled = match_fillna(match_col, fill_col)
        total_data[fill_col] = filled

    match_col = "srp_2_desc"
    for fill_col in match_on_products:
        filled = match_fillna(match_col, fill_col)
        total_data[fill_col] = filled
    # Label the top portion which would be empty to 'True', bottom set will already be labelled as 'False'
    total_data["iswon"].fillna("True", inplace=True)
    total_data["stagename"].fillna("Closed Won", inplace=True)

    # Impute value for 'family' for NC_Maintenance
    total_data["family"] = total_data["family"].where(
        total_data["srp_2_desc"] != "NC MAINTENANCE", "COMMUNICATIONS"
    )
    # Impute value for 'family' for TRUMPF TABLES
    total_data["family"] = total_data["family"].where(
        total_data["srp_2_desc"] != "TRUMPF TABLES", "Trumpf Medical tables"
    )

    selected_features = [
        "addr_ln_1",
        "bill_to_yn",
        "city",
        "county",
        "csms_prod_family_desc",
        "customer_num",
        "customer_price_group",
        "customer_price_group_desc",
        "hdr_supergroup",
        "hdr_supergroup_desc",
        "primary_physical_loc_yn",
        "search_type",
        "ship_to_yn",
        "dtl_item_cd",
        "dtl_line_type",
        "item_desc",
        "family",
        "csms_prod_family",
        "srp_2_desc",
        "iswon",
        "dtl_qty_ordered",
        "dtl_unit_cost",
        "dtl_unit_price",
        "cutoff_tag",
    ]

    return total_data[selected_features].dropna()

# COMMAND ----------

# DBTITLE 1,Adding the cutoff tag to data
BEFORE_CUTOFF_CSD["cutoff_tag"] = ["BEFORE" for i in range(len(BEFORE_CUTOFF_CSD))] 
BEFORE_CUTOFF_SFD["cutoff_tag"] = ["BEFORE" for i in range(len(BEFORE_CUTOFF_SFD))] 
AFTER_CUTOFF_CSD["cutoff_tag"] = ["AFTER" for i in range(len(AFTER_CUTOFF_CSD))] 
AFTER_CUTOFF_SFD["cutoff_tag"] = ["AFTER" for i in range(len(AFTER_CUTOFF_SFD))]

TOTAL_CSD = pd.concat([BEFORE_CUTOFF_CSD,AFTER_CUTOFF_CSD],axis=0).reset_index(drop=True)
TOTAL_SFD = pd.concat([BEFORE_CUTOFF_SFD,AFTER_CUTOFF_SFD],axis=0).reset_index(drop=True)

# COMMAND ----------

# DBTITLE 1,Applying the function to fill the quadrants
before_cutoff_prediction_data = fill_quadrants(BEFORE_CUTOFF_CSD,BEFORE_CUTOFF_SFD)
after_cutoff_prediction_data = fill_quadrants(TOTAL_CSD,TOTAL_SFD)

# COMMAND ----------

# DBTITLE 1,Dropping the tag from the data
before_cutoff_prediction_data.drop("cutoff_tag",axis=1,inplace=True)
after_cutoff_prediction_data = after_cutoff_prediction_data[after_cutoff_prediction_data["cutoff_tag"]=="AFTER"].drop("cutoff_tag",axis=1)

# COMMAND ----------

# DBTITLE 1,Save the custom dataset
utils.save_parquet(before_cutoff_prediction_data, f'{output_data_path}/prediction_data_train.parquet')
utils.save_parquet(after_cutoff_prediction_data, f'{output_data_path}/prediction_data_test.parquet')

# COMMAND ----------

# DBTITLE 1,Save dataset stats
stats = {}

utils.customer_product_stats(stats,before_cutoff_prediction_data,"prediction train")
utils.customer_product_stats(stats,after_cutoff_prediction_data,"prediction test")

utils.save_stats(stats)