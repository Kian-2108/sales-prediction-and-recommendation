# Databricks notebook source
# MAGIC %md
# MAGIC ### Clean Raw Data
# MAGIC This notebook loads the raw csv files, cleans it and store it as parquet files.

# COMMAND ----------

# DBTITLE 1,Import Libraries
# utils is a python script imported from Utility_function folder

import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import os
import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,Read config file
# Load config file
config = utils.read_config()
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Data paths
#path for Capital Sales Data
CSD_path = config["CSD_path"]["value"]
#path for Salesforce Data
SFD_path = config["SFD_path"]["value"]
#path for Rental Sales Data
RSD_path = config["RSD_path"]["value"]
#path for crosswalk Data
CWD_path = config["CWD_path"]["value"]
#path for customer baseline Data
CBD_path = config["CBD_path"]["value"]
#path for external Sales Data
EDD_path = config["EDD_path"]["value"]
#path for Curated Product List
CPL_path = config["CPL_path"]["value"]
#path for Unfiltered Product List
UPL_path = config["UPL_path"]["value"]

# COMMAND ----------

# DBTITLE 1,Load files
#load Capital Sales data
CSD = utils.read_csv(CSD_path, None)
#load Salesforce data
SFD = utils.read_csv(SFD_path, None)
#load Rental Sales data
RSD = utils.read_csv(RSD_path, None)
#load Crosswalk data
CWD = utils.read_csv(CWD_path, None)
#load Customer Baseline data
CBD = utils.read_csv(CBD_path, None)
#load External Definitve data
EDD = utils.read_csv(EDD_path, None)
#load Curated Product List
CPL = utils.read_csv(CPL_path, None)
#load Unfiltered Product List
UPL = utils.read_csv(UPL_path, None)

clean_data_path = f"DATA_VERSIONS/{data_version}/CLEAN_RAW_DATA"

# COMMAND ----------

# DBTITLE 1,Define uniform feature names
#Define a uniform feature name across different dataset

definitive_id = "definitive_id"
customer_name = "customer_name"
customer_num = "customer_num"
srp_1_desc = "srp_1_desc"
srp_1_code = "srp_1_code"
srp_2_desc = "srp_2_desc"
srp_2_code = "srp_2_code"
srp_3_desc = "srp_3_desc"
srp_3_code = "srp_3_code"
srp_4_desc = "srp_4_desc"
srp_4_code = "srp_4_code"
total_staffed_beds = "total_staffed_beds"

# COMMAND ----------

# DBTITLE 1,Define helper functions to clean data

def convert_dtype(data, convert):
    """
    Convert columns of a DataFrame to the specified data types

    Parameters:
        data (pandas.DataFrame): The DataFrame to modify
        convert (dict): A dictionary where keys are column names and values are the desired data types

    Returns:
        None
    """
    for col, col_type in convert.items():
        data[col] = data[col].astype(col_type)

def replace_na(df, l=[ ".", "~", "-", "None", "none", "NONE", "nan", "NaN", "NAN", "*BLANK*", "", " ", "null", "NULL", "Null"]):
    """
    Replace specified values with NaN in the DataFrame

    Parameters:
        df (pandas.DataFrame): The DataFrame to modify
        l (list): List of values to replace with NaN. Default is a list of common representations of missing values

    Returns:
        None
    """
    for c in l:
        df.replace(c, np.nan, inplace=True)


def assign(x):
    """
    Function to remove special charecters from numbers

    Parameters:
        x: The value to process.

    Returns:
        processed_value: The processed value with replacements and conversions applied.
    """
    if x == "nan":
        return np.nan
    else:
        replaced = x.replace(",", "").replace("%", "")
        return replaced
    
def convert_dtype(data,convert):
    """
     Function to remove commas from string and convert to float

    Parameters:
        data (pandas.DataFrame): The DataFrame to modify.
        convert (dict): A dictionary where keys are column names and values are the desired data types.

    Returns:
        None
    """
    for col,col_type in convert.items():
        if col_type == 'float64':
            data[col] = data[col].astype('str').apply(assign)
        else:
            pass
        data[col] = data[col].astype(col_type)

# COMMAND ----------

# DBTITLE 1,Cleaning curated product data
# Cleaning curated product data

# display(CPL)

# Define the desired datatype for each feature
CL_convert = {
    "SRP2 - Description": "str",
    "SRP2 Code": "str",
    "Product Type": "str",
    "ACTIVE": "str",
    "REPLACEMENT": "str",
    "Replacement SRP2": "str",
    "LOCATION": "str",
}

# Define features names for uniformity
CL_columns = {
    "SRP2 - Description": "code_" + srp_2_desc,
    "SRP2 Code": srp_2_code,
    "Product Type": "product_type",
    "ACTIVE": "active",
    "REPLACEMENT": "replacement",
    "Replacement SRP2": "replacement_srp_2_code",
    "LOCATION": "location",
}

# Use the helper functions to clean the Curated Product List
convert_dtype(CPL, CL_convert)
CPL.rename(columns=CL_columns, inplace=True)

# Get the srp codes for the replacement if the product is inactive
CPL["replacement_srp_2_code"] = CPL["replacement_srp_2_code"].astype("float").fillna(0).astype("int").astype("str").replace("0",np.nan)
CPL["replacement_srp_2_code"] = CPL["replacement_srp_2_code"].fillna(CPL["srp_2_code"])

# Seprate product code and product name
CPL["srp_2_desc"] = CPL["code_srp_2_desc"].apply(lambda x: x[4:])
CPL["srp_2_replacement"] = CPL["replacement"].apply(lambda x: x[4:] if x != None else x)

# Create feature for replacement product
CPL["updated"] = CPL["srp_2_replacement"].where(
    CPL["srp_2_replacement"] != "", CPL["srp_2_desc"]
)

# Fill "empty" cells with nan
replace_na(CPL)

# Save cleaned data
utils.save_parquet(CPL, f"{clean_data_path}/curated_products_data.parquet")

# display(CPL)

# COMMAND ----------

# DBTITLE 1,Cleaning unfiltered product data
# Cleaning unfiltered product data

# display(UPL)

# Define the desired datatype for each feature
UL_convert = {
    "ods ods_capital_sales_order[DTL_SRP_1]": "str",
    "ods ods_capital_sales_order[DTL_SRP_1_DESC]": "str",
    "ods ods_capital_sales_order[DTL_SRP_2]": "str",
    "ods ods_capital_sales_order[DTL_SRP_2_DESC]": "str",
    "ods ods_capital_sales_order[DTL_SRP_3]": "str",
    "ods ods_capital_sales_order[DTL_SRP_3_DESC]": "str",
    "ods ods_capital_sales_order[DTL_SRP_4]": "str",
    "ods ods_capital_sales_order[DTL_SRP_4_DESC]": "str",
}

# Define features names for uniformity
UL_columns = {
    "ods ods_capital_sales_order[DTL_SRP_1]": srp_1_code,
    "ods ods_capital_sales_order[DTL_SRP_1_DESC]": srp_1_desc,
    "ods ods_capital_sales_order[DTL_SRP_2]": srp_2_code,
    "ods ods_capital_sales_order[DTL_SRP_2_DESC]": srp_2_desc,
    "ods ods_capital_sales_order[DTL_SRP_3]": srp_3_code,
    "ods ods_capital_sales_order[DTL_SRP_3_DESC]": srp_3_desc,
    "ods ods_capital_sales_order[DTL_SRP_4]": srp_4_code,
    "ods ods_capital_sales_order[DTL_SRP_4_DESC]": srp_4_desc,
}

# Select unwanted features to drop
UL_drop = ["Column1"]

# Apply above transforms
UPL.drop(UL_drop, axis=1, inplace=True)
convert_dtype(UPL, UL_convert)
UPL.rename(columns=UL_columns, inplace=True)

# Fill "empty" cells with nan
replace_na(UPL)

# Save cleaned data
utils.save_parquet(UPL, f"{clean_data_path}/unfiltered_products_data.parquet")

# display(UPL)

# COMMAND ----------

# DBTITLE 1,Cleaning rental sales data
# Cleaning rental sales data

# display(RSD)

# Define the desired datatype for each feature
RS_convert = {
    "order_type": "str",
    "order_type_desc": "str",
    "ln_type": "str",
    "ln_type_desc": "str",
    "billing_from_dt": "datetime64[ns]",
    "billing_through_dt": "datetime64[ns]",
    "bill_days": "int64",
    "extended_amount": "float64",
    "gl_dt": "datetime64[ns]",
    "item_cd": "str",
    "item_desc": "str",
    "sales_category_lvl_1": "str",
    "sales_category_lvl_1_desc": "str",
    "sales_category_lvl_2": "str",
    "sales_category_lvl_2_desc": "str",
    "sales_category_lvl_3": "str",
    "sales_category_lvl_3_desc": "str",
    "sales_category_lvl_4": "str",
    "sales_category_lvl_4_desc": "str",
    "customer_num": "str",
    "customer_nm": "str",
    "country_cd": "str",
}

# Define features names for uniformity
RS_columns = {
    "sales_category_lvl_1": srp_1_code,
    "sales_category_lvl_1_desc": srp_1_desc,
    "sales_category_lvl_2": srp_2_code,
    "sales_category_lvl_2_desc": srp_2_desc,
    "sales_category_lvl_3": srp_3_code,
    "sales_category_lvl_3_desc": srp_3_desc,
    "sales_category_lvl_4": srp_4_code,
    "sales_category_lvl_4_desc": srp_4_desc,
    "customer_num": customer_num,
    "customer_nm": customer_name,
}

#Use the helper functions to clean the Unfiltered Product List
convert_dtype(RSD, RS_convert)
RSD.rename(columns=RS_columns, inplace=True)

# Fill "empty" cells with nan
replace_na(RSD)

# Save cleaned data
utils.save_parquet(RSD, f"{clean_data_path}/rental_sales_data.parquet")

# display(RSD)

# COMMAND ----------

# DBTITLE 1,Cleaning sales force data
# display(SFD)

# Define the desired datatype for each feature
SF_convert = {
    "jde_number": "str",
    "opp_id": "str",
    "amount": "float64",
    "commit": "str",
    "iswon": "str",
    "createddate": "datetime64[ns]",
    "loss_comment": "str",
    "age_of_opportunity_days": "int64",
    "since_last_update_days": "int64",
    "total_discount": "float64",
    "isclosed": "str",
    "stagename": "str",
    "closedate": "datetime64[ns]",
    "currencyisocode": "str",
    "iso_country": "str",
    "isactive": "str",
    "package_code": "str",
    "package_desc": "str",
    "productcode": "str",
    "name": "str",
    "isdeleted": "str",
    "opplin_id": "str",
    "family": "str",
    "package_udc_code": "str",
    "price_from_jde": "float64",
    "srp_1_description": "str",
    "srp_2_description": "str",
    "srp_3_description": "str",
}

# Select unwanted features to drop
SF_drop = ["revenue_type", "END"]

# Define features names for uniformity
SF_columns = {
    "jde_number": customer_num,
    "createddate": "create_dt",
    "closedate": "close_dt",
    "srp_1_description": srp_1_desc,
    "srp_2_description": srp_2_desc,
    "srp_3_description": srp_3_desc,
}

#Use the helper functions to clean the Salesforce Data
SFD.drop(SF_drop, axis=1, inplace=True)
convert_dtype(SFD, SF_convert)
SFD.rename(columns=SF_columns, inplace=True)

# Fill "empty" cells with nan
replace_na(SFD)

# Save cleaned data
utils.save_parquet(SFD, f"{clean_data_path}/sales_force_data.parquet")

# display(SFD)

# COMMAND ----------

# DBTITLE 1,Cleaning customer baseline data
# Cleaning customer baseline data

# display(CBD)

# Define the desired datatype for each feature
CB_convert = {
    "customer_nm": "str",
    "customer_num": "str",
    "active_cust_yn": "str",
    "addr_ln_1": "str",
    "bill_to_yn": "str",
    "ship_to_yn": "str",
    "city": "str",
    "country": "str",
    "country_cd": "str",
    "county": "str",
    "customer_class": "str",
    "customer_class_desc": "str",
    "customer_price_group": "str",
    "customer_price_group_desc": "str",
    "customer_type": "str",
    "customer_type_desc": "str",
    "ownership_type": "str",
    "ownership_type_desc": "str",
    "gl_acct_classification": "str",
    "gl_acct_classification_desc": "str",
    "market_segment": "str",
    "market_segment_desc": "str",
    "patient_yn": "str",
    "primary_physical_loc_yn": "str",
    "search_type": "str",
    "search_type_desc": "str",
    "state": "str",
    "state_desc": "str",
    "zip": "str",
}

# Select unwanted features to drop
CB_drop = ["ship_to_yn.1", "active_cust_yn.1", "END"]

# Define features names for uniformity
CB_columns = {
    "customer_nm": customer_name,
    "customer_num": customer_num,
}

#Use the helper functions to clean the Customer Baseline Data
CBD.drop(CB_drop, axis=1, inplace=True)
convert_dtype(CBD, CB_convert)
CBD.rename(columns=CB_columns, inplace=True)

# Fill "empty" cells with nan
replace_na(CBD)

# Save cleaned data
utils.save_parquet(CBD, f"{clean_data_path}/customer_baseline_data.parquet")

# display(CBD)

# COMMAND ----------

# DBTITLE 1,Cleaning customer crosswalk data
# Cleaning customer crosswalk data

# display(CWD)

# Define the desired datatype for each feature
CW_convert = {
    "definitive_id_number": "str",
    "jde_id_number": "str",
    "mdm_reltio_uri": "str",
    "pk_edw_dim_customer": "str",
    "mdm_source_system": "str",
    "mdm_source_primarykey": "str",
    "name": "str",
    "country_code": "str",
    "city": "str",
    "state": "str",
    "street_address": "str",
    "zip5": "str",
    "postal_code": "str",
    "partner_function": "str",
    "pk_edw_dim_customer.1": "str",
    "status": "str",
    "total_staffed_beds": "str",
    "class_of_trade": "str",
    "hr_mdm_id": "str",
    "latitude": "str",
    "longitude": "str",
    "market_segment": "str",
}

# Select unwanted features to drop
CW_drop = ["definitive_id_alpha", "jde_id_alpha", "END"]

# Define features names for uniformity
CW_columns = {
    "definitive_id_number": definitive_id,
    "jde_id_number": customer_num,
    "name": customer_name,
    "total_staffed_beds": total_staffed_beds,
}

#Use the helper functions to clean the Crosswalk Data
CWD.drop(CW_drop, axis=1, inplace=True)
convert_dtype(CWD, CW_convert)
CWD.rename(columns=CW_columns, inplace=True)

# Fill "empty" cells with nan
replace_na(CWD)

# Save cleaned data
utils.save_parquet(CWD, f"{clean_data_path}/customer_crosswalk_data.parquet")

# display(CWD)

# COMMAND ----------

# DBTITLE 1,Cleaning Capital sales data
# Cleaning customer sales data

# display(CSD)

# Define the desired datatype for each feature
CS_convert = {
    "dtl_addr_num_ship_to": "str",
    "dtl_actual_ship_dt": "datetime64[ns]",
    "config_rollup_cost": "float64",
    "dtl_crtd_dt": "datetime64[ns]",
    "dtl_currency_cd_from": "str",
    "dtl_doc_type": "str",
    "dtl_extended_cost": "float64",
    "dtl_extended_price": "float64",
    "dtl_gl_dt": "datetime64[ns]",
    "dtl_invoice_dt": "datetime64[ns]",
    "dtl_invoice_num": "str",
    "dtl_item_cd": "str",
    "dtl_item_desc_line_1": "str",
    "dtl_order_num": "str",
    "dtl_line_num": "str",
    "dtl_line_type": "str",
    "dtl_line_type_desc": "str",
    "dtl_order_co": "str",
    "dtl_order_dt": "datetime64[ns]",
    "dtl_order_type": "str",
    "dtl_order_type_desc": "str",
    "dtl_orig_promised_deliv_dt": "datetime64[ns]",
    "dtl_promised_deliv_dt": "datetime64[ns]",
    "dtl_promished_ship_dt": "datetime64[ns]",
    "dtl_qty_ordered": "int64",
    "dtl_qty_shipped": "int64",
    "dtl_qty_shipped_to_date": "int64",
    "dtl_ref": "str",
    "dtl_requested_dt": "datetime64[ns]",
    "dtl_status": "str",
    "dtl_status_desc": "str",
    "dtl_unit_cost": "float64",
    "dtl_unit_price": "float64",
    "dtl_value_package": "str",
    "dtl_value_package_desc": "str",
    "hdr_supergroup": "str",
    "hdr_supergroup_desc": "str",
    "line_enter_date": "datetime64[ns]",
    "line_open_date": "datetime64[ns]",
    "tot_contract_discount": "float64",
    "tot_dscrtnry_disc": "str",
    "total_discount": "float64",
    "item_cd": "str",
    "item_desc": "str",
    "rental_cd": "str",
    "rental_cd_desc": "str",
    "csms_prod_family": "str",
    "csms_prod_family_desc": "str",
    "sales_category_lvl_1": "str",
    "sales_category_lvl_1_desc": "str",
    "sales_category_lvl_2": "str",
    "sales_category_lvl_2_desc": "str",
    "sales_category_lvl_3": "str",
    "sales_category_lvl_3_desc": "str",
    "sales_category_lvl_4": "str",
    "sales_category_lvl_4_desc": "str",
    "active_cust_yn": "str",
    "addr_ln_1": "str",
    "bill_to_yn": "str",
    "ship_to_yn": "str",
    "city": "str",
    "country": "str",
    "country_cd": "str",
    "county": "str",
    "customer_class": "str",
    "customer_class_desc": "str",
    "customer_nm": "str",
    "customer_price_group": "str",
    "customer_price_group_desc": "str",
    "customer_type": "str",
    "customer_type_desc": "str",
    "ownership_type": "str",
    "ownership_type_desc": "str",
    "gl_acct_classification": "str",
    "gl_acct_classification_desc": "str",
    "market_segment": "str",
    "market_segment_desc": "str",
    "patient_yn": "str",
    "primary_physical_loc_yn": "str",
    "search_type": "str",
    "search_type_desc": "str",
    "ship_to_yn.1": "str",
    "state": "str",
    "state_desc": "str",
    "zip": "str",
}

#S elect unwanted features to drop
CS_drop = ["END"]

# Define features names for uniformity
CS_columns = {
    "dtl_addr_num_ship_to": customer_num,
    "sales_category_lvl_1": srp_1_code,
    "sales_category_lvl_1_desc": srp_1_desc,
    "sales_category_lvl_2": srp_2_code,
    "sales_category_lvl_2_desc": srp_2_desc,
    "sales_category_lvl_3": srp_3_code,
    "sales_category_lvl_3_desc": srp_3_desc,
    "sales_category_lvl_4": srp_4_code,
    "sales_category_lvl_4_desc": srp_4_desc,
    "customer_nm": customer_name,
}

# Fix typos in dates
CSD.replace("2717-04-03", "2017-04-03", inplace=True)
CSD.replace("2716-04-20", "2016-04-20", inplace=True)
CSD.replace("2717-10-02", "2017-10-02", inplace=True)

#Use the helper functions to clean the Capital Sales Data
convert_dtype(CSD, CS_convert)
CSD.drop(CS_drop, axis=1, inplace=True)
CSD.rename(columns=CS_columns, inplace=True)

# Fill "empty" cells with nan
replace_na(CSD)

# Save cleaned data
utils.save_parquet(CSD, f"{clean_data_path}/customer_sales_data.parquet")

# display(CSD)

# COMMAND ----------

# DBTITLE 1,Cleaning external definitive data
# Cleaning external definitive data

# display(EDD)

# Define the desired datatype for each feature
DD_convert = {
    "Definitive ID": "str",
    "Hospital Name": "object",
    "Hospital Type": "object",
    "# of Discharges": "float64",
    "# of Staffed Beds": "float64",
    "340B Classification": "object",
    "340B ID Number": "str",
    "Academic": "object",
    "Address": "object",
    "Address1": "object",
    "Adjusted Patient Days": "float64",
    "All Cause Hospital Wide Readmission Cases": "object",
    "All Cause Hospital-Wide Readmission Rate": "object",
    "Average Age of Facility (Years)": "float64",
    "Average Daily Census": "float64",
    "Average Length of Stay": "float64",
    "Bed Utilization Rate": "float64",
    "Burn Intensive Care Unit Beds": "float64",
    "Capital Expenditures": "float64",
    "Case Mix Index": "float64",
    "Cash on Hand": "object",
    "CC/MCC Rate": "object",
    "Chronic Obstructive Pulmonary Disease (COPD) Readmission Rate": "object",
    "Chronic Obstructive Pulmonary Disease COPD Readmission Cases": "float64",
    "City": "object",
    "Coronary Care Unit Beds": "object",
    "County": "object",
    "Current Ratio": "float64",
    "Data_Breach": "object",
    "Debt to Equity Ratio": "float64",
    "Definitive IDN ID": "float64",
    "Definitive IDN Parent ID": "float64",
    "Detox Intensive Care Unit Beds": "float64",
    "DHC Profile Link": "object",
    "Electronic Health/Medical Record - Inpatient": "object",
    "Est # of ER Visits": "object",
    "Est # of Inpatient Surgeries": "object",
    "Est # of Outpatient Surgeries": "object",
    "Est # of Outpatient Visits": "object",
    "Est # of Total Surgeries": "object",
    "Est. IT Capital Budget": "object",
    "Est. IT Operating Expense Budget": "object",
    "Excess Readmission Ratio: CABG Patients": "float64",
    "Excess Readmission Ratio: COPD Patients": "float64",
    "Financial Data Date": "object",
    "FIPS County Code": "int64",
    "Firm Type": "object",
    "Fiscal Year End Date": "object",
    "Geographic Classification": "object",
    "HAC FY2020 Received Penalty": "object",
    "HAC FY2020 Total Score": "float64",
    "HAC FY2021 Received Penalty": "object",
    "HAC FY2021 Total Score": "float64",
    "Hospice Beds": "float64",
    "Hospital Compare Overall Rating": "float64",
    "IDN": "object",
    "IDN Parent": "object",
    "Inpatient Cost Center Square Footage": "object",
    "Inpatient Revenue": "object",
    "Intensive Care Unit Beds": "object",
    "Latitude": "float64",
    "Longitude": "float64",
    "Magnet": "object",
    "Market Concentration Index": "float64",
    "Medical CMI": "float64",
    "Medicare Spending per Patient": "float64",
    "Neonatal Intensive Care Unit Beds": "object",
    "Net Income": "float64",
    "Net Income Margin": "object",
    "Net Operating Profit Margin": "float64",
    "Net Patient Revenue": "float64",
    "NPI Number": "float64",
    "Number of Births": "object",
    "Number of Operating Rooms": "object",
    "Nursing Facility Beds": "float64",
    "Operating Income": "object",
    "Operating Room Square Feet": "object",
    "Other Income": "object",
    "Other Long-Term Care Beds": "object",
    "Other Special Care Beds": "float64",
    "Outpatient Cost Center Square Footage": "object",
    "Outpatient Revenue": "object",
    "Ownership": "object",
    "Patient Survey (HCAHPS) Summary Star Rating": "float64",
    "Patients at each hospital who reported that yes they were given information about what to do during their recovery at home Star Rating (out of 5)": "float64",
    "Payor Mix: Medicaid Days": "object",
    "Payor Mix: Medicare Days": "object",
    "Payor Mix: Private/Self-Pay/Other Days": "object",
    "Pediatric Intensive Care Unit Beds": "float64",
    "Pneumonia Readmission Cases": "float64",
    "Pneumonia Readmission Rate": "float64",
    "Premature Intensive Care Unit Beds": "float64",
    "Pressure Sores Rate": "float64",
    "Primary GPO ID": "float64",
    "Primary GPO Name": "object",
    "Provider Number": "object",
    "Psychiatric Intensive Care Unit Beds": "float64",
    "Psychiatric Unit Beds": "object",
    "Quick Ratio": "float64",
    "Region": "object",
    "Rehabilitation Unit Beds": "object",
    "Routine Service Beds": "object",
    "Serious Complication Rate": "float64",
    "Skilled Nursing Facility Beds": "float64",
    "Staffing Firm Relationships": "object",
    "State": "object",
    "Subcomponent Beds": "float64",
    "Surgical CMI": "float64",
    "Surgical Intensive Care Unit Beds": "float64",
    "Tax ID": "object",
    "Total Acute Beds": "object",
    "Total Contract Labor": "object",
    "Total Facility Square Footage": "object",
    "Total Med/Surg Supply Costs": "float64",
    "Total Operating Expenses": "float64",
    "Total Other Beds": "float64",
    "Total Patient Revenue": "float64",
    "Total Performance Score": "float64",
    "Total Salaries": "float64",
    "Trauma Intensive Care Unit Beds": "float64",
    "Website": "object",
    "Zip Code": "int64",
    "# of Discharges Divided By Sum Of All": "float64",
    "# of Discharges Median All": "object",
    "# of Discharges Sum All": "object",
    "Average Age of Facility (Years) Median Of All": "float64",
    "Average Age of Facility (Years) Sum Of All": "object",
    "Average Age of Facility (Years)t Divided By Sum Of All": "float64",
    "Average Length of Stay Divided By Sum Of All": "float64",
    "Average Length of Stay Median All": "float64",
    "Average Length of Stay Sum All": "object",
    "Bed Utilization Rate Divided By Sum Of All": "float64",
    "Bed Utilization Rate Median Of All": "float64",
    "Bed Utilization Rate Sum Of All": "object",
    "Case Mix Index Divided By Sum Of All": "float64",
    "Case Mix Index Median All": "float64",
    "Case Mix Index Sum All": "object",
    "COPD Rate Divided By Sum Of All": "float64",
    "COPD Rate Median All": "float64",
    "COPD Rate Sum All": "float64",
    "Est. IT Capital Budget Divided By Sum Of All": "float64",
    "Est. IT Capital Budget Median Of All": "int64",
    "Est. IT Capital Budget Sum Of All": "object",
    "HCAHPS Summary Rating Divided By Sum Of All": "float64",
    "HCAHPS Summary Rating Median All": "int64",
    "HCAHPS Summary Rating Sum All": "object",
    "Pneumonia Readmission Cases Divided By Sum Of All": "float64",
    "Pneumonia Readmission Cases Median Of All": "float64",
    "Pneumonia Readmission Cases Sum Of All": "float64",
    "Pneumonia Readmission Rate Divided By Sum Of All": "float64",
    "Pneumonia Readmission Rate Median Of All": "float64",
    "Pneumonia Readmission Rate Sum Of All": "float64",
    "Pressure Sores Rate Divided By Sum Of All": "float64",
    "Pressure Sores Rate Median All": "float64",
    "Pressure Sores Rate Sum All": "float64",
    "Sum Operational Factor Results": "float64",
    "Sum Clinical Factor Results": "float64",
}

# Define features names for uniformity
DD_rename = {
    "Definitive ID": definitive_id,
    "Hospital Name": "customer_name",
    "# of Discharges": "No. of Discharges",
    "# of Staffed Beds": "No. of Staffed Beds",
    "Est # of ER Visits": "Est No. of ER Visits",
    "Est # of Inpatient Surgeries": "Est No. of Inpatient Surgeries",
    "Est # of Outpatient Surgeries": "Est No. of Outpatient Surgeries",
    "Est # of Outpatient Visits": "Est No. of Outpatient Visits",
    "Est # of Total Surgeries": "Est No. of Total Surgeries",
    "# of Discharges Divided By Sum Of All": "No. of Discharges Divided By Sum Of All",
    "# of Discharges Median All": "No. of Discharges Median All",
    "# of Discharges Sum All": "No. of Discharges Sum All",
}

# Remove commas from string and conver to float
EDD["# of Discharges"] = EDD["# of Discharges"].astype("str").apply(assign)
EDD["# of Staffed Beds"] = EDD["# of Staffed Beds"].astype("str").apply(assign)

#Use the helper functions to clean the External Definitve Data 
convert_dtype(EDD, DD_convert)
EDD.rename(columns=DD_rename, inplace=True)

# Fill "empty" cells with nan
replace_na(EDD)

# Save cleaned data
utils.save_parquet(EDD, f"{clean_data_path}/external_definitive_data.parquet")

# display(EDD)