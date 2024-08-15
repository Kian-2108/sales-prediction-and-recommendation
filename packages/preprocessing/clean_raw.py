import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from ..utils import utils

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

CSD_path = config["CSD_path"]["value"]
SFD_path = config["SFD_path"]["value"]
RSD_path = config["RSD_path"]["value"]
CWD_path = config["CWD_path"]["value"]
CBD_path = config["CBD_path"]["value"]
EDD_path = config["EDD_path"]["value"]
CPL_path = config["CPL_path"]["value"]
UPL_path = config["UPL_path"]["value"]

CSD = utils.read_table(CSD_path)
SFD = utils.read_table(SFD_path)
RSD = utils.read_table(RSD_path)
CWD = utils.read_table(CWD_path)
CBD = utils.read_table(CBD_path)
EDD = utils.read_table(EDD_path)
CPL = utils.read_table(CPL_path)
UPL = utils.read_table(UPL_path)

clean_data_path = f"./data/DATA_VERSIONS/{data_version}/CLEAN_RAW_DATA"
for dirs in [clean_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

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

# def convert_dtype(data, convert):
#     for col, col_type in convert.items():
#         data[col] = data[col].astype(col_type)

null_values = [
    ".",
    "~",
    "-",
    "None",
    "none",
    "NONE",
    "nan",
    "NaN",
    "NAN", 
    "*BLANK*",
    "",
    " ",
    "null",
    "NULL",
    "Null"
    ]

def replace_na(df, l=null_values):
    for c in l:
        df.replace(c, np.nan, inplace=True)

def fix_floats(x):
    if x == "nan":
        return np.nan
    else:
        replaced = x.replace(",", "").replace("%", "")
        return replaced
    
def convert_dtype(data,convert):
    for col,col_type in convert.items():
        if col_type == "float64":
            data[col] = data[col].astype('str').apply(fix_floats)
        # elif col_type == "datetime64[ns]":
        #     data[col] = datetime.strptime(str(data[col]), '%d/%m/%Y %H:%M:%S%p')
        else:
            pass
        data[col] = data[col].astype(col_type)

CL_convert = {
    "SRP2 - Description": "str",
    "SRP2 Code": "str",
    "Product Type": "str",
    "ACTIVE": "str",
    "REPLACEMENT": "str",
    "Replacement SRP2": "str",
    "LOCATION": "str",
}

CL_columns = {
    "SRP2 - Description": "code_" + srp_2_desc,
    "SRP2 Code": srp_2_code,
    "Product Type": "product_type",
    "ACTIVE": "active",
    "REPLACEMENT": "replacement",
    "Replacement SRP2": "replacement_srp_2_code",
    "LOCATION": "location",
}

convert_dtype(CPL, CL_convert)
CPL.rename(columns=CL_columns, inplace=True)
CPL["replacement_srp_2_code"] = CPL["replacement_srp_2_code"].astype("float").fillna(0).astype("int").astype("str").replace("0",np.nan)
CPL["replacement_srp_2_code"] = CPL["replacement_srp_2_code"].fillna(CPL["srp_2_code"])
CPL["srp_2_desc"] = CPL["code_srp_2_desc"].apply(lambda x: x[4:])
CPL["srp_2_replacement"] = CPL["replacement"].apply(lambda x: x[4:] if x != None else x)
CPL["updated"] = CPL["srp_2_replacement"].where(CPL["srp_2_replacement"] != "", CPL["srp_2_desc"])
replace_na(CPL)

CPL.to_parquet(f"{clean_data_path}/curated_products_data.parquet")


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

UL_drop = ["Column1"]
UPL.drop(UL_drop, axis=1, inplace=True)
convert_dtype(UPL, UL_convert)
UPL.rename(columns=UL_columns, inplace=True)
replace_na(UPL)

UPL.to_parquet(f"{clean_data_path}/unfiltered_products_data.parquet")

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

convert_dtype(RSD, RS_convert)
RSD.rename(columns=RS_columns, inplace=True)
replace_na(RSD)

RSD.to_parquet(f"{clean_data_path}/rental_sales_data.parquet")

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

SF_drop = ["revenue_type", "END"]

SF_columns = {
    "jde_number": customer_num,
    "createddate": "create_dt",
    "closedate": "close_dt",
    "srp_1_description": srp_1_desc,
    "srp_2_description": srp_2_desc,
    "srp_3_description": srp_3_desc,
}

SFD.drop(SF_drop, axis=1, inplace=True)
convert_dtype(SFD, SF_convert)
SFD.rename(columns=SF_columns, inplace=True)
replace_na(SFD)

SFD.to_parquet(f"{clean_data_path}/sales_force_data.parquet")

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

CB_drop = ["ship_to_yn.1", "active_cust_yn.1", "END"]

CB_columns = {
    "customer_nm": customer_name,
    "customer_num": customer_num,
}

CBD.drop(CB_drop, axis=1, inplace=True)
convert_dtype(CBD, CB_convert)
CBD.rename(columns=CB_columns, inplace=True)
replace_na(CBD)

CBD.to_parquet(f"{clean_data_path}/customer_baseline_data.parquet")


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

CW_drop = ["definitive_id_alpha", "jde_id_alpha", "END"]

CW_columns = {
    "definitive_id_number": definitive_id,
    "jde_id_number": customer_num,
    "name": customer_name,
    "total_staffed_beds": total_staffed_beds,
}

CWD.drop(CW_drop, axis=1, inplace=True)
convert_dtype(CWD, CW_convert)
CWD.rename(columns=CW_columns, inplace=True)
replace_na(CWD)

CWD.to_parquet(f"{clean_data_path}/customer_crosswalk_data.parquet")

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

CS_drop = ["END"]

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

CSD.replace("2717-04-03", "2017-04-03", inplace=True)
CSD.replace("2716-04-20", "2016-04-20", inplace=True)
CSD.replace("2717-10-02", "2017-10-02", inplace=True)
convert_dtype(CSD, CS_convert)
CSD.drop(CS_drop, axis=1, inplace=True)
CSD.rename(columns=CS_columns, inplace=True)
replace_na(CSD)

CSD.to_parquet(f"{clean_data_path}/customer_sales_data.parquet")


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

EDD["# of Discharges"] = EDD["# of Discharges"].astype("str").apply(fix_floats)
EDD["# of Staffed Beds"] = EDD["# of Staffed Beds"].astype("str").apply(fix_floats)

convert_dtype(EDD, DD_convert)
EDD.rename(columns=DD_rename, inplace=True)
replace_na(EDD)

EDD.to_parquet(f"{clean_data_path}/external_definitive_data.parquet")
