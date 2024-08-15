from .. utils import utils
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

cutoff_date = pd.to_datetime(config["cutoff_date"]["value"])
max_days_between_purchase = config["max_days_between_purchase"]["value"]
near_sale_cutoff = config["near_sale_cutoff"]["value"]

before_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"
after_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"
propensity_data_path = f"./data/DATA_VERSIONS/{data_version}/PROPENSITY_DATA"

for dirs in [before_cutoff_path,after_cutoff_path,propensity_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

BEFORE_CUTOFF_CSD = pd.read_parquet(f"{before_cutoff_path}/customer_sales_data.parquet")
AFTER_CUTOFF_CSD = pd.read_parquet(f"{after_cutoff_path}/customer_sales_data.parquet")

def grouped(data):
    grouped_data = data.groupby(["customer_num","srp_2_desc","dtl_invoice_num"]).first().reset_index()
    data_non_single = []
    for col in grouped_data.columns:
        if len(grouped_data[col].unique())>1:
            data_non_single.append(col)
    grouped_data = grouped_data[data_non_single].sort_values(['customer_num','srp_2_desc','dtl_invoice_dt'])
    return grouped_data

BeforeCutoffCSD = grouped(BEFORE_CUTOFF_CSD)
AfterCutoffCSD = grouped(AFTER_CUTOFF_CSD)

NextPurchase = AfterCutoffCSD.groupby(['customer_num','srp_2_desc']).dtl_invoice_dt.min().reset_index()
NextPurchase.columns = ['customer_num','srp_2_desc','NextPurchaseAfterCutoff']
NextPurchase

LastPurchase = BeforeCutoffCSD.groupby(['customer_num','srp_2_desc']).dtl_invoice_dt.max().reset_index()
LastPurchase.columns = ['customer_num','srp_2_desc','LastPurchaseBeforeCutoff']
LastPurchase

PurchaseDatesDF = pd.merge(LastPurchase,NextPurchase,on=['customer_num','srp_2_desc'], how='left')
PurchaseDatesDF['DaysBetweenLastTwoPurchases'] = (PurchaseDatesDF['NextPurchaseAfterCutoff']-PurchaseDatesDF['LastPurchaseBeforeCutoff']).dt.days
PurchaseDatesDF['LastPurchaseToCutoff'] = (cutoff_date-PurchaseDatesDF['LastPurchaseBeforeCutoff']).dt.days

PurchaseDatesDF['DaysBetweenLastTwoPurchases'].fillna(max_days_between_purchase,inplace=True)
PurchaseDatesDF['CutoffToNextPurchase'] = (PurchaseDatesDF['DaysBetweenLastTwoPurchases']-PurchaseDatesDF['LastPurchaseToCutoff'])
PurchaseDatesDF

no_of_invoice = 5
for i in range(1,no_of_invoice+1):
    BeforeCutoffCSD[f"#{i}InvoiceDate"] = BeforeCutoffCSD.groupby(['customer_num','srp_2_desc'],as_index=False)['dtl_invoice_dt'].shift(i)

BeforeCutoffCSD[['customer_num','srp_2_desc','dtl_invoice_dt']+[f"#{i}InvoiceDate" for i in range(1,no_of_invoice+1)]]

BeforeCutoffCSD[f"DayDiff1"] = (BeforeCutoffCSD[f"dtl_invoice_dt"] - BeforeCutoffCSD[f"#1InvoiceDate"]).dt.days
for i in range(2,no_of_invoice+1):

    BeforeCutoffCSD[f"DayDiff{i}"] = (BeforeCutoffCSD[f"#{i-1}InvoiceDate"] - BeforeCutoffCSD[f"#{i}InvoiceDate"]).dt.days
BeforeCutoffCSD[
    ["customer_num", "srp_2_desc", "dtl_invoice_dt"]
    + [f"#{i}InvoiceDate" for i in range(1, no_of_invoice + 1)]
    + [f"DayDiff{i}" for i in range(1, no_of_invoice)]
]

sales_day_diff = (
    BeforeCutoffCSD.groupby(["customer_num", "srp_2_desc"])
    .agg({"DayDiff1": ["mean", "std"]})
    .reset_index()
)
sales_day_diff.columns = ["customer_num", "srp_2_desc", "DayDiffMean", "DayDiffStd"]
sales_day_diff

BeforeCutoffCSD2 = BeforeCutoffCSD.drop_duplicates(subset=['customer_num','srp_2_desc'],keep='last')

BeforeCutoffCSD4ML = BeforeCutoffCSD2.copy()

BeforeCutoffCSD4ML = pd.merge(BeforeCutoffCSD4ML, sales_day_diff, on=['customer_num','srp_2_desc'])

BeforeCutoffCSD4ML = pd.merge(BeforeCutoffCSD4ML, PurchaseDatesDF, on=['customer_num','srp_2_desc'])
BeforeCutoffCSD4ML

BeforeCutoffCSD4ML["outcome"] = BeforeCutoffCSD4ML["CutoffToNextPurchase"].apply(lambda x: 1 if x <= near_sale_cutoff else 0)

BeforeCutoffCSD4ML[BeforeCutoffCSD4ML["CutoffToNextPurchase"]<= near_sale_cutoff]

Corr_data = BeforeCutoffCSD4ML.copy()
for col in Corr_data.select_dtypes(["object","datetime64[ns]"]).columns:
    Corr_data[col] = pd.factorize(Corr_data[col])[0]
# abs(Corr_data.corr()).style.background_gradient()

plt.figure(figsize=(50,40))
plt.rcParams.update({'font.size': 20})
sns.heatmap(abs(Corr_data.corr()),annot=False)

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


Corr_data = BeforeCutoffCSD4ML[LimitedCols].copy()
for col in Corr_data.select_dtypes(["object","datetime64[ns]"]).columns:
    Corr_data[col] = pd.factorize(Corr_data[col])[0]
# abs(Corr_data.corr()).style.background_gradient()

plt.figure(figsize=(50,40))
plt.rcParams.update({'font.size': 30})
sns.heatmap(abs(Corr_data.corr()),annot=False)

BeforeCutoffCSD4ML = BeforeCutoffCSD4ML[LimitedCols].dropna()

AfterCutoffCSD4ML = AfterCutoffCSD.drop_duplicates(subset=["customer_num","srp_2_desc"],keep="first")
AfterCutoffCSD4ML["outcome"] = AfterCutoffCSD4ML["dtl_invoice_dt"].apply(lambda x: 1 if x<=cutoff_date+pd.Timedelta(near_sale_cutoff,unit="d") else 0)
AfterCutoffCSD4ML = AfterCutoffCSD4ML[["customer_num","srp_2_desc","outcome"]]

BeforeCutoffCSD4ML.to_parquet(f"{propensity_data_path}/propensity_data_train.parquet")
AfterCutoffCSD4ML.to_parquet(f"{propensity_data_path}/propensity_data_test.parquet")