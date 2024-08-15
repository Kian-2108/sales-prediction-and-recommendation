from ..utils import utils
import pandas as pd
import os

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

raw_data_path = f"./data/DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"

customer_data_path = f"./data/DATA_VERSIONS/{data_version}/CUSTOMER_DATA"
for dirs in [raw_data_path,customer_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

CSD = pd.read_parquet(f"{raw_data_path}/customer_sales_data.parquet")
SFD = pd.read_parquet(f"{raw_data_path}/sales_force_data.parquet")
RSD = pd.read_parquet(f"{raw_data_path}/rental_sales_data.parquet")
CBD = pd.read_parquet(f"{raw_data_path}/customer_baseline_data.parquet")
CWD = pd.read_parquet(f"{raw_data_path}/customer_crosswalk_data.parquet")
EDD = pd.read_parquet(f"{raw_data_path}/external_definitive_data.parquet")
CPL = pd.read_parquet(f"{raw_data_path}/curated_products_data.parquet")
UPL = pd.read_parquet(f"{raw_data_path}/unfiltered_products_data.parquet")

IN_EDD = EDD[["definitive_id", "customer_name"]].groupby("definitive_id").first().reset_index()
IN_CWD = CWD[["definitive_id", "customer_num", "customer_name"]].groupby("customer_num").first().reset_index()
IN_CSD = CSD[["customer_num", "customer_name"]].groupby("customer_num").first().reset_index()
IN_RSD = RSD[["customer_num", "customer_name"]].groupby("customer_num").first().reset_index()
IN_SFD = SFD[["customer_num"]].groupby("customer_num").first().reset_index()
IN_CBD = CBD[["customer_num", "customer_name"]].groupby("customer_num").first().reset_index()

all_customer_tables = {"IN_EDD":IN_EDD,"IN_CWD":IN_CWD,"IN_CSD":IN_CSD,"IN_RSD":IN_RSD,"IN_SFD":IN_SFD,"IN_CBD":IN_CBD}

for col,table in all_customer_tables.items():
    table.fillna("",inplace=True)
    table[col] = [True for i in range(len(table))]

customers = IN_CWD[["definitive_id","customer_num","IN_CWD"]].merge(IN_EDD[["definitive_id","IN_EDD"]],on="definitive_id",how="outer")
customers = customers.merge(IN_CSD[["customer_num","IN_CSD"]],on="customer_num",how="outer")
customers = customers.merge(IN_SFD[["customer_num","IN_SFD"]],on="customer_num",how="outer")
customers = customers.merge(IN_RSD[["customer_num","IN_RSD"]],on="customer_num",how="outer")
customers = customers.merge(IN_CBD[["customer_num","IN_CBD"]],on="customer_num",how="outer")
customers[["definitive_id","customer_num"]] = customers[["definitive_id","customer_num"]].fillna("")

customers.fillna(False,inplace=True)

customers = customers.merge(IN_CBD.drop("IN_CBD",axis=1),on="customer_num",how="outer")

customers = customers.merge(IN_EDD.drop("IN_EDD",axis=1),on=["definitive_id"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

customers = customers.merge(IN_CSD.drop("IN_CSD",axis=1),on=["customer_num"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

customers = customers.merge(IN_CWD.drop(["IN_CWD","definitive_id"],axis=1),on=["customer_num"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

customers = customers.merge(IN_RSD.drop("IN_RSD",axis=1),on=["customer_num"],how="outer")
customers["customer_name_x"] = customers["customer_name_x"].fillna(customers["customer_name_y"])
customers = customers.rename(columns={"customer_name_x":"customer_name"}).drop("customer_name_y",axis=1)

customers["customer_name"] = customers["customer_name"].str.upper()

customers = customers[["customer_num","definitive_id","customer_name","IN_CBD","IN_CSD","IN_CWD","IN_EDD","IN_RSD","IN_SFD"]]

customer_details = customers[(customers["IN_CSD"]) | (customers["IN_SFD"]) | (customers["IN_EDD"])][["definitive_id", "customer_num", "customer_name"]]

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

temp_prod1 = (
    CSD[["customer_num", "srp_2_desc"]]
    .groupby("customer_num")["srp_2_desc"]
    .apply(list)
    .reset_index()
)
customer_details = customer_details.merge(temp_prod1, on="customer_num", how="left")
customer_details.rename(columns={"srp_2_desc": "sales_list"}, inplace=True)

temp_prod2 = (
    SFD[SFD["iswon"] == "True"][["customer_num", "srp_2_desc"]]
    .dropna()
    .groupby("customer_num")["srp_2_desc"]
    .apply(list)
    .reset_index()
)
customer_details = customer_details.merge(temp_prod2, on="customer_num", how="left")
customer_details.rename(columns={"srp_2_desc": "salesforce_list"}, inplace=True)

temp_prod3 = (
    RSD[["customer_num", "srp_2_desc"]]
    .groupby("customer_num")["srp_2_desc"]
    .apply(list)
    .reset_index()
)
customer_details = customer_details.merge(temp_prod3, on="customer_num", how="left")
customer_details.rename(columns={"srp_2_desc": "rental_list"}, inplace=True)

temp_prod4 = CSD.groupby("customer_num").agg(
    {"dtl_qty_ordered": "sum", "dtl_order_dt": ["min", "max"]}
)

temp_prod4["delta_days"] = (
    temp_prod4[("dtl_order_dt", "max")] - temp_prod4[("dtl_order_dt", "min")]
).dt.days

temp_prod4["purchases_per_year"] = (
    temp_prod4[("dtl_qty_ordered", "sum")] / temp_prod4["delta_days"] * 365
)

temp_prod4 = temp_prod4.reset_index()[["customer_num", "purchases_per_year"]]
temp_prod4.columns = ["customer_num", "purchase_per_year"]
customer_details = customer_details.merge(temp_prod4, on="customer_num", how="left")

customer_details["unique_id"] = customer_details["customer_num"] + "_" + customer_details["definitive_id"]

customer_details["customer_num"] = customer_details["customer_num"].where(customer_details["customer_num"]!="",customer_details["unique_id"])

customer_details.set_index("unique_id", inplace=True)

customers.to_parquet(f"{customer_data_path}/customer_interactions.parquet")
customer_details.to_parquet(f"{customer_data_path}/customer_details.parquet")

stats = {}
utils.customer_product_stats(stats,customer_details,"customer clustering")
utils.save_stats(stats)