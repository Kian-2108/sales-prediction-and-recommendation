from ..utils import utils
import os
import pandas as pd
import numpy as np

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

raw_data_path = f"./data/DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"
product_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA"

for dirs in [raw_data_path,product_data_path]:
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

products.rename(
    columns={
        # "srp_2_code": "old_srp_2_code",
        # "srp_2_desc": "old_srp_2_desc",
        "replacement_srp_2_code": "new_srp_2_code",
        "updated": "new_srp_2_desc",
    },
    inplace=True,
)

products = products.merge(UPL[["srp_2_code","srp_2_desc","srp_3_desc"]].groupby(["srp_2_code","srp_2_desc"]).first().reset_index(),how="left",on=["srp_2_code","srp_2_desc"])

temp_prod1 = (
    CSD[["srp_2_code","srp_2_desc", "customer_num"]]
    .groupby(["srp_2_code","srp_2_desc"])["customer_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod1, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"customer_num": "sales_list"}, inplace=True)

temp_prod2 = (
    SFD[["srp_2_desc", "customer_num"]]
    .groupby("srp_2_desc")["customer_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod2, on="srp_2_desc", how="left")
products.rename(columns={"customer_num": "salesforce_list"}, inplace=True)

temp_prod3 = (
    RSD[["srp_2_code","srp_2_desc", "customer_num"]]
    .groupby(["srp_2_code","srp_2_desc"])["customer_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod3, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"customer_num": "rental_list"}, inplace=True)

temp_prod4 = (
    CSD[["srp_2_code","srp_2_desc", "dtl_order_num"]]
    .groupby(["srp_2_code","srp_2_desc"])["dtl_order_num"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod4, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"dtl_order_num": "order_list"}, inplace=True)

temp_prod6 = (
    CSD[["srp_2_code","srp_2_desc", "dtl_qty_ordered"]]
    .groupby(["srp_2_code","srp_2_desc"])["dtl_qty_ordered"]
    .mean()
    .reset_index()
)
products = products.merge(temp_prod6, on=["srp_2_code","srp_2_desc"], how="left")
products.rename(columns={"dtl_qty_ordered": "avg_order_amount"}, inplace=True)

temp_prod7 = CSD.groupby(["srp_2_code","srp_2_desc"]).agg({"dtl_qty_ordered":"sum","dtl_order_dt":["min","max"]})

temp_prod7["delta_days"]=(temp_prod7[("dtl_order_dt","max")]-temp_prod7[("dtl_order_dt","min")]).dt.days

temp_prod7["avg_sales_per_year"]=temp_prod7[("dtl_qty_ordered","sum")]/temp_prod7["delta_days"]*365

temp_prod7 = temp_prod7.reset_index()[["srp_2_code","srp_2_desc","avg_sales_per_year"]]
temp_prod7.columns = ["srp_2_code","srp_2_desc","avg_sales_per_year"]
products = products.merge(temp_prod7,on=["srp_2_code","srp_2_desc"], how="left")

temp_prod8 = (
    SFD[SFD["iswon"]=='True'][["srp_2_desc", "opp_id"]]
    .groupby("srp_2_desc")["opp_id"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod8, on="srp_2_desc", how="left")
products.rename(columns={"opp_id": "opp_won_list"}, inplace=True)

temp_prod9 = (
    SFD[SFD["iswon"]=='False'][["srp_2_desc", "opp_id"]]
    .groupby("srp_2_desc")["opp_id"]
    .apply(list)
    .reset_index()
)
products = products.merge(temp_prod9, on="srp_2_desc", how="left")
products.rename(columns={"opp_id": "opp_lost_list"}, inplace=True)

products.to_parquet(f"{product_data_path}/product_details.parquet")

stats = {}

utils.customer_product_stats(stats,products,"product clustering")

utils.save_stats(stats)