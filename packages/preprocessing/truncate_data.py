from ..utils import utils
import pandas as pd
import os

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]
cutoff_date = pd.to_datetime(config["cutoff_date"]["value"])

raw_data_path = f"./data/DATA_VERSIONS/{data_version}/CLEAN_RAW_DATA"
before_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"
after_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

for dirs in [raw_data_path,before_cutoff_path,after_cutoff_path]:
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

SFD_before_cutoff = SFD[SFD["create_dt"] <= cutoff_date]
SFD_after_cutoff = SFD[SFD["create_dt"] > cutoff_date]

SFD_before_cutoff.to_parquet(f"{before_cutoff_path}/sales_force_data.parquet")
SFD_after_cutoff.to_parquet(f"{after_cutoff_path}/sales_force_data.parquet")

CSD_before_cutoff = CSD[CSD["dtl_invoice_dt"] <= cutoff_date]
CSD_after_cutoff = CSD[CSD["dtl_invoice_dt"] > cutoff_date]

CSD_before_cutoff.to_parquet(f"{before_cutoff_path}/customer_sales_data.parquet")
CSD_after_cutoff.to_parquet(f"{after_cutoff_path}/customer_sales_data.parquet")

RSD_before_cutoff = RSD[RSD["billing_from_dt"] <= cutoff_date]
RSD_after_cutoff = RSD[RSD["billing_from_dt"] > cutoff_date]

RSD_before_cutoff.to_parquet(f"{before_cutoff_path}/rental_sales_data.parquet")
RSD_after_cutoff.to_parquet(f"{after_cutoff_path}/rental_sales_data.parquet")

CBD.to_parquet(f"{before_cutoff_path}/customer_baseline_data.parquet")
CWD.to_parquet(f"{before_cutoff_path}/customer_crosswalk_data.parquet")
EDD.to_parquet(f"{before_cutoff_path}/external_definitive_data.parquet")
CPL.to_parquet(f"{before_cutoff_path}/curated_products_data.parquet")
UPL.to_parquet(f"{before_cutoff_path}/unfiltered_products_data.parquet")

CBD.to_parquet(f"{after_cutoff_path}/customer_baseline_data.parquet")
CWD.to_parquet(f"{after_cutoff_path}/customer_crosswalk_data.parquet")
EDD.to_parquet(f"{after_cutoff_path}/external_definitive_data.parquet")
CPL.to_parquet(f"{after_cutoff_path}/curated_products_data.parquet")
UPL.to_parquet(f"{after_cutoff_path}/unfiltered_products_data.parquet")

stats = {}

utils.customer_product_stats(stats,CSD,"customer sales")
utils.customer_product_stats(stats,SFD,"sales force")
utils.customer_product_stats(stats,RSD,"rental sales")
utils.customer_product_stats(stats,CBD,"customer baseline")
utils.customer_product_stats(stats,CWD,"customer crosswalk")
utils.customer_product_stats(stats,EDD,"external definitive")
utils.customer_product_stats(stats,CPL,"curated products")
utils.customer_product_stats(stats,UPL,"unfiltered products")

utils.customer_product_stats(stats,CSD_before_cutoff,"before cutoff customer sales")
utils.customer_product_stats(stats,SFD_before_cutoff,"before cutoff sales force")
utils.customer_product_stats(stats,RSD_before_cutoff,"before cutoff rental sales")

utils.customer_product_stats(stats,CSD_after_cutoff,"after cutoff customer sales")
utils.customer_product_stats(stats,SFD_after_cutoff,"after cutoff sales force")
utils.customer_product_stats(stats,RSD_after_cutoff,"after cutoff rental sales")

utils.save_stats(stats)