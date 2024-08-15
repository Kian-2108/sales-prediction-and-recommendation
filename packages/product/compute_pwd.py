import os
from ..utils import utils, cluster
import numpy as np
import pandas as pd

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

product_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA"
pwd_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA/PWD"


for dirs in [pwd_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass
    
products = pd.read_parquet(f"{product_data_path}/product_details.parquet")

distance_type = {
    "product_type"          : "matching",
    "location"              : "matching",
    # "srp_3_desc"          : "jaccard", # matching
    "sales_list"            : "jaccard",
    "salesforce_list"       : "jaccard",
    "rental_list"           : "jaccard",
    "order_list"            : "jaccard",
    "avg_order_amount"      : "euclidean",
    "avg_sales_per_year"    : "euclidean",
    "opp_won_list"          : "jaccard",
    "opp_lost_list"         : "jaccard",
}


for col,dist in distance_type.items():
    pd.DataFrame(cluster.pairwise(products[col],dist),columns=[col]).to_parquet(f"{pwd_data_path}/{col}.parquet")