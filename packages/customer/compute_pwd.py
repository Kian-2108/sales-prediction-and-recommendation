import os
from ..utils import utils, cluster
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer

config = utils.read_json("./config.json")

data_version = config["data_version"]["value"]

customer_data_path = f"./data/DATA_VERSIONS/{data_version}/CUSTOMER_DATA"
pwd_data_path = f"./data/DATA_VERSIONS/{data_version}/CUSTOMER_DATA/PWD"

for dirs in [pwd_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

customers = pd.read_parquet(f"{customer_data_path}/customer_details.parquet")

distance_type = {
    "active_cust_yn"            : "matching",
    "customer_class"            : "matching",
    "customer_price_group"      : "matching",
    "customer_type"             : "matching",
    # "gl_acct_classification"    : "matching",
    "No. of Discharges"         : "euclidean",
    "No. of Staffed Beds"       : "euclidean",
    "340B Classification"       : "matching",
    "sales_list"                : "jaccard",
    "salesforce_list"           : "jaccard",
    "rental_list"               : "jaccard",
    "purchase_per_year"         : "euclidean",
    # "Latitude"                  : "euclidean",
    # "Longitude"                 : "euclidean",
    "Hospital Type"             : "matching",
    # "IDN"                       : "matching",
    # "IDN Parent"                : "matching",
    "Bed Utilization Rate"      : "euclidean",
    "Capital Expenditures"      : "euclidean",
    "Net Income"                : "euclidean",
}


for col,dist in distance_type.items():
    pd.DataFrame(cluster.pairwise(customers[col],dist),columns=[col]).to_parquet(f"{pwd_data_path}/{col}.parquet")