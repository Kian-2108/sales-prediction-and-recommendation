from ..utils import utils
import os
import numpy as np
import pandas as pd

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

before_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"
after_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"
output_data_path = f"./data/DATA_VERSIONS/{data_version}/PREDICTION_DATA"

BEFORE_CUTOFF_CSD = pd.read_parquet(f"{before_cutoff_path}/customer_sales_data.parquet")
BEFORE_CUTOFF_SFD = pd.read_parquet(f"{before_cutoff_path}/sales_force_data.parquet")
AFTER_CUTOFF_CSD = pd.read_parquet(f"{after_cutoff_path}/customer_sales_data.parquet")
AFTER_CUTOFF_SFD = pd.read_parquet(f"{after_cutoff_path}/sales_force_data.parquet")

for dirs in [before_cutoff_path,after_cutoff_path,output_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

def fill_quadrants(input_CSD, input_SFD):
    grouped_SFD = (
        input_SFD.groupby(["customer_num", "srp_2_desc", "opp_id"])
        .first()
        .reset_index()
    )
    SFD_non_single = []
    for col in grouped_SFD.columns:
        if len(grouped_SFD[col].unique()) > 1:
            SFD_non_single.append(col)

    grouped_SFD = grouped_SFD[list(set(SFD_non_single + ["cutoff_tag"]))]

    grouped_CSD = (
        input_CSD.groupby(["customer_num", "srp_2_desc", "dtl_invoice_num"])
        .first()
        .reset_index()
    )
    CSD_non_single = []
    for col in grouped_CSD.columns:
        if len(grouped_CSD[col].unique()) > 1:
            CSD_non_single.append(col)

    grouped_CSD = grouped_CSD[list(set(CSD_non_single + ["cutoff_tag"]))]

    top_L_data = grouped_CSD
    bot_R_data = grouped_SFD[grouped_SFD["iswon"] == "False"]
    tem_R_data = grouped_SFD[grouped_SFD["iswon"] == "True"]

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
    df = df[df["delta"] <= 90].reset_index(drop=True)
    df.drop("delta", axis=1, inplace=True)

    top_data = pd.concat([top_data, df], axis=0).drop_duplicates(
        subset=["customer_num", "srp_2_desc", "dtl_invoice_num"], keep="last"
    )

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

BEFORE_CUTOFF_CSD["cutoff_tag"] = ["BEFORE" for i in range(len(BEFORE_CUTOFF_CSD))] 
BEFORE_CUTOFF_SFD["cutoff_tag"] = ["BEFORE" for i in range(len(BEFORE_CUTOFF_SFD))] 
AFTER_CUTOFF_CSD["cutoff_tag"] = ["AFTER" for i in range(len(AFTER_CUTOFF_CSD))] 
AFTER_CUTOFF_SFD["cutoff_tag"] = ["AFTER" for i in range(len(AFTER_CUTOFF_SFD))]

TOTAL_CSD = pd.concat([BEFORE_CUTOFF_CSD,AFTER_CUTOFF_CSD],axis=0).reset_index(drop=True)
TOTAL_SFD = pd.concat([BEFORE_CUTOFF_SFD,AFTER_CUTOFF_SFD],axis=0).reset_index(drop=True)

before_cutoff_prediction_data = fill_quadrants(BEFORE_CUTOFF_CSD,BEFORE_CUTOFF_SFD)
after_cutoff_prediction_data = fill_quadrants(TOTAL_CSD,TOTAL_SFD)

before_cutoff_prediction_data.drop("cutoff_tag",axis=1,inplace=True)
after_cutoff_prediction_data = after_cutoff_prediction_data[after_cutoff_prediction_data["cutoff_tag"]=="AFTER"].drop("cutoff_tag",axis=1)

before_cutoff_prediction_data.to_parquet(f'{output_data_path}/prediction_data_train.parquet')
after_cutoff_prediction_data.to_parquet(f'{output_data_path}/prediction_data_test.parquet')

stats = {}

utils.customer_product_stats(stats,before_cutoff_prediction_data,"prediction train")
utils.customer_product_stats(stats,after_cutoff_prediction_data,"prediction test")

utils.save_stats(stats)