# Databricks notebook source
# MAGIC %md
# MAGIC ### Building the Recommendation Model
# MAGIC This notebook uses the collaborative matrices created in the previous notebook to create the recommendation system. The custom recommendaton system is tailor built to meet various business requirements.

# COMMAND ----------

# DBTITLE 1,Import libraries
# utils is a python script loaded from the Utility_functions folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration settings from the configuration file
config = utils.read_config()

# Extract the value associated with the "data_version" key from the configuration
data_version = config["data_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Data paths
#path for propensity data based on the data version
propensity_data_path = f"DATA_VERSIONS/{data_version}/PROPENSITY_DATA"

#path for data before cutoffbased on the data version
before_cutoff_path = f"DATA_VERSIONS/{data_version}/BEFORE_CUTOFF_RAW"

#path for data after cutoff  based on the data version
after_cutoff_path = f"DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

#path for data related to recommendation (eg: collaborative matrices) based on the data version
recommendation_data_path = f"DATA_VERSIONS/{data_version}/RECOMMENDATION_DATA"

#path for the stats.json file
json_path = f"DATA_VERSIONS/{data_version}"

# COMMAND ----------

# DBTITLE 1,Load files
#load curated product list
CPL = utils.read_parquet(f"{before_cutoff_path}/curated_products_data.parquet")

#load Capital Sales Data before cutoff
BEFORE_CUTOFF_CSD = utils.read_parquet(f"{before_cutoff_path}/customer_sales_data.parquet")

#load curated product list
AFTER_CUTOFF_CSD = utils.read_parquet(f'{after_cutoff_path}/customer_sales_data.parquet')

#load propensity model
propensity_model = utils.read_pkl(f"{propensity_data_path}/propensity_model.pkl")

#load collaborative matrices
cf_matrix = utils.read_matrix(f"{recommendation_data_path}/CF_MATRIX")

# COMMAND ----------

# DBTITLE 1,Data that maps Products with Services
service = [
    ["NAVICARE PATIENT SAFETY", "NC MAINTENANCE"],
    ["NAVICARE PATIENT FLOW", "NC MAINTENANCE"],
    ["NC MAINTENANCE", "NC MAINTENANCE"],
    ["VOALTE HARDWARE", "VOALTE PROFESSIONAL SERVICES"],
    ["VOALTE HARDWARE", "VOALTE MAINTENANCE & SUPPORT"],
    ["VOALTE HARDWARE", "VOALTE THIRD PARTY"],
    ["VOALTE PROFESSIONAL SERVICES", "VOALTE PROFESSIONAL SERVICES"],
    ["VOALTE MAINTENANCE & SUPPORT", "VOALTE MAINTENANCE & SUPPORT"],
    ["VOALTE THIRD PARTY", "VOALTE THIRD PARTY"]
]
service = pd.DataFrame(service,columns=["product","service"])

# COMMAND ----------

# DBTITLE 1,Create Recommendation System
# Build a recommendation model using the collaborative matrices created in the last notebook. This recommendation system is custom built as per business requirements.


class CustomRecommendationModel:
    def __init__(self, cf_matrix, propensity_model, BEFORE_CUTOFF_CSD, CPL, service):
        """
        Initialize an instance of the class with the provided parameters.

        Args:
            cf_matrix (dict): The collaborative filtering matrix
            propensity_model: The propensity model
            BEFORE_CUTOFF_CSD: The before cutoff CSD data
            CPL: Curated product list.
            service: Product-Service data
        """
        self.cf_matrix = cf_matrix
        self.propensity_model = propensity_model
        self.BEFORE_CUTOFF_CSD = BEFORE_CUTOFF_CSD
        self.CPL = CPL
        self.service = service

    def recommend(
        self, customer_num, top=0, remark=True, replacement=True, curated=True
    ):
        """
        Recommend products for a given customer

        Args:
            customer_num: The customer number for whom recommendations are generated
            top (int): The number of top recommendations to return (default is 0, which returns all recommendations)
            remark (bool): Whether to include propensity model remarks in the recommendations (default is True)
            replacement (bool): Whether to include recommended replacement products (default is True)
            curated (bool): Whether to consider recommendations from the customer propensity list (default is True)

        Returns:
            DataFrame: A DataFrame containing the recommended products, ratings, outcomes, and imputed values.
        """

        cust_matrix = None
        for cluster, matrix in self.cf_matrix.items():
            if customer_num in matrix.index:
                cust_matrix = matrix.copy()
        if cust_matrix is not None:
            temp = pd.DataFrame(
                cust_matrix.loc[customer_num, :].sort_values(ascending=False)
            ).rename(columns={customer_num: "rating"}).reset_index()

            if curated:
                temp = temp[temp["srp_2_desc"].isin(CPL["srp_2_desc"])].copy()

            if replacement:
                temp = temp.merge(
                    self.CPL[["srp_2_desc", "updated"]], on="srp_2_desc", how="left"
                )
                temp["srp_2_desc"] = temp["updated"].fillna(temp["srp_2_desc"])
                temp = (
                    temp.groupby("srp_2_desc")[["rating"]]
                    .agg(
                        {
                            "rating": "mean",
                        }
                    )
                    .reset_index()
                )

            if remark:
                propensity_outcome = self.propensity_model.predict_from_lists(
                    [customer_num], temp["srp_2_desc"]
                )
                temp = temp.merge(
                    propensity_outcome[["srp_2_desc", "outcome", "imputed"]],
                    on="srp_2_desc",
                    how="left",
                )
            else:
                temp[["outcome", "imputed"]] = [[-1, -1] for _ in range(len(temp))]
                temp.reset_index(inplace=True)

            temp = temp.sort_values(
                ["rating", "outcome","imputed"], ascending=[False, False, True]
            ).reset_index(drop=True)
            temp = temp[["srp_2_desc","rating","outcome","imputed"]]
            temp_a = set(
                self.BEFORE_CUTOFF_CSD[
                    self.BEFORE_CUTOFF_CSD["customer_num"] == customer_num
                ]["srp_2_desc"]
            )
            temp_b = set(self.service["product"])
            if len(temp_b.difference(temp_a)) > 0:
                drop_services = self.service[
                    self.service["product"].isin(temp_b.difference(temp_a))
                ]["service"].unique()
                temp = temp[~(temp.index.isin(drop_services))]

        else:

            temp = pd.DataFrame(columns=["srp_2_desc", "rating", "outcome", "imputed"])

        if (top == 0) or (len(temp) < top):
            return temp
        else:
            return temp.head(top)

    def recommend_with_remark(
        self, customer_num, top=0, remark=True, replacement=True, curated=True
    ):
        """
        Recommend products for a given customer with remarks and history labels.
        Args:
        customer_num: The customer number for whom recommendations are generated
        top (int): The number of top recommendations to return (default is 0, which returns all recommendations)
        remark (bool): Whether to include propensity model remarks in the recommendations (default is True)
        replacement (bool): Whether to include recommended replacement products (default is True)
        curated (bool): Whether to consider recommendations from the customer propensity list (default is True)

        Returns:
        DataFrame: A DataFrame containing the recommended products, ratings, remarks, and history labels
        """
        output = self.recommend(customer_num, top, remark, replacement, curated)
        output["outcome"] = output["outcome"].replace(
            {
                1: "Near Sale",
                0: "Far Sale",
                -1: "No Remark (Unknown Customer)",
                -2: "No Remark (Unknown Product)",
                -3: "No Remark (Unknown Customer and Product)",
            }
        )
        output["imputed"] = output["imputed"].apply(
            lambda x: "Known Pair" if x == 0 else "New Pair"
        )
        output.rename(
            columns={
                "srp_2_desc": "Product",
                "rating": "Rating",
                "outcome": "Remark",
                "imputed": "History",
            },
            inplace=True,
        )
        return output

    def mrr_score(self, X):
        """
        Calculate the Mean Reciprocal Rank (MRR) score for a set of recommendations

        Args:
            X (DataFrame): Input data containing customer numbers and product descriptions

        Returns:
            DataFrame: A DataFrame with MRR scores, reciprocal ranks, and product locations
        """
        data = X[["customer_num", "srp_2_desc"]].copy()
        concat_list = []
        for cluster in self.cf_matrix.keys():
            temp = self.cf_matrix[cluster].rank(axis=1, ascending=False)
            temp = pd.DataFrame(temp.unstack().reset_index()).rename(
                columns={"level_1": "customer_num", 0: "rank"}
            )
            temp["customer_num"] = temp["customer_num"].astype("object")
            concat_list.append(
                data.merge(
                    temp[["customer_num", "srp_2_desc", "rank"]],
                    on=["customer_num", "srp_2_desc"],
                    how="left",
                )
            )

        data = pd.concat(concat_list, axis=0).dropna()
        data["reciprocal_rank"] = 1 / data["rank"]
        data = data.merge(
            self.CPL[["srp_2_desc", "location"]], on="srp_2_desc", how="left"
        ).fillna("Others")
        return data

    def precision_recall_score(self, X, top_k=10):
        """
        Calculate precision and recall scores for a set of recommendations.

        Args:
            X (DataFrame): Input data containing customer numbers and product descriptions.
            top_k (int): Number of top recommendations considered for precision and recall calculation (default is 10).

        Returns:
            dict: A dictionary containing the average precision and recall scores.
        """
        data = self.mrr_score(X)
        pr_df = (
            data.groupby(["customer_num", "srp_2_desc"])[["rank"]].first().reset_index()
        )
        pr_df["rank"] = pr_df["rank"].apply(lambda x: 1 if x <= top_k else 0)
        pr_df = (
            pr_df.groupby(["customer_num"])[["rank"]]
            .agg({"rank": ["sum", "count"]})
            .reset_index()
        )
        pr_df.columns = ["customer_num", "sum", "count"]
        pr_df["recall"] = pr_df["sum"] / pr_df["count"]
        pr_df["precision"] = pr_df["sum"] / top_k
        return {
            "precision": pr_df["precision"].mean(),
            "recall": pr_df["recall"].mean(),
        }

    def precision_score(self, X, top_k=10):
        """
        Calculate the precision score for a set of recommendations.

        Args:
            X (DataFrame): Input data containing customer numbers and product descriptions.
            top_k (int): Number of top recommendations considered (default is 10).

        Returns:
            float: Precision score.
        """
        return self.precision_recall_score(X, top_k)["precision"]

    def recall_score(self, X, top_k=10):
        """
        Calculate the recall score for a set of recommendations.

        Args:
            X (DataFrame): Input data containing customer numbers and product descriptions.
            top_k (int): Number of top recommendations considered (default is 10).

        Returns:
            float: Recall score.
        """
        return self.precision_recall_score(X, top_k)["recall"]

# COMMAND ----------

# DBTITLE 1,Create a recommendation object
recommendation_model = CustomRecommendationModel(cf_matrix, propensity_model,BEFORE_CUTOFF_CSD, CPL, service)

# COMMAND ----------

# DBTITLE 1,Save the recommendation system as a pickle file
utils.save_pkl(recommendation_model,f"{recommendation_data_path}/recommendation_model.pkl")

# COMMAND ----------

# DBTITLE 1,Sample example
recommendation_model.recommend("604934")

# COMMAND ----------

recommendation_model.recommend_with_remark("604934")

# COMMAND ----------

recommendation_model.mrr_score(AFTER_CUTOFF_CSD)

# COMMAND ----------

# DBTITLE 1,overall Precision and Recall Score on after cutoff data
recommendation_model.precision_score(AFTER_CUTOFF_CSD,10),recommendation_model.recall_score(AFTER_CUTOFF_CSD,10)

# COMMAND ----------

ALS_test = recommendation_model.mrr_score(AFTER_CUTOFF_CSD)

# COMMAND ----------

# DBTITLE 1,Overall MRR on after cutoff data
overall_mrr = ALS_test['reciprocal_rank'].mean()
total_customer_product_pairs = len(ALS_test)
print(f'The overall MRR score for {total_customer_product_pairs} customer-product pairs is {overall_mrr}')

# COMMAND ----------

# DBTITLE 1,Plot of Rank Vs No of purchases
plt.rcParams['legend.fontsize'] = 20
plt.figure(figsize=(30,10))
plot_df = ALS_test.sort_values("rank")
sns.histplot(ALS_test,x="rank",hue="location",bins=len(set(ALS_test["rank"])),multiple="stack")
plt.xlabel("Rank",fontsize=20 )
plt.ylabel("No. of purchases",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

# COMMAND ----------

# DBTITLE 1,Plot of MRR Vs Location
plt.figure(figsize=(10,10))
plot_df = ALS_test.groupby("location")[["reciprocal_rank"]].mean().sort_values("reciprocal_rank",ascending=False)
sns.barplot(x=plot_df.index,y=plot_df["reciprocal_rank"])
plt.xlabel("Location")
plt.ylabel("MRR")

# COMMAND ----------

plt.figure(figsize=(10,10))
plot_df = ALS_test.groupby("location")[["rank"]].mean().sort_values("rank",ascending=True)
sns.barplot(x=plot_df.index,y=plot_df["rank"])
plt.xlabel("Location")
plt.ylabel("Rank")

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.histplot(ALS_test["reciprocal_rank"],bins=10,kde=True)
plt.xlabel("MRR")
plt.ylabel("No. of Purchase Events")

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.histplot(ALS_test.groupby("customer_num")[["reciprocal_rank"]].mean(),bins=10,kde=True)
plt.xlabel("MRR")
plt.ylabel("No. of Customers")

# COMMAND ----------

plt.figure(figsize=(20,20))
plot_df = ALS_test.groupby("srp_2_desc")[["reciprocal_rank"]].mean().sort_values("reciprocal_rank",ascending=False)
sns.barplot(y=plot_df.index,x=plot_df["reciprocal_rank"],orient="h")
plt.xlabel("MRR")
plt.ylabel("SRP_2_DESC")

# COMMAND ----------

stats = {}

stats["over all mrr"] = {"value":overall_mrr,"description":"Mean reciprocal rank for purchase events after cutoff"}
utils.customer_product_stats(stats,ALS_test,"recommendation test")

utils.save_stats(stats)

# COMMAND ----------

utils.save_json(f"{json_path}/stats.json","/Workspace/Users/davide@baxter.com/Solution/stats.json")
utils.save_json(f"{json_path}/config.json","/Workspace/Users/davide@baxter.com/Solution/config.json")