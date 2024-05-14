# Databricks notebook source
# MAGIC %md
# MAGIC ### Cluster Customers
# MAGIC Using the pairwise distance and the optimal weights from TPE, agglomerative clustering is performed.

# COMMAND ----------

# DBTITLE 1,Load Libraries
# utility and cluster are python scripts imported from Utility_fuction folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import cluster
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
import optuna

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration file
config = utils.read_config()

# Extract the data version from the configuration file
data_version = config["data_version"]["value"]

# Extract the cutoff distance for forming the clusters
customer_clustering_cutoff_distance = config["customer_clustering_cutoff_distance"]["value"]

# xtract the boolean value which determines whether to run feature wrapping or not
run_customer_feature_wrapping = config["run_customer_feature_wrapping"]["value"]

# Extract the number of features to select after wrapping
customer_top_feature = config["customer_top_feature"]["value"]

# Extract the boolean value which determines whether to run weight tuning or not
run_customer_weight_tuning = config["run_customer_weight_tuning"]["value"]

# Extract the number of trials that need to be run
customer_weight_tuning_trials = config["customer_weight_tuning_trials"]["value"]

# Extract the data version tag to filter the experiment whose weights will be used
if run_customer_weight_tuning:
    customer_weight_tuning_version = data_version
else:
    customer_weight_tuning_version = config["customer_weight_tuning_version"]["value"]

# Extract the data version tag to filter the experiment whose features will be used
if run_customer_feature_wrapping:
    customer_feature_wrapping_version = data_version
else:
    customer_feature_wrapping_version = config["customer_feature_wrapping_version"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file paths
customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"

# COMMAND ----------

# DBTITLE 1,Load files
# Getting the set of all features for which the pairwise distance was computed
features = utils.read_filenames(f"{customer_data_path}/PWD")

# Locally save pwd arrays
for feature in features:
    with open(f"{feature}","wb") as npy:
        np.save(npy,utils.read_parquet(f"{customer_data_path}/PWD/{feature}.parquet")[feature].to_numpy())

# COMMAND ----------

# DBTITLE 1,Select features
# Set path to the mlflow experiment
customer_feature_wrapping_experiment = "/Users/davide@baxter.com/Solution/2A_Customer_Clustering/Feature-Wrapping"

# Optionally run feature wrapping
if run_customer_feature_wrapping:

    # Set the mlflow experiment to the above path
    mlflow.set_experiment(customer_feature_wrapping_experiment)

    selected_features = []
    while set(selected_features) != set(features):
        next_features = set(features.copy()) - set(selected_features.copy())
        wcss_dict = dict()
        cutoff_distance = customer_clustering_cutoff_distance
        for feature in next_features:
            with mlflow.start_run(run_name=str(datetime.now())):
                temp_features = selected_features.copy() + [feature]
                weights = {f: 1 for f in temp_features}
                wpwd = cluster.weighted_pairwise(weights)
                Z = linkage(wpwd,"ward")
                clusters = fcluster(Z,t=cutoff_distance,criterion="distance")
                wcss = cluster.WCSS(wpwd,clusters)
                wcss_dict[wcss] = feature

                tags = {"data_version":data_version,"timestamp":str(datetime.now()),"type":"wcss"}
                mlflow.set_tags(tags)
                mlflow.log_param("feature_set", temp_features)
                mlflow.log_param("feature_len", len(temp_features))
                mlflow.log_metric("WCSS", wcss)
                mlflow.log_metric("clusters", len(set(clusters)))
        selected_features.append(wcss_dict[min(wcss_dict)])

# COMMAND ----------

# DBTITLE 1,Connect to MLflow client to get selected feature
# Connecting to MLflow client
client = mlflow.MlflowClient()

# Getting the experiment from path specified in config
experiment_id = dict(mlflow.get_experiment_by_name(customer_feature_wrapping_experiment))["experiment_id"]

# Filetering TPE optimization runs with required tags and sorting them by WCSS 
runs = [
    run.info.run_id
    for run in client.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"tags.data_version = '{customer_feature_wrapping_version}'",
        order_by=[f"params.feature_len DESC"]
    )
]

# Extracting the parameters/ weights that performed the best
best_params = client.get_run(runs[0]).data.params.copy()["feature_set"]

selected_features = best_params.replace("[","").replace("]","").replace("'","").split(", ")
selected_features = selected_features[:customer_top_feature]

# COMMAND ----------

# DBTITLE 1,Weight optimization (Optional)
# Set path to the mlflow experiment
customer_weight_tuning_experiment = "/Users/davide@baxter.com/Solution/2A_Customer_Clustering/Feature-Weights"

# Optionally run weight optimization
if run_customer_weight_tuning:

    # Set the mlflow experiment to the above path
    mlflow.set_experiment(customer_weight_tuning_experiment)

    # Select sampler (search algorithm) for optimization
    sampler = optuna.samplers.TPESampler()

    # Define the objective function which need to be minimized
    def objective_WCSS(trial):
        with mlflow.start_run(run_name=str(datetime.now())):
            weights = {feature:trial.suggest_int(feature,1,10) for feature in selected_features}
            wpwd = cluster.weighted_pairwise(weights)
            cutoff_distance = customer_clustering_cutoff_distance
            Z = linkage(wpwd,"ward")
            clusters = fcluster(Z,t=cutoff_distance,criterion="distance")
            wcss = cluster.WCSS(wpwd,clusters)

            # Log parameters, metrics and tags to mlflow
            tags = {"data_version":data_version,"timestamp":str(datetime.now()),"search":str(sampler)}
            mlflow.set_tags(tags)
            mlflow.log_param("cutoff_distance",cutoff_distance)
            mlflow.log_params(weights)
            mlflow.log_metric("clusters",len(set(clusters)))
            mlflow.log_metric("WCSS",wcss)
        return wcss

    study_WCSS = optuna.create_study(direction="minimize",sampler=sampler)
    study_WCSS.optimize(objective_WCSS, n_trials=customer_weight_tuning_trials)

# COMMAND ----------

# DBTITLE 1,Connect to MLflow client to get feature weights 
# Connecting to MLflow client
client = mlflow.MlflowClient()

# Getting the experiment from path specified in config
experiment_id = dict(mlflow.get_experiment_by_name(customer_weight_tuning_experiment))["experiment_id"]

# Filetering TPE optimization runs with required tags and sorting them by WCSS 
runs = [
    run.info.run_id
    for run in client.search_runs(
        experiment_ids=experiment_id,
        filter_string=f"tags.data_version = '{customer_weight_tuning_version}'",
        order_by=["metrics.WCSS"],
    )
]

# Extracting the parameters/ weights that performed the best
best_params = client.get_run(runs[0]).data.params.copy()

# Overwriting the cutoff for clustering mentioned in the runs
del best_params["cutoff_distance"]
cutoff_distance = customer_clustering_cutoff_distance

# Creating the dictionary of weights 
weights = {feature: float(weight) for feature, weight in best_params.items()}
for feature in set(features).difference(set(best_params.keys())):
    weights[feature] = sum([float(weight) for weight in best_params.values()])/len(best_params)

# Restricting to selected features 
weights = {feature: weights[feature] for feature in selected_features}

# COMMAND ----------

# DBTITLE 1,Feature Weights
# Printing custoff distance and feature weights
print(f"Cutoff Distance: {cutoff_distance}")
wt_df = pd.DataFrame([[feature,weight] for feature,weight in weights.items()],columns=["Feature","Weights"])
wt_df.sort_values("Weights",ascending=False).reset_index(drop=True)

# COMMAND ----------

# DBTITLE 1,Compute and save weighted pairwise distances
# Calculating the weighted pairwise distance between customers
wpwd = cluster.weighted_pairwise(weights)

# Delete locally saved arrays
for feature in selected_features:
    dbutils.fs.rm(f"file:/databricks/driver/{feature}")

# Load customer data
customers = utils.read_parquet(f"{customer_data_path}/customer_details.parquet")

# Converting the above condensed matrix into a square dataframe
wpwd_df = pd.DataFrame(squareform(wpwd),index=customers["customer_num"].astype("str"),columns=customers["customer_num"].astype("str"))

# Saving the weighted pairwise distance as a parquet file
utils.save_parquet(wpwd_df,f"{customer_data_path}/wpwd.parquet")

# COMMAND ----------

# DBTITLE 1,Perform clustering
# Computing the linkage matrix based on pairwised distance
Z = linkage(wpwd,"ward")

# Calculating the clusters based on linkage matrix and cutoff distance
clusters = fcluster(Z,t=cutoff_distance,criterion="distance")

# Appending the cluster labels to the customers table
customers["cluster"] = clusters

# Saving the updated customers table
utils.save_parquet(customers,f"{customer_data_path}/customer_details.parquet")

# COMMAND ----------

# DBTITLE 1,Metrics for evaluating clustering
# Computing metric for the performed clustering like wcss and avg silhouette score
ss = cluster.silhouette_scores(wpwd,clusters)
wcss = cluster.WCSS(wpwd,clusters)
silh_score = round(ss.mean(),4)
wcss_score = round(wcss,4)
metrics = []
metrics.append(["Metrics","Scores"])
metrics.append(["Avg. Silhouette Score",silh_score])
metrics.append(["WCSS",wcss_score])

# Plotting silhouette score of each customer considered in the clustering
fig = cluster.silhouette_plot(ss, clusters,weights,metrics)

# COMMAND ----------

# DBTITLE 1,Save the results in the stats.json file
stats = {}

stats["customer clustering weights"] = {"value":weights,"description":"Set of weights for customer clustering"}
stats["customer clustering cutoff distance"] = {"value":cutoff_distance,"description":"Cutoff distance for customer clustering"}
stats["customer clustering n clusters"] = {"value":len(customers["cluster"].unique()),"description":"No. of customer clusters"}
stats["customer clustering silh score"] = {"value":silh_score,"description":"Avg. Silhouette Score for customer clustering"}
stats["customer clustering wcss score"] = {"value":wcss_score,"description":"WCSS for customer clustering"}

utils.save_stats(stats)