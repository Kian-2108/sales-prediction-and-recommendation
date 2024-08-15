from ..utils import utils, cluster
import os
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
import optuna

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cut_tree

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

product_clustering_cutoff_distance = config["product_clustering_cutoff_distance"]["value"]
run_product_feature_wrapping = config["run_product_feature_wrapping"]["value"]
product_top_feature = config["product_top_feature"]["value"]
run_product_weight_tuning = config["run_product_weight_tuning"]["value"]
product_weight_tuning_trials = config["product_weight_tuning_trials"]["value"]

if run_product_weight_tuning:
    product_weight_tuning_version = data_version
else:
    product_weight_tuning_version = config["product_weight_tuning_version"]["value"]

if run_product_feature_wrapping:
    product_feature_wrapping_version = data_version
else:
    product_feature_wrapping_version = config["product_feature_wrapping_version"]["value"]

product_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA"
temp_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_TEMP_DATA"

for dirs in [product_data_path,temp_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass
    
features = utils.read_filenames(f"{product_data_path}/PWD")

for feature in features:
    with open(f"{temp_data_path}/{feature}","wb") as npy:
        np.save(npy,pd.read_parquet(f"{product_data_path}/PWD/{feature}.parquet")[feature].to_numpy())

product_feature_wrapping_experiment = "/Users/davide@baxter.com/Solution/2B_Product_Clustering/Feature-Wrapping"

features = utils.read_filenames(f"{product_data_path}/PWD")

if run_product_feature_wrapping:

    mlflow.set_experiment(product_feature_wrapping_experiment)

    selected_features = []
    while set(selected_features) != set(features):
        next_features = set(features.copy()) - set(selected_features.copy())
        wcss_dict = dict()
        cutoff_distance = product_clustering_cutoff_distance
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

    client = mlflow.MlflowClient()

    experiment_id = dict(mlflow.get_experiment_by_name(product_feature_wrapping_experiment))["experiment_id"]

    runs = [
        run.info.run_id
        for run in client.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.data_version = '{product_feature_wrapping_version}'",
            order_by=[f"params.feature_len DESC"]
        )
    ]

    best_params = client.get_run(runs[0]).data.params.copy()["feature_set"]

    selected_features = best_params.replace("[","").replace("]","").replace("'","").split(", ")
    selected_features = selected_features[:product_top_feature]

else: 
    selected_features = features
    
product_weight_tuning_experiment = "/Users/davide@baxter.com/Solution/2B_Product_Clustering/Feature-Weights"

if run_product_weight_tuning:
    
    mlflow.set_experiment(product_weight_tuning_experiment)

    sampler = optuna.samplers.TPESampler()

    def objective_WCSS(trial):
        with mlflow.start_run(run_name=str(datetime.now())):
            weights = {feature:trial.suggest_int(feature,1,10) for feature in selected_features}
            wpwd = cluster.weighted_pairwise(weights)
            cutoff_distance = product_clustering_cutoff_distance
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
    study_WCSS.optimize(objective_WCSS, n_trials=product_weight_tuning_trials)

    client = mlflow.MlflowClient()

    experiment_id = dict(mlflow.get_experiment_by_name(product_weight_tuning_experiment))["experiment_id"]

    runs = [
        run.info.run_id
        for run in client.search_runs(
            experiment_ids=experiment_id,
            filter_string=f"tags.data_version = '{product_weight_tuning_version}'",
            order_by=["metrics.WCSS"],
        )
    ]

    best_params = client.get_run(runs[0]).data.params.copy()

    del best_params["cutoff_distance"]
 
    weights = {feature: float(weight) for feature, weight in best_params.items()}
    for feature in set(features).difference(set(best_params.keys())):
        weights[feature] = sum([float(weight) for weight in best_params.values()])/len(best_params)

    weights = {feature: weights[feature] for feature in selected_features}

else:
    weights = {feature: 1/len(feature) for feature in selected_features}
    
cutoff_distance = product_clustering_cutoff_distance

# print(f"Cutoff Distance: {cutoff_distance}")
wt_df = pd.DataFrame([[feature,weight] for feature,weight in weights.items()],columns=["Feature","Weights"])
wt_df.sort_values("Weights",ascending=False).reset_index(drop=True)

wpwd = cluster.weighted_pairwise(weights,temp_data_path)

# for feature in selected_features:
#     dbutils.fs.rm(f"file:/databricks/driver/{feature}")

products = pd.read_parquet(f"{product_data_path}/product_details.parquet")

wpwd_df = pd.DataFrame(squareform(wpwd),index=products["srp_2_desc"].astype("str"),columns=products["srp_2_desc"].astype("str"))

wpwd_df.to_parquet(f"{product_data_path}/wpwd.parquet")

Z = linkage(wpwd,"ward")

clusters = fcluster(Z,t=cutoff_distance,criterion="distance")

products["cluster"] = clusters

products.to_parquet(f"{product_data_path}/product_details.parquet")

ss = cluster.silhouette_scores(wpwd,clusters)
wcss = cluster.WCSS(wpwd,clusters)
silh_score = round(ss.mean(),4)
wcss_score = round(wcss,4)
metrics = []
metrics.append(["Metrics","Scores"])
metrics.append(["Avg. Silhouette Score",silh_score])
metrics.append(["WCSS",wcss_score])

fig = cluster.silhouette_plot(ss, clusters,weights,metrics)

stats = {}

stats["product clustering weights"] = {"value":weights,"description":"Set of weights for product clustering"}
stats["product clustering cutoff distance"] = {"value":cutoff_distance,"description":"Cutoff distance for product clustering"}
stats["product clustering n clusters"] = {"value":len(products["cluster"]),"description":"No. of product clusters"}
stats["product clustering silh score"] = {"value":silh_score,"description":"Avg. Silhouette Score for product clustering"}
stats["product clustering wcss score"] = {"value":wcss_score,"description":"WCSS for product clustering"}

utils.save_stats(stats)