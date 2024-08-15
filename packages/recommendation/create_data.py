from ..utils import utils, predict
import mlflow
import numpy as np
import pandas as pd

# from implicit.als import AlternatingLeastSquares as ALS
# from implicit.cpu.lmf import LogisticMatrixFactorization as LMF
# from implicit.cpu.bpr import BayesianPersonalizedRanking as BPR
from sklearn.decomposition import NMF

import scipy

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import seaborn as sns 
import matplotlib.pyplot as plt

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]
matrix_model = config["matrix_model"]["value"]

prediction_data_path = f"./data/DATA_VERSIONS/{data_version}/PREDICTION_DATA"
customer_data_path = f"./data/DATA_VERSIONS/{data_version}/CUSTOMER_DATA"
product_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA"
recommendation_data_path = f"./data/DATA_VERSIONS/{data_version}/RECOMMENDATION_DATA"
after_cutoff_path = f"./data/DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

for dirs in [prediction_data_path,customer_data_path,product_data_path,recommendation_data_path,after_cutoff_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

prediction_data_train = pd.read_parquet(f'{prediction_data_path}/prediction_data_train.parquet')
prediction_data_test = pd.read_parquet(f'{prediction_data_path}/prediction_data_test.parquet')
CPL = pd.read_parquet(f"{after_cutoff_path}/curated_products_data.parquet")
customers = pd.read_parquet(f"{customer_data_path}/customer_details.parquet")
products = pd.read_parquet(f"{product_data_path}/product_details.parquet")
prod_wpwd = pd.read_parquet(f"{product_data_path}/wpwd.parquet")
cust_wpwd = pd.read_parquet(f"{customer_data_path}/wpwd.parquet")

prediction_model = utils.read_pkl(f"{prediction_data_path}/prediction_model.pkl")

target = "iswon"

prediction_data_train.reset_index(drop=True,inplace=True)
prediction_data_test.reset_index(drop=True,inplace=True)

X_train = prediction_data_train.drop([target],axis=1)
y_train = prediction_data_train[target]

X_test = prediction_data_test.drop([target],axis=1)
y_test = prediction_data_test[target] 

categorical_values = X_train.select_dtypes(["object","datetime64[ns]"]).columns
numerical_values = X_train.select_dtypes(["int","float"]).columns

outcome_prob = prediction_model.predict(X_train,probability=True)
outcome_class = prediction_model.predict(X_train)

output_probabilities = pd.DataFrame(outcome_prob.values,columns=["win_prob"],index=outcome_prob.index)
output_probabilities['predicted'] = outcome_class
output_probabilities['ground_truth'] = y_train.copy()

output_probabilities['rating'] = output_probabilities.apply(lambda row: 1-row['win_prob'] if row['ground_truth'] != row['predicted'] else row['win_prob'], axis=1)

output_probabilities['rating'] = 2*output_probabilities['rating']-1
X_train['ratings'] = output_probabilities['rating']

X_subset = X_train[['customer_num','srp_2_desc','ratings']].dropna()
X_subset = X_subset.merge(customers[["customer_num","cluster"]],on="customer_num",how="left")

X_subset.dropna(inplace=True)
ratings = dict()
for cluster in X_subset["cluster"].unique():
    ratings[cluster] = X_subset[X_subset["cluster"]==cluster].drop("cluster",axis=1)

unique_customer = len(X_subset["customer_num"].unique())
number_of_purchase_rec = len(X_subset)
print(f"Number of Unique Customers with ratings: {unique_customer} \nNumber of purchase records: {number_of_purchase_rec}")

for wpwd in [cust_wpwd,prod_wpwd]:
    wpwd.replace(0,0.000001,inplace=True)
    wpwd = 1/wpwd

cf_matrix = dict()

for customers_cluster,partial_matrix in ratings.items():
    clustered = customers[customers["cluster"]==customers_cluster]
    
    df1 = partial_matrix.groupby(["customer_num","srp_2_desc"])[["ratings"]].mean().unstack()
    df1.columns = [col[-1] for col in df1.columns]

    cf_matrix[customers_cluster] = dict()

    MODELS = [ALS,BPR,LMF]

    for MODEL in MODELS:

        dft = df1.copy().fillna(0)
        # errors = []
        # for i in range(1,101):
        #     model = MODEL(factors=i,random_state=0)
        #     model.fit(scipy.sparse.csr_matrix(dft.values),show_progress=False)
        #     df2 = pd.DataFrame(np.matmul(model.user_factors,model.item_factors.transpose()))
        #     df2.index = df1.index
        #     df2.columns = df1.columns
        #     errors.append([i,np.sqrt(np.square((df2-df1)).mean().mean())])
        # temp = pd.DataFrame(errors,columns=["factors","rmse"])
        # temp["delta"] = temp["rmse"].shift(1)-temp["rmse"]
        # best_factors = temp[temp["delta"]==temp["delta"].max()]["factors"].values[0]
        best_factors = 50

        model = MODEL(factors=best_factors,iterations=30,random_state=0,num_threads=1)

        model.fit(scipy.sparse.csr_matrix(dft.values),show_progress=False)

        df2 = pd.DataFrame((model.user_factors)@(model.item_factors.transpose()))
        df2.index = df1.index
        df2.columns = df1.columns

        rmse = np.sqrt(np.square((df2-df1)).mean().mean())

        df3 = df2.merge(clustered[["customer_num","cluster"]],left_index=True,right_on="customer_num",how="outer").set_index("customer_num")
        df3.drop("cluster",axis=1,inplace=True)

        temp_wpwd_cols = df2.index
        temp_wpwd_rows = df3.index.difference(temp_wpwd_cols)
        temp_L = cust_wpwd.loc[temp_wpwd_rows,temp_wpwd_cols].copy()
        temp_R = df2.loc[temp_wpwd_cols,:].copy()

        for row in temp_L.index:
            temp_L.loc[row,:] = temp_L.loc[row,:]/(temp_L.loc[row,:].sum())
        
        temp_fill = temp_L@temp_R
        df3.loc[temp_fill.index,temp_fill.columns] = temp_fill

        df4 = df3.transpose().merge(products[["srp_2_desc","cluster"]],left_index=True,right_on="srp_2_desc",how="left").set_index("srp_2_desc")
        df4_clusters = df4["cluster"].unique()

        df5 = df3.transpose().merge(products[["srp_2_desc","cluster"]],left_index=True,right_on="srp_2_desc",how="outer").set_index("srp_2_desc")
        df5 = df5[df5["cluster"].isin(df4_clusters)]

        for cluster in df5["cluster"].unique():
            if cluster != np.nan:
                temp_wpwd_rows = df5[df5["cluster"]==cluster].dropna().index
                temp_wpwd_cols = df5[df5["cluster"]==cluster].index.difference(temp_wpwd_rows)
                temp_L = df5.loc[temp_wpwd_rows,:].copy().drop("cluster",axis=1).transpose()
                temp_R = prod_wpwd.loc[temp_wpwd_rows,temp_wpwd_cols].copy()
                for col in temp_R.columns:
                    temp_R[col] = temp_R[col]/(temp_R[col].sum())
                temp_fill = (temp_L@temp_R).transpose()
                temp_fill["cluster"] = [cluster for _ in range(len(temp_fill))]
                df5.loc[temp_fill.index,temp_fill.columns] = temp_fill
        df5 = df5.drop("cluster",axis=1).transpose()

        cf_matrix[customers_cluster][f"{MODEL.__name__}"] = {"matrix":df5,"rmse":rmse,"matrix_1":df1,"matrix_2":df2,"matrix_3":df3,"matrix_4":df4}

def save_matrix(matrix,model):
    for cluster,models in matrix.items():
        matrix = models[model]["matrix"]
        cluster = int(cluster)
        utils.save_parquet(matrix, f"{recommendation_data_path}/CF_MATRIX/{cluster}.parquet")

save_matrix(cf_matrix,matrix_model)

result = pd.DataFrame(index=cf_matrix.keys(),columns=["Step 1 Shape","Step 2 Shape","Final Matrix Shape",ALS.__name__,BPR.__name__,LMF.__name__])

for cluster,models in cf_matrix.items():
    for model,info in models.items():
        result.loc[cluster,model] = info["rmse"]
        result.loc[cluster,"Step 1 Shape"] = info["matrix_2"].shape
        result.loc[cluster,"Step 2 Shape"] = info["matrix_3"].shape
        result.loc[cluster,"Final Matrix Shape"] = info["matrix"].shape

result

pd.DataFrame(result.mean(),columns=["Avg. rmse"])

recommendable_customers = sum(len(cf_matrix[key][ALS.__name__]["matrix"].index) for key in cf_matrix.keys())
recommendable_clusters = len(cf_matrix.keys())

stats = {}

stats["recommendable customers"] = {"value":recommendable_customers,"description":"No. customers known to the recommendation model"}
stats["recommendable clusters"] = {"value":recommendable_clusters,"description":"No. clusters known to the recommendation model"}

utils.save_stats(stats)