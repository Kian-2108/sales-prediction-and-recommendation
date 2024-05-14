# Databricks notebook source
# MAGIC %md
# MAGIC ### Dataset Creation for Recommendation Model
# MAGIC This notebook involves using the prediction model to partially fill the collaborative matrices. In the later section of the notebook, the partially filled collaborative matirces are filled using different matrix factorization algorithms.

# COMMAND ----------

# DBTITLE 1,Import Libraries
# utils and predict are python scripts imported from the Utility_functions folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import predict
import mlflow
import numpy as np
import pandas as pd

from implicit.als import AlternatingLeastSquares as ALS
from implicit.cpu.lmf import LogisticMatrixFactorization as LMF
from implicit.cpu.bpr import BayesianPersonalizedRanking as BPR
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

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration settings from a configuration file
config = utils.read_config()
data_version = config["data_version"]["value"]
matrix_model = config["matrix_model"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file paths
#File path where data related to prediction model are stored
prediction_data_path = f"DATA_VERSIONS/{data_version}/PREDICTION_DATA"
#path for customer data based on the data version
customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"
#path for product data based on the data version
product_data_path = f"DATA_VERSIONS/{data_version}/PRODUCT_DATA"
#path for recommendation data based on the data version
recommendation_data_path = f"DATA_VERSIONS/{data_version}/RECOMMENDATION_DATA"
#path for after cutoff raw data based on the data version
after_cutoff_path = f"DATA_VERSIONS/{data_version}/AFTER_CUTOFF_RAW"

# COMMAND ----------

# DBTITLE 1,Load Data
#Reads the training data for prediction from a Parquet file
prediction_data_train = utils.read_parquet(f'{prediction_data_path}/prediction_data_train.parquet')

#Reads the test data for prediction from a Parquet file
prediction_data_test = utils.read_parquet(f'{prediction_data_path}/prediction_data_test.parquet')

#Reads the curated products data after cutoff from a Parquet file
CPL = utils.read_parquet(f"{after_cutoff_path}/curated_products_data.parquet")

#Reads the customer cluster information from a Parquet file
customers = utils.read_parquet(f"{customer_data_path}/customer_details.parquet")

#Reads the product cluster information from a Parquet file
products = utils.read_parquet(f"{product_data_path}/product_details.parquet")

#Reads the customer cluster pairwise distance
prod_wpwd = utils.read_parquet(f"{product_data_path}/wpwd.parquet")

#Reads the product cluster pairwise distance
cust_wpwd = utils.read_parquet(f"{customer_data_path}/wpwd.parquet")

#Loads the prediction pickle file
prediction_model = utils.read_pkl(f"{prediction_data_path}/prediction_model.pkl")

# COMMAND ----------

# DBTITLE 1,Split the data into feature matrix (X) and target (y)
#The data is split into a feature matrix (X) and a target variable (y) to facilitate further analysis and modeling
target = "iswon"

prediction_data_train.reset_index(drop=True,inplace=True)
prediction_data_test.reset_index(drop=True,inplace=True)

X_train = prediction_data_train.drop([target],axis=1)
y_train = prediction_data_train[target]

X_test = prediction_data_test.drop([target],axis=1)
y_test = prediction_data_test[target] 

categorical_values = X_train.select_dtypes(["object","datetime64[ns]"]).columns
numerical_values = X_train.select_dtypes(["int","float"]).columns

# COMMAND ----------

# DBTITLE 1,Get the output probability for train
#Generates predicted outcome probabilities using the trained prediction model. This probability score will be used later for filling the ratings in the collaborative matrix
outcome_prob = prediction_model.predict(X_train,probability=True)

#Generates predicted outcome classes using the trained prediction model
outcome_class = prediction_model.predict(X_train)

# COMMAND ----------

# DBTITLE 1,Get Ratings for train data
#A DataFrame is created to store the predicted outcome probabilities
output_probabilities = pd.DataFrame(outcome_prob.values,columns=["win_prob"],index=outcome_prob.index)
output_probabilities['predicted'] = outcome_class
output_probabilities['ground_truth'] = y_train.copy()

#If the ground truth and predicted outcomes do not match, the rating is calculated as 1 minus the predicted win probability, otherwise, the rating is equal to the predicted win probability
output_probabilities['rating'] = output_probabilities.apply(lambda row: 1-row['win_prob'] if row['ground_truth'] != row['predicted'] else row['win_prob'], axis=1)

#Transform the ratings to the range [-1, 1] by scaling and shifting the values
output_probabilities['rating'] = 2*output_probabilities['rating']-1
X_train['ratings'] = output_probabilities['rating']

# COMMAND ----------

# DBTITLE 1,Merge Customer cluster with the ratings data
# X_subset consists of only those features that will be used in the collaborative filter.
X_subset = X_train[['customer_num','srp_2_desc','ratings']].dropna()

#For each customer-product-rating triplet append the customer cluster info
X_subset = X_subset.merge(customers[["customer_num","cluster"]],on="customer_num",how="left")

# COMMAND ----------

# DBTITLE 1,Create Collaborative matrices for different customer clusters
#initialize an empty dictionary called ratings, which will be used to store the collaborative matrix for each cluster.
X_subset.dropna(inplace=True)
ratings = dict()
for cluster in X_subset["cluster"].unique():
    ratings[cluster] = X_subset[X_subset["cluster"]==cluster].drop("cluster",axis=1)

# COMMAND ----------

# DBTITLE 1,Print Unique Customers with ratings and purchase records
unique_customer = len(X_subset["customer_num"].unique())
number_of_purchase_rec = len(X_subset)
print(f"Number of Unique Customers with ratings: {unique_customer} \nNumber of purchase records: {number_of_purchase_rec}")

# COMMAND ----------

# DBTITLE 1,Load Product Cluster 
for wpwd in [cust_wpwd,prod_wpwd]:
    wpwd.replace(0,0.000001,inplace=True)
    wpwd = 1/wpwd

# COMMAND ----------

# DBTITLE 1,Create and Fill Collaborative Matrices
#Upto now the Collaborative Matrices are only partially filled. This cell uses different matrix factorization algorithm like ALS (Alternating Least Squares), BPR (Bayesian personalized ranking) and LMF (Linked matrix factorization) to fill the collaborative matrices

# Create a dictionary to store collaborative filtering matrices for each customer cluster
cf_matrix = dict()

# Iterate over each customer cluster and its corresponding partial matrix
for customers_cluster,partial_matrix in ratings.items():
    # Filter customers belonging to the current cluster
    clustered = customers[customers["cluster"]==customers_cluster]
    
    df1 = partial_matrix.groupby(["customer_num","srp_2_desc"])[["ratings"]].mean().unstack()
    df1.columns = [col[-1] for col in df1.columns]

    # Store the results in the cf_matrix dictionary
    cf_matrix[customers_cluster] = dict()

    # Define a list of collaborative filtering models
    MODELS = [ALS,BPR,LMF]

    for MODEL in MODELS:

        # Get the optimal the number of latent factors for the current model
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

        # Instantiate the collaborative filtering model with the specified parameters            
        model = MODEL(factors=best_factors,iterations=30,random_state=0,num_threads=1)

        # Fit the model using the data in sparse matrix format
        model.fit(scipy.sparse.csr_matrix(dft.values),show_progress=False)

        # Calculate the predicted ratings matrix by multiplying user and item factors
        df2 = pd.DataFrame((model.user_factors)@(model.item_factors.transpose()))
        df2.index = df1.index
        df2.columns = df1.columns

        # Calculate RMSE (root mean squared error) between the predicted and actual ratings
        rmse = np.sqrt(np.square((df2-df1)).mean().mean())

        # Merge the predicted ratings matrix with the clustered customer data
        df3 = df2.merge(clustered[["customer_num","cluster"]],left_index=True,right_on="customer_num",how="outer").set_index("customer_num")
        df3.drop("cluster",axis=1,inplace=True)

        # Perform additional calculations for rows with missing values in df3
        temp_wpwd_cols = df2.index
        temp_wpwd_rows = df3.index.difference(temp_wpwd_cols)
        temp_L = cust_wpwd.loc[temp_wpwd_rows,temp_wpwd_cols].copy()
        temp_R = df2.loc[temp_wpwd_cols,:].copy()

        # Normalize the rows of temp_L
        for row in temp_L.index:
            temp_L.loc[row,:] = temp_L.loc[row,:]/(temp_L.loc[row,:].sum())
        
        # Fill missing values in df3 using matrix multiplication
        temp_fill = temp_L@temp_R
        df3.loc[temp_fill.index,temp_fill.columns] = temp_fill

        # Merge the result with product clusters
        df4 = df3.transpose().merge(products[["srp_2_desc","cluster"]],left_index=True,right_on="srp_2_desc",how="left").set_index("srp_2_desc")
        df4_clusters = df4["cluster"].unique()

        # Merge the result with product clusters while keeping only the common clusters
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

        # Store the final matrices and RMSE values in the cf_matrix dictionary
        cf_matrix[customers_cluster][f"{MODEL.__name__}"] = {"matrix":df5,"rmse":rmse,"matrix_1":df1,"matrix_2":df2,"matrix_3":df3,"matrix_4":df4}

# COMMAND ----------

# DBTITLE 1,Data Checkpoint (Save matrix)
#Save the collaborative matrices
def save_matrix(matrix,model):
    for cluster,models in matrix.items():
        matrix = models[model]["matrix"]
        cluster = int(cluster)
        utils.save_parquet(matrix, f"{recommendation_data_path}/CF_MATRIX/{cluster}.parquet")

save_matrix(cf_matrix,matrix_model)

# COMMAND ----------

# DBTITLE 1,Get Results for different Matrix Factorization algorithms
result = pd.DataFrame(index=cf_matrix.keys(),columns=["Step 1 Shape","Step 2 Shape","Final Matrix Shape",ALS.__name__,BPR.__name__,LMF.__name__])

for cluster,models in cf_matrix.items():
    for model,info in models.items():
        result.loc[cluster,model] = info["rmse"]
        result.loc[cluster,"Step 1 Shape"] = info["matrix_2"].shape
        result.loc[cluster,"Step 2 Shape"] = info["matrix_3"].shape
        result.loc[cluster,"Final Matrix Shape"] = info["matrix"].shape

# COMMAND ----------

# DBTITLE 1,Results for each cluster for different Matrix Factorization Algorithm
result

# COMMAND ----------

# DBTITLE 1,Average Result for different Matrix Factorization Algorithms (RMSE)
pd.DataFrame(result.mean(),columns=["Avg. rmse"])

# COMMAND ----------

recommendable_customers = sum(len(cf_matrix[key][ALS.__name__]["matrix"].index) for key in cf_matrix.keys())
recommendable_clusters = len(cf_matrix.keys())

# COMMAND ----------

stats = {}

stats["recommendable customers"] = {"value":recommendable_customers,"description":"No. customers known to the recommendation model"}
stats["recommendable clusters"] = {"value":recommendable_clusters,"description":"No. clusters known to the recommendation model"}

utils.save_stats(stats)