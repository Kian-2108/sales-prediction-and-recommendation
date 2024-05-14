# Databricks notebook source
# MAGIC %md
# MAGIC ### Build Prediction Model
# MAGIC In this notebook the prediction model is built using the custom dataset prepared in **2C1_Create_Data_For_Prediction**. The custom dataset created is unbalanced, the later section of the notebook covers various upsampling and downsampling techniques to train and test the model.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Load Libraries
#Utils and predict are python scripts imported from the Utility_functions folder
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

import scipy

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import seaborn as sns 
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration settings from the configuration file
config = utils.read_config()
data_version = config["data_version"]["value"]
prediction_train_data_sampling = config["prediction_train_data_sampling"]["value"]

# COMMAND ----------

# DBTITLE 1,Set data paths
#path of prediction folder based on the data version
prediction_data_path = f"DATA_VERSIONS/{data_version}/PREDICTION_DATA"

#path of product cluster folder based on the data version
product_data_path = f"DATA_VERSIONS/{data_version}/PRODUCT_DATA"

#path of customer cluster folder based on the data version
customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"

# COMMAND ----------

# DBTITLE 1,Load Dataset for Prediction model
#load prediction train data
prediction_data_train = utils.read_parquet(f'{prediction_data_path}/prediction_data_train.parquet')

#load prediction test data
prediction_data_test = utils.read_parquet(f'{prediction_data_path}/prediction_data_test.parquet')

#load product pairwise distance data
product_wpwd = utils.read_parquet(f"{product_data_path}/wpwd.parquet")

#load customer pairwise distance data
customer_wpwd = utils.read_parquet(f"{customer_data_path}/wpwd.parquet")

# COMMAND ----------

# DBTITLE 1,Split Data
#set the target feature
target = "iswon"

# Resetting the index of the training and test datasets for prediction data
prediction_data_train.reset_index(drop=True,inplace=True)
prediction_data_test.reset_index(drop=True,inplace=True)

# Separating the features (X) and the target variable (y) for training data
X_train = prediction_data_train.drop([target],axis=1)
y_train = prediction_data_train[target]

# Separating the features (X) and the target variable (y) for test data
X_test = prediction_data_test.drop([target],axis=1)
y_test = prediction_data_test[target] 

# Identifying the columns with categorical and numerical values
categorical_values = X_train.select_dtypes(["object","datetime64[ns]"]).columns
numerical_values = X_train.select_dtypes(["int","float"]).columns

# COMMAND ----------

# DBTITLE 1,Encode and Scale values
#Create a train and test copy
train_X = X_train.copy()
train_y = y_train.copy()

#Encode categorical and datetime values
encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan)
train_X[categorical_values] = encoder.fit_transform(train_X[categorical_values])

#Scale the numerical values
scaler = StandardScaler()
train_X[numerical_values] = scaler.fit_transform(train_X[numerical_values])

# COMMAND ----------

# DBTITLE 1,Optional sampling
if prediction_train_data_sampling == "upsample":
    train_X, train_y = predict.upsample(X_train,y_train,encoder,scaler)
elif prediction_train_data_sampling == "downsample":
    train_X, train_y = predict.downsample(X_train,y_train,encoder,scaler)

# COMMAND ----------

# DBTITLE 1,Train model on Gradient Boosted Decision Tree
GBDT = GradientBoostingClassifier(random_state=0)
GBDT.fit(train_X, train_y)

# COMMAND ----------

# DBTITLE 1,Create Prediction Pipeline
#This pipeline is built using sklearn pipeline. This pipeline stores the encoder, scaler and the trained model

pipeline = Pipeline(
    [
        ("encoder",encoder),
        ("scaler",scaler),
        ("estimator",GBDT)
    ]
)

# COMMAND ----------

# DBTITLE 1,Create a prediction object
prediction_model = predict.CustomClassifierModel(X_train,pipeline,product_wpwd,customer_wpwd)

# COMMAND ----------

# DBTITLE 1,Save the prediction object as a pickle file
utils.save_pkl(prediction_model,f"{prediction_data_path}/prediction_model.pkl")

# COMMAND ----------

train_test_acc = {"train accuracy":prediction_model.evaluate(X_train,y_train),"test accuracy":prediction_model.evaluate(X_test,y_test)}
# cross_val_acc = cross_val_score(GradientBoostingClassifier(random_state=0),train_X,train_y,cv=3)

# COMMAND ----------

report = prediction_model.report(X_test,y_test)
confusion = prediction_model.confusion(X_test,y_test).tolist()

# COMMAND ----------

# DBTITLE 1,Feature Imp Plot
result = predict.feature_importance(X_train, y_train,pipeline)
result = result[result['score']!=0]
feature_imp = pd.DataFrame({'features':result['features'] , 'score':result['score']}) 
feature_imp = feature_imp[feature_imp["score"]!=0]
plt.figure(figsize=(20,10))
feature_imp = feature_imp.sort_values('score')
sns.barplot(y=feature_imp['features'],x=feature_imp['score'], orient='h')
plt.rcParams.update({'font.size': 15})
plt.xlabel("Score")
plt.ylabel("Feature Importance",fontsize=20)
plt.title('Feature Importance plot',fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

# COMMAND ----------

stats = {}

# stats["prediciton cross val accuracy"] = {"value":list(cross_val_acc),"description":"Cross val accuracy of prediction model"}
stats["prediciton train test accuracy"] = {"value":train_test_acc,"description":"Train and Test accuracy of prediction model"}
stats["prediction report"] = {"value":report,"description":"Report of prediction model"}
stats["prediction confusion"] = {"value":confusion,"description":"Confusion matrix of prediction model"}

utils.save_stats(stats)