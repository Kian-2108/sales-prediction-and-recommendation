# Databricks notebook source
# MAGIC %md
# MAGIC ### Build Propensity Model
# MAGIC This model uses the custom dataset from **3B1_Create_Data_For_Propensity** to train the propensity model. This notebook also include running a k-fold cross validation over different Classifier models to choose the best model for propensity.

# COMMAND ----------

# DBTITLE 1,Load Libraries
#utility and predict are python scripts imported from Utility_fuction folder
import sys
sys.path.append("/Workspace/Users/davide@baxter.com/Solution/0A_Utility_Functions")
import utils
import predict
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
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

# COMMAND ----------

# DBTITLE 1,Read Config file
# Read the configuration settings from a configuration file
config = utils.read_config()
data_version = config["data_version"]["value"]
propensity_train_data_sampling = config["propensity_train_data_sampling"]["value"]

# COMMAND ----------

# DBTITLE 1,Set file paths
#path for propensity model based on the data version
propensity_data_path = f"DATA_VERSIONS/{data_version}/PROPENSITY_DATA"
#path for product cluster info based on the data version
product_data_path = f"DATA_VERSIONS/{data_version}/PRODUCT_DATA"
#path for customer cluster info based on the data version
customer_data_path = f"DATA_VERSIONS/{data_version}/CUSTOMER_DATA"

# COMMAND ----------

# DBTITLE 1,Load files
#Reads the training data for propensity from a Parquet file
propensity_data_train = utils.read_parquet(f"{propensity_data_path}/propensity_data_train.parquet")
#Reads the testing data for prediction from a Parquet file
propensity_data_test = utils.read_parquet(f"{propensity_data_path}/propensity_data_test.parquet")
#Reads the product pairwise data for prediction from a Parquet file
product_wpwd = utils.read_parquet(f"{product_data_path}/wpwd.parquet")
#Reads the customer pairwise data for prediction from a Parquet file
customer_wpwd = utils.read_parquet(f"{customer_data_path}/wpwd.parquet")

# COMMAND ----------

# DBTITLE 1,Split Data
#set the target feature
target = "outcome"


# Resetting the index of the training and test datasets for propensity data
propensity_data_train.reset_index(drop=True, inplace=True)
propensity_data_test.reset_index(drop=True, inplace=True)

# Separating the features (X) and the target variable (y) for training data
X = propensity_data_train.drop([target], axis=1) 
y = propensity_data_train[target]

# Splitting the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Identifying the columns with categorical and numerical values
categorical_values = X_train.select_dtypes(["object", "datetime64[ns]"]).columns
numerical_values = X_train.select_dtypes(["int", "float"]).columns


# COMMAND ----------

# DBTITLE 1,Encode and Scale values
# Creating copies of the training features and target variables
train_X = X_train.copy()
train_y = y_train.copy()

# Initializing an ordinal encoder to handle unknown values
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
# Encoding the categorical columns in the training features using the ordinal encoder
train_X[categorical_values] = encoder.fit_transform(train_X[categorical_values])

# Initializing a standard scaler to scale the numerical columns
scaler = StandardScaler()
# Scaling the numerical columns in the training features using the standard scaler
train_X[numerical_values] = scaler.fit_transform(train_X[numerical_values])

# COMMAND ----------

if propensity_train_data_sampling == "upsample":
    train_X, train_y = predict.upsample(X_train,y_train,encoder,scaler)
elif propensity_train_data_sampling == "downsample":
    train_X, train_y = predict.downsample(X_train,y_train,encoder,scaler)

# COMMAND ----------

# DBTITLE 1,Test the propensity model on different classifiers
# # Create an array of models
# models = []
# models.append(("LR",LogisticRegression()))
# models.append(("NB",GaussianNB()))
# models.append(("RF",RandomForestClassifier()))
# models.append(("SVC",SVC()))
# models.append(("DT",DecisionTreeClassifier()))
# models.append(("XGB",xgb.XGBClassifier()))
# models.append(("KNN",KNeighborsClassifier()))

# # Measure the accuracy using k fold cross validation
# Scores_list = []
# k=5
# for name,model in models:
#     kfold = KFold(n_splits=k)
#     cv_result = cross_val_score(model,train_X,train_y, cv = kfold,scoring = "accuracy")
#     Scores_list.append([name]+list(cv_result))

# COMMAND ----------

# DBTITLE 1,Print results
# # Results of k fold cross validation
# Scores = pd.DataFrame(Scores_list,columns=["Model"]+[f"Accuracy_#{i}" for i in range(1,k+1)])
# Scores[f"Avg. Accuracy"] = Scores[Scores.drop("Model",axis=1).columns].mean(axis=1)
# Scores = Scores.sort_values(f"Avg. Accuracy",ascending=False).set_index("Model")
# Scores = (Scores*100).round(2)
# Scores

# COMMAND ----------

# DBTITLE 1,Train Model 
# Initializing a random forest classifier with a random state of 0
RFC = RandomForestClassifier(random_state=0)

# Fitting the random forest classifier on the scaled training features and target variables
RFC.fit(train_X, train_y)

# COMMAND ----------

# DBTITLE 1,Create a Pipeline
#This pipeline is built using sklearn pipeline. This pipeline stores the encoder, scaler and the trained model

pipeline = Pipeline(
    [
        ("encoder",encoder),
        ("scaler",scaler),
        ("estimator",RFC)
    ]
)

# COMMAND ----------

# DBTITLE 1,Create a Propensity model object
propensity_model = predict.CustomClassifierModel(X_train,pipeline,product_wpwd,customer_wpwd)

# COMMAND ----------

# DBTITLE 1,Test Model
train_test_acc = {"train accuracy":propensity_model.evaluate(X_train,y_train),"test accuracy":propensity_model.evaluate(X_test,y_test)}
# cross_val_acc = cross_val_score(RandomForestClassifier(random_state=0),train_X,train_y,cv=3)

# COMMAND ----------

# DBTITLE 1,Create a classification report and the confusion score
report = propensity_model.report(X_test,y_test)
confusion = propensity_model.confusion(X_test,y_test).tolist()

# COMMAND ----------

# DBTITLE 1,Feature Imp Plot
result = predict.feature_importance(X_train, y_train,pipeline)
feature_imp = pd.DataFrame({'features':result['features'] , 'score':result['score']})
plt.figure(figsize=(20,10))
feature_imp = feature_imp.sort_values('score')
sns.barplot(y=feature_imp['features'],x=feature_imp['score'], orient='h')
plt.xlabel("Score")
plt.ylabel("Feature Importance")
plt.title('Feature Importance plot')

# COMMAND ----------

# DBTITLE 1,Save the object as a pickle file
utils.save_pkl(propensity_model,f"{propensity_data_path}/propensity_model.pkl")

# COMMAND ----------

# DBTITLE 1,Save the results in the stats.json file
stats = {}

utils.customer_product_stats(stats,X_train,"propensity train")
utils.customer_product_stats(stats,X_test,"propensity test")

# stats["propensity cross val accuracy"] = {"value":list(cross_val_acc),"description":"Train and Test cross val accuracy of propensity model"}
stats["propensity train test accuracy"] = {"value":train_test_acc,"description":"Train and Test accuracy of propensity model"}
stats["propensity report"] = {"value":report,"description":"Report of propensity model"}
stats["propensity confusion"] = {"value":confusion,"description":"Confusion matrix of propensity model"}

utils.save_stats(stats)