from ..utils import utils, predict
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

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

propensity_train_data_sampling = config["propensity_train_data_sampling"]["value"]

propensity_data_path = f"./data/DATA_VERSIONS/{data_version}/PROPENSITY_DATA"
product_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA"
customer_data_path = f"./data/DATA_VERSIONS/{data_version}/CUSTOMER_DATA"

for dirs in [propensity_data_path,product_data_path,customer_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass
    
propensity_data_train = pd.read_parquet(f"{propensity_data_path}/propensity_data_train.parquet")
propensity_data_test = pd.read_parquet(f"{propensity_data_path}/propensity_data_test.parquet")
product_wpwd = pd.read_parquet(f"{product_data_path}/wpwd.parquet")
customer_wpwd = pd.read_parquet(f"{customer_data_path}/wpwd.parquet")

target = "outcome"

propensity_data_train.reset_index(drop=True, inplace=True)
propensity_data_test.reset_index(drop=True, inplace=True)

X = propensity_data_train.drop([target], axis=1) 
y = propensity_data_train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

categorical_values = X_train.select_dtypes(["object", "datetime64[ns]"]).columns
numerical_values = X_train.select_dtypes(["int", "float"]).columns

train_X = X_train.copy()
train_y = y_train.copy()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
train_X[categorical_values] = encoder.fit_transform(train_X[categorical_values])

scaler = StandardScaler()
train_X[numerical_values] = scaler.fit_transform(train_X[numerical_values])

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


RFC = RandomForestClassifier(random_state=0)

RFC.fit(train_X, train_y)

pipeline = Pipeline(
    [
        ("encoder",encoder),
        ("scaler",scaler),
        ("estimator",RFC)
    ]
)

propensity_model = predict.CustomClassifierModel(X_train,pipeline,product_wpwd,customer_wpwd)

train_test_acc = {"train accuracy":propensity_model.evaluate(X_train,y_train),"test accuracy":propensity_model.evaluate(X_test,y_test)}

report = propensity_model.report(X_test,y_test)
confusion = propensity_model.confusion(X_test,y_test).tolist()

result = predict.feature_importance(X_train, y_train,pipeline)
feature_imp = pd.DataFrame({'features':result['features'] , 'score':result['score']})
plt.figure(figsize=(20,10))
feature_imp = feature_imp.sort_values('score')
sns.barplot(y=feature_imp['features'],x=feature_imp['score'], orient='h')
plt.xlabel("Score")
plt.ylabel("Feature Importance")
plt.title('Feature Importance plot')

utils.save_pkl(propensity_model,f"{propensity_data_path}/propensity_model.pkl")

stats = {}

utils.customer_product_stats(stats,X_train,"propensity train")
utils.customer_product_stats(stats,X_test,"propensity test")

stats["propensity train test accuracy"] = {"value":train_test_acc,"description":"Train and Test accuracy of propensity model"}
stats["propensity report"] = {"value":report,"description":"Report of propensity model"}
stats["propensity confusion"] = {"value":confusion,"description":"Confusion matrix of propensity model"}

utils.save_stats(stats)