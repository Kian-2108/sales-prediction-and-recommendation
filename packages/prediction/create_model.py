from ..utils import utils
from ..utils import predict
import os
import mlflow
import numpy as np
import pandas as pd

# from implicit.als import AlternatingLeastSquares as ALS
# from implicit.cpu.lmf import LogisticMatrixFactorization as LMF
# from implicit.cpu.bpr import BayesianPersonalizedRanking as BPR

# import scipy

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import seaborn as sns 
import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report

config = utils.read_json("./config.json")
data_version = config["data_version"]["value"]

prediction_train_data_sampling = config["prediction_train_data_sampling"]["value"]

prediction_data_path = f"./data/DATA_VERSIONS/{data_version}/PREDICTION_DATA"
product_data_path = f"./data/DATA_VERSIONS/{data_version}/PRODUCT_DATA"
customer_data_path = f"./data/DATA_VERSIONS/{data_version}/CUSTOMER_DATA"

for dirs in [prediction_data_path,product_data_path,customer_data_path]:
    try:
        os.makedirs(dirs)
    except:
        pass

prediction_data_train = pd.read_parquet(f'{prediction_data_path}/prediction_data_train.parquet')
prediction_data_test = pd.read_parquet(f'{prediction_data_path}/prediction_data_test.parquet')

product_wpwd = pd.read_parquet(f"{product_data_path}/wpwd.parquet")
customer_wpwd = pd.read_parquet(f"{customer_data_path}/wpwd.parquet")

target = "iswon"

prediction_data_train.reset_index(drop=True,inplace=True)
prediction_data_test.reset_index(drop=True,inplace=True)

X_train = prediction_data_train.drop([target],axis=1)
y_train = prediction_data_train[target]

X_test = prediction_data_test.drop([target],axis=1)
y_test = prediction_data_test[target] 

categorical_values = X_train.select_dtypes(["object","datetime64[ns]"]).columns
numerical_values = X_train.select_dtypes(["int","float"]).columns

train_X = X_train.copy()
train_y = y_train.copy()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan)
train_X[categorical_values] = encoder.fit_transform(train_X[categorical_values])

scaler = StandardScaler()
train_X[numerical_values] = scaler.fit_transform(train_X[numerical_values])

if prediction_train_data_sampling == "upsample":
    train_X, train_y = predict.upsample(X_train,y_train,encoder,scaler)
elif prediction_train_data_sampling == "downsample":
    train_X, train_y = predict.downsample(X_train,y_train,encoder,scaler)

GBDT = GradientBoostingClassifier(random_state=0)
GBDT.fit(train_X, train_y)

pipeline = Pipeline(
    [
        ("encoder",encoder),
        ("scaler",scaler),
        ("estimator",GBDT)
    ]
)

prediction_model = predict.CustomClassifierModel(X_train,pipeline,product_wpwd,customer_wpwd)

utils.save_pkl(prediction_model,f"{prediction_data_path}/prediction_model.pkl")

train_test_acc = {"train accuracy":prediction_model.evaluate(X_train,y_train),"test accuracy":prediction_model.evaluate(X_test,y_test)}

report = prediction_model.report(X_test,y_test)
confusion = prediction_model.confusion(X_test,y_test).tolist()

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

stats = {}

stats["prediciton train test accuracy"] = {"value":train_test_acc,"description":"Train and Test accuracy of prediction model"}
stats["prediction report"] = {"value":report,"description":"Report of prediction model"}
stats["prediction confusion"] = {"value":confusion,"description":"Confusion matrix of prediction model"}

utils.save_stats(stats)