#import libraries
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def upsample(X,y,encoder,scaler):
    categorical_values = X.select_dtypes(["object","datetime64[ns]"]).columns
    numerical_values = X.select_dtypes(["int","float"]).columns

    X_temp = X.copy()
    X_temp[categorical_values] = encoder.transform(X_temp[categorical_values])
    X_temp[numerical_values] = scaler.transform(X_temp[numerical_values])
    X_temp.dropna(inplace=True)
    index = X_temp.index
    y = y.loc[index]

    sm = SMOTE(sampling_strategy = 1 ,k_neighbors = 5, random_state=1)   #Synthetic Minority Over Sampling Technique
    X_upsample, y_upsample = sm.fit_resample(X_temp, y.ravel())

    return X_upsample, y_upsample

def downsample(X,y,encoder,scaler):
    categorical_values = X.select_dtypes(["object","datetime64[ns]"]).columns
    numerical_values = X.select_dtypes(["int","float"]).columns

    X_temp = X.copy()
    X_temp[categorical_values] = encoder.transform(X_temp[categorical_values])
    X_temp[numerical_values] = scaler.transform(X_temp[numerical_values])
    X_temp.dropna(inplace=True)
    index = X_temp.index
    y = y.loc[index]

    sm = RandomUnderSampler(random_state=1)    #Synthetic Minority down Sampling Technique
    X_downsample, y_downsample = sm.fit_resample(X_temp, y.ravel())

    return X_downsample, y_downsample

def batch_predict(X,pipeline,probability=False):
    """
    Perform batch prediction on a given dataset using a pre-trained pipeline

    Parameters:
        X (pandas.DataFrame): The input dataset for prediction.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for prediction.
        probability (bool): Whether to return probabilities for classification models. Default is False

    Returns:
        predictions (pandas.Series): The predicted values or probabilities.
    """

    categorical_values = X.select_dtypes(["object","datetime64[ns]"]).columns
    numerical_values = X.select_dtypes(["int","float"]).columns

    encoder = pipeline.named_steps["encoder"]
    scaler = pipeline.named_steps["scaler"]
    estimator = pipeline.named_steps["estimator"]

    X_temp = X.copy()
    X_temp[categorical_values] = encoder.transform(X_temp[categorical_values])
    X_temp[numerical_values] = scaler.transform(X_temp[numerical_values])
    X_temp = X_temp.dropna()

    if probability:
        return pd.Series(estimator.predict_proba(X_temp)[:,1],index=X_temp.index)
    else:
        return pd.Series(estimator.predict(X_temp),index=X_temp.index)

def batch_evaluate(X,y,pipeline):
    """
    Perform batch evaluation on a given dataset using a pre-trained pipeline.

    Parameters:
        X (pandas.DataFrame): The input dataset for evaluation.
        y (pandas.Series): The true labels for evaluation.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for evaluation.

    Returns:
        accuracy (float): The accuracy score of the predictions.
    """
    yh_temp = batch_predict(X,pipeline)
    y_temp = y[yh_temp.index]
    return accuracy_score(y_temp,yh_temp)

def batch_report(X,y,pipeline):
    """
    Perform batch classification report on a given dataset using a pre-trained pipeline.

    Parameters:
        X (pandas.DataFrame): The input dataset for report.
        y (pandas.Series): The true labels for report.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for report.

    Returns:
        report (dict): The classification report containing precision, recall, f1-score, and support.
    """

    yh_temp = batch_predict(X,pipeline)
    y_temp = y[yh_temp.index]
    return classification_report(y_temp,yh_temp,output_dict=True)

def batch_confusion(X,y,pipeline):
    """
    Perform batch confusion matrix on a given dataset using a pre-trained pipeline.

    Parameters:
        X (pandas.DataFrame): The input dataset for confusion matrix.
        y (pandas.Series): The true labels for confusion matrix.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for confusion matrix.

    Returns:
        matrix (numpy.ndarray): The confusion matrix.
    """
    yh_temp = batch_predict(X,pipeline)
    y_temp = y[yh_temp.index]
    return confusion_matrix(y_temp,yh_temp)

def impute_predict(X,X_train,pipeline,product_wpwd,customer_wpwd,probability=False):
    """
    Perform imputation and prediction on a given dataset using a pre-trained pipeline.

    Parameters:
        X (pandas.DataFrame): The input dataset for imputation and prediction.
        X_train (pandas.DataFrame): The training dataset used for imputation.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for prediction.
        product_wpwd : pairwise distance of products
        customer_wpwd : paairwise distance of customers
        probability (bool, optional): Flag indicating whether to return probabilities. Defaults to False.

    Returns:
        output (pandas.DataFrame): The predicted outcomes and imputation flags.
    """

    intersection_products = set(X_train["srp_2_desc"]).intersection(product_wpwd.index)
    intersection_customers = set(X_train["customer_num"]).intersection(customer_wpwd.index)

    union_products = set(X_train["srp_2_desc"]).union(product_wpwd.index)
    union_customers = set(X_train["customer_num"]).union(customer_wpwd.index)

    output = []
    for customer_num,srp_2_desc in X[["customer_num","srp_2_desc"]].values:
        if (customer_num in union_customers) & (srp_2_desc in union_products):
            ref = X_train[(X_train["customer_num"]==customer_num) & (X_train["srp_2_desc"]==srp_2_desc)]
            imputed = 0
            if len(ref)==0:
                imputed = 1
                closest_srp_2_desc = srp_2_desc
                closest_customer_num = customer_num
                if closest_srp_2_desc not in list(X_train["srp_2_desc"]):
                    closest_srp_2_desc = product_wpwd.loc[srp_2_desc,intersection_products].idxmin()
                if closest_customer_num not in list(X_train["customer_num"]):
                    closest_customer_num = customer_wpwd.loc[customer_num,intersection_customers].idxmin()
                ref = X_train[(X_train["customer_num"]==closest_customer_num) | (X_train["srp_2_desc"]==closest_srp_2_desc)]
            prediction = batch_predict(ref,pipeline,probability).mode()[0]
        else:
            imputed = -1
            unknown_customer = 0
            unknown_product = 0
            if customer_num not in union_customers:
                unknown_customer = -1
            if srp_2_desc not in union_products:
                unknown_product = -2
            prediction = unknown_product+unknown_customer
        output.append([customer_num,srp_2_desc,prediction,imputed])
    output = pd.DataFrame(output,columns=["customer_num","srp_2_desc","outcome","imputed"])[["outcome","imputed"]]

    return output

def impute_predict_from_lists(customer_list,srp_2_list,X_train,pipeline,product_wpwd,customer_wpwd,probability=False):
    """
    Perform imputation and prediction on a given list of customers and products using a pre-trained pipeline.

    Parameters:
        customer_list (list): List of customer numbers.
        srp_2_list (list): List of product descriptions.
        X_train (pandas.DataFrame): The training dataset used for imputation.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for prediction.
        product_wpwd : pairwise distance of products
        customer_wpwd : paairwise distance of customers
        probability (bool, optional): Flag indicating whether to return probabilities. Defaults to False.

    Returns:
        X (pandas.DataFrame): The input data with predicted outcomes and imputation flags.
    """


    X = pd.DataFrame([[customer_num,srp_2_desc] for customer_num in customer_list for srp_2_desc in srp_2_list],columns=["customer_num","srp_2_desc"])
    X[["outcome","imputed"]] = impute_predict(X,X_train,pipeline,product_wpwd,customer_wpwd,probability)
    return X

def impute_evaluate(X,y,X_train,pipeline,product_wpwd,customer_wpwd):
    """
    Perform imputation, prediction, and evaluate the accuracy of predictions on the given input data.

    Parameters:
        X (pandas.DataFrame): Input data containing customer and product information.
        y (pandas.Series): True labels or outcomes corresponding to the input data.
        X_train (pandas.DataFrame): The training dataset used for imputation.
        pipeline (sklearn.pipeline.Pipeline): The pre-trained pipeline for prediction.
        product_wpwd : pairwise distance of products
        customer_wpwd : paairwise distance of customers

    Returns:
        accuracy (float): Accuracy of the predictions compared to the true labels.
    """
    yh = impute_predict(X,X_train,pipeline,product_wpwd,customer_wpwd)["outcome"]
    return accuracy_score(y,yh)

def feature_importance(X,y,pipeline):
    categorical_values = X.select_dtypes(["object","datetime64[ns]"]).columns
    numerical_values = X.select_dtypes(["int","float"]).columns

    encoder = pipeline.named_steps["encoder"]
    scaler = pipeline.named_steps["scaler"]
    estimator = pipeline.named_steps["estimator"]

    X_temp = X.copy()
    X_temp[categorical_values] = encoder.transform(X_temp[categorical_values])
    X_temp[numerical_values] = scaler.transform(X_temp[numerical_values])
    X_temp = X_temp.dropna()
    y_temp = y.loc[X_temp.index]

    result = permutation_importance(estimator, X_temp, y_temp, n_repeats=5,random_state=0)
    feature_imp = pd.DataFrame({'features':X.columns , 'score':result['importances_mean']})
    return feature_imp

class CustomClassifierModel:

    def __init__(self, X_train, pipeline,product_wpwd,customer_wpwd):
        """
        Initializes the CustomClassifierModel with the provided parameters.

        Args:
            X_train: The training data.
            pipeline: The classifier pipeline.
            product_wpwd: Product data with pairwise distances.
            customer_wpwd: Customer data with pairwise distances.
        """
        self.X_train = X_train
        self.pipeline = pipeline
        self.product_wpwd = product_wpwd
        self.customer_wpwd = customer_wpwd
        
    def predict(self,X,probability=False):
        """
        Performs batch prediction on the given data using the classifier pipeline.

        Args:
            X: The input data.
            probability: Whether to return probability estimates.

        Returns:
            The predictions.
        """
        return batch_predict(X,self.pipeline,probability)
    
    def evaluate(self,X,y):
        """
        Evaluates the classifier pipeline on the given data.

        Args:
            X: The input data.
            y: The true labels.

        Returns:
            The evaluation metrics.
        """
        return batch_evaluate(X,y,self.pipeline)
    
    def report(self,X,y):
        """
        Generates a classification report for the classifier pipeline on the given data.

        Args:
            X: The input data.
            y: The true labels.

        Returns:
            The classification report.
        """
        return batch_report(X,y,self.pipeline)
    
    def confusion(self,X,y):
        """
        Generates a confusion matrix for the classifier pipeline on the given data.

        Args:
            X: The input data.
            y: The true labels.

        Returns:
            The confusion matrix.
        """
        return batch_confusion(X,y,self.pipeline)

    def impute_predict(self,X,probability=False):
        """
        Performs imputed prediction on the given data using the classifier pipeline.

        Args:
            X: The input data.
            probability: Whether to return probability estimates.

        Returns:
            The imputed predictions.
        """

        return impute_predict(X,self.X_train,self.pipeline,self.product_wpwd,self.customer_wpwd,probability)

    def impute_evaluate(self,X,y):
        """
        Evaluates the classifier pipeline on imputed data using the given data.

        Args:
            X: The input data.
            y: The true labels.

        Returns:
            The evaluation metrics.
        """
        return impute_evaluate(X,y,self.X_train,self.pipeline,self.product_wpwd,self.customer_wpwd)
    
    def predict_from_lists(self,customer_list,srp_2_list,probability=False):
        """
        Performs imputed prediction from lists of customer and product data using the classifier pipeline.

        Args:
            customer_list: The list of customers.
            srp_2_list: The list of products.
            probability: Whether to return probability estimates.

        Returns:
            The imputed predictions.
        """
        return impute_predict_from_lists(customer_list,srp_2_list,self.X_train,self.pipeline,self.product_wpwd,self.customer_wpwd,probability)
    
    def predict_from_lists_with_remark(self,customer_list,srp_2_list):
        """
        Performs imputed prediction from lists of customer and product data using the classifier pipeline,
        and provides additional remarks and history information.

        Args:
            customer_list: The list of customers.
            srp_2_list: The list of products.

        Returns:
            The imputed predictions with remarks and history.
        """
        output = self.predict_from_lists(customer_list,srp_2_list)
        output["outcome"] = output["outcome"].replace({
            1:"Near Sale",
            0:"Far Sale",
            -1:"No Remark (Unknown Customer)",
            -2:"No Remark (Unknown Product)",
            -3:"No Remark (Unknown Customer and Product)"})
        output["imputed"] = output["imputed"].apply(lambda x: "Known Pair" if x==0 else "New Pair")
        output.rename(columns={"customer_num":"Customer","srp_2_desc":"Product","outcome":"Remark","imputed":"History"},inplace=True)
        return output