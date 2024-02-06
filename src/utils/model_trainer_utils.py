import pandas as pd
import numpy as np 
import joblib
from sklearn.metrics import accuracy_score,fbeta_score,recall_score,precision_score,confusion_matrix
from src.logger import logging
from src.utils.preprocessing_utils import data_split

def evalulate_train_data(X,y,knn_model):
    # Evaluate the performance of the XGBoost classifier
    y_pred_knn_model = knn_model.predict(X)
    accuracy = accuracy_score(y,y_pred_knn_model)
    precision = precision_score(y, y_pred_knn_model)
    conf_matrix = confusion_matrix(y, y_pred_knn_model)
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    f2_score = fbeta_score(y, y_pred_knn_model, beta=2)
    logging.info('Trainind Data')
    logging.info(f'Accuracy of KNN using HyperOPT: {accuracy}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F2 Score: {f2_score}')

def evalulate_test_data(test_data,knn_model):
    #Test Data 
    xt, yt = data_split(test_data)
    logging.info(f'Test Data Loaded Successfully')

    # Load scaler and pca models
    pca = joblib.load('artifacts\components\pca.joblib')
    logging.info(f'PCA Model loaded Successfully')
    xt = pca.transform(xt)

    scaler = joblib.load('artifacts\components\standard.joblib')
    logging.info(f'Standardization Model loaded Successfully')
    xt = scaler.transform(xt)

    y_pred_test = knn_model.predict(xt)
    test_acc = accuracy_score(yt, y_pred_test)
    precision = precision_score(yt, y_pred_test)
    conf_matrix = confusion_matrix(yt, y_pred_test)
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    f2_score = fbeta_score(yt, y_pred_test, beta=2)
    logging.info('Testing Data')
    logging.info(f'Accuracy of KNN using HyperOPT: {test_acc}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F2 Score: {f2_score}')

def start_validating_data():

    df_small = pd.read_csv('data\dataset_small.csv')
    select_columns = list(pd.read_csv(r'artifacts\cleaned_data\final_data.csv').columns)
    df_small = df_small[select_columns]
    x_t = df_small.drop('phishing',axis=1)
    y_t = df_small['phishing']

    pca = joblib.load('artifacts\components\pca.joblib')
    pca_xt = pca.transform(x_t)

    std = joblib.load('artifacts\components\standard.joblib')
    std_xt = std.transform(pca_xt)

    #Validate details
    knn_model = joblib.load('artifacts\model\model.joblib')
    y_pred = knn_model.predict(std_xt)
    accuracy = accuracy_score(y_t,y_pred)
    precision = precision_score(y_t,y_pred)
    recall = recall_score(y_t,y_pred)
    f2_scoree = fbeta_score(y_t,y_pred,beta=2)
    logging.info(f'Test data Details : \n accuracy = {accuracy} , \n precision = {precision} ,\n recall = {recall} ,\n f2_scoree = {f2_scoree}')