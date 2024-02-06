# Project is live at 
https://phished-url-checker.onrender.com/

# Phished Url Detector
This project aims to build a machine model that can detect a given URL isished or not. project is divided into several modules, responsible for a specific in the pipeline.

# Train Pipeline

## Overview

The `train_pipeline.py` file is the main component of the training pipeline for the machine learning project. It features the `Train_Pipeline` class, which outlines the steps for data ingestion, preprocessing, and model training. The pipeline can be executed with or without hyperparameter tuning.

## Prerequisites

- Python 3.11.4
- Required libraries: pandas, numpy, sklearn, xgboost, hyperopt

## How to Run

1. Install the required libraries.

   ```python
        pip install -r requirements.txt
        pip install git+https://github.com/hyperopt/hyperopt-sklearn
   ```
2. Run the following cammand fot Training

    ```python
        python src/pipeline/train_pipeline.py
    ```
    The above cammand will run the training pipeline and you will get an option to select wheather you wish to train the model on the best parameter which is already provided or want to redo Hyperparameter Tunning. Press "N" or "n" to train on the best parameter which is already provided.
3. Run following cammand to run the Streamlit app

    ```python
        streamlit run app.py
    ```

# Phishing URL Detection Prediction Pipeline

This repository features a Python script designed for extracting features from a URL and predicting whether it is a phishing URL or not, utilizing a pre-trained machine learning model.

## Features

The feature extraction functions include:

- **URL-based features:** dot, hyphen, underscore, slash, question mark, equal, at, exclamation, space, tilde, comma, plus, percent, TLD, and length.
- **Domain-based features:** dot, hyphen, underscore, at, vowels, domain in IP, and server-client.
- **Page-based features:** dot, hyphen, underscore, percent, and length.
- **File-based feature:** length.
- **Params-based features:** dot, hyphen, underscore, slash, question mark, and percent.
- **Additional features:** email in URL, TLS/SSL certificate, and URL shortened.

## Usage

To use the prediction pipeline, simply call the `prediction()` function with a URL as input:

```python
prediction = prediction("https://www.example.com/page/file?param1=value1&param2=value2")
```

# Datagestion
The first step in the pipeline is data ingestion. This module is responsible for reading the raw data, cleaning it, and splitting it into training and testing sets. The data is ingested from a CSV file named dataset_full.csv.

# Ingestion Class
The Ingestion class is responsible for starting the data ingestion process. It takes in two arguments:

file: The name of the CSV file containing the raw data.
k_neighbors: The number of neighbors to use while preparing SMOTE the data .
The start_ingestion method performs the following steps:

1. Reads the CSV file using pd.read_csv.
2. Cleans the data using the DataCleaning class.
3. Scales the features using the feature_scaling_df method.
4. Splits the data into training and testing sets using train_test_split.
5. Saves the training and testing sets as CSV files.
6. Here's an example usage of the Ingestion class:

```python
    ingestion = Ingestion('dataset_full.csv', 5)
    ingestion.start_ingestion()
```
It returns cleaned_data,train_data,test_data Directory 
1. cleaned_data = artifacts/cleaned_data/final_data.csv
2. train_data = artifacts/cleaned_data/train_data.csv
3. test_data = artifacts/cleaned_data/test_data.csv

# Data Preprocessing
The next step is data preprocessing. This module is responsible for transforming the data into a format that is suitable for training the machine learning model. This includes feature scaling, dimensionality reduction, and removing outliers.

# Preprocessing Class
The Preprocessing class is responsible for preprocessing the data. It takes in four arguments:

1. cleaned_data: The cleaned data from the DataCleaning class.
2. train_data: The training data from the Ingestion class.
3. test_data: The testing data from the Ingestion class.
4. n_components: The number of components to use in PCA for dimensionality reduction.
5. The preprocessing method performs the following steps:

```python
Applies PCA to the features using 
pca = PCA(n_components=7).
x_pca = pca.fit_transform(X,y)
```
It returns X,y,pca_dir,standard_dir in form of dictionary
1. X = Standarized train_x
2. y = y parameter of train_x
3. pca_dir = artifacts/components/pca.joblib
4. standard_dir = artifacts/components/standard.joblib

# Model Trainer

## Overview

The `Model_Trainer` class is designed for training a machine learning model for phishing detection. This class takes training data, test data, and cleaned data as input. It utilizes hyperparameter tuning through the Hyperopt library and trains the model based on the best hyperparameters. The trained model is then saved to disk using the joblib library.

## Methods

### `start_hyperparameter_tuning(self)`

This method performs hyperparameter tuning using the Hyperopt library. It defines a search space for hyperparameters and utilizes the `fmin` function to search for the best hyperparameters. The best hyperparameters are printed to the console and returned.

### `start_training(self, best_params)`

This method trains the machine learning model using the best hyperparameters obtained from the hyperparameter tuning process. It creates an instance of the `KNNClassifier` class from the sklearn.meighbours library with the best hyperparameters and fits it to the training data. The trained model is then saved to disk using the joblib library.

The method also evaluates the trained model on both the training and test datasets using functions from the `model_trainer_utils` module. Additionally, it performs model validation on a separate validation dataset using the `start_validating_data` function from the same module.

## Usage

Ensure that all dependencies are installed and configured properly. You can then instantiate the `Model_Training` class and use its methods for hyperparameter tuning and model training. Make sure to adjust the class parameters and dependencies according to your specific use case.

It returns model_dir Directory 
1. model_dir = artifacts/model/model.joblib

Note:
- The provided best parameter gave following result:
    ### Train Data
    1. Accuracy : 0.9751379310344828
    2. Precision: 0.9962390158172232
    3. Recall: 0.9537774725274726
    4. F2 Score: 0.96186710825916

    ### Test Data
    1. Accuracy : 0.9516666666666667
    2. Precision: 0.9810344827586207
    3. Recall: 0.9219047619047619
    4. F2 Score: 0.9317241241241241

    ### Validate Dataset
    1. Accuracy: 0.9251155256202575
    2. Precision: 0.9619047619047619
    3. Recall: 0.8806773909354913
    4. F2 Score: 0.90425176155492
