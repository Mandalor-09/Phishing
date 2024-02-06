from dataclasses import dataclass
from src.logger import logging
import pandas as pd
import os
from src.utils.ingestion_utils import DataCleaning,concat_x_y
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

@dataclass
class Ingestion_files_dir():
    os.makedirs('artifacts/cleaned_data/',exist_ok=True)
    cleaned_data = 'artifacts/cleaned_data/final_data.csv'
    train_data = 'artifacts/cleaned_data/train_data.csv'
    test_data = 'artifacts/cleaned_data/test_data.csv'

class Ingestion():
    def __init__(self,file):
        self.file = file
        self.dir = Ingestion_files_dir()

    def start_ingestion(self):
        df = pd.read_csv(self.file)
        logging.info(f'Data Read Succesfully from {self.file}')
        logging.info('Initiating Data Ingestion Method')
        clean = DataCleaning(df, 0.8, 0.8)
        df = clean.feature_scaling_df()

        phished = df[df['phishing'] == 1]
        not_phished = df[df['phishing'] == 0]
        logging.info(f"Shape of Phished : {phished.shape},Shape of Non Phished is : {not_phished.shape}")

        resample_phished = resample(phished,replace= True,n_samples=45000,random_state=42)
        resample_not_phished = resample(not_phished,replace= True,n_samples=45000,random_state=42)
        df_final = pd.concat([resample_phished,resample_not_phished],axis=0)
        df_final = df_final.sample(df_final.shape[0])

        logging.info(f"Resampleing Completed Succesfully The Value counts are {df['phishing'].value_counts()}")
        #print(f"Smoting Completed Succesfully The Vale counts are {df['phishing'].value_counts()}")
        df_final.to_csv(self.dir.cleaned_data,index=False)

        logging.info(f'Clean Data Saved to {self.dir.cleaned_data}')

        X = df_final.drop('phishing',axis=1)
        y = df_final['phishing']
        X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.25, random_state=42)
        train_df = concat_x_y(X_train,y_train)
        train_df.to_csv(self.dir.train_data,index=False)
        logging.info(f'Training Dataset Saved at {self.dir.train_data}')

        test_df = concat_x_y(X_test,y_test)
        test_df.to_csv(self.dir.test_data,index=False)
        logging.info(f'Testing Dataset Saved at {self.dir.test_data}')
        logging.info('Completed Data Ingestion Method')
        return {
            'cleaned_data' : self.dir.cleaned_data,
            'train_data' : self.dir.train_data,
            'test_data' : self.dir.test_data
        }