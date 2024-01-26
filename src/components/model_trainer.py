from dataclasses import dataclass
import os
import joblib
from src.logger import logging
from src.components.data_ingestion import Ingestion
from src.components.data_preprocessing import Preprocessing
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials,space_eval
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score , precision_score, fbeta_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from src.utils.model_trainer_utils import start_validating_data,evalulate_train_data,evalulate_test_data

@dataclass
class Model_Training_files_dir():
    os.makedirs('artifacts/model',exist_ok=True)
    model_train_dir = 'artifacts/model/model.joblib'


class Model_Training():
    def __init__(self, X=None, y=None,standar_scalar_dir =None, pca_dir =None, test_data =None,best_params=None):
        self.X = X
        self.y = y 
        self.standar_scalar_dir = standar_scalar_dir
        self.pca_dir = pca_dir
        self.test_data = test_data
        self.best_params = best_params
        #self.clean_dir = clean_dir
        self.dir = Model_Training_files_dir()
    
    def objective_with_smote_pca(self,space):
        i = int(space['k_neighbour'])
        j = int(space['n_components'])

        smote = SMOTE(sampling_strategy='all', k_neighbors=i)
        smote_x, smote_y = smote.fit_resample(self.X, self.y)

        pca = PCA(n_components=j)
        pca_x = pca.fit_transform(smote_x)

        scaler = StandardScaler()
        standard_x = scaler.fit_transform(pca_x)

        model = XGBClassifier(
            learning_rate=space['learning_rate'],
            max_depth=space['max_depth'],
            min_child_weight=space['min_child_weight'],
            subsample=space['subsample'],
            gamma=space['gamma'],
            colsample_bytree=space['colsample_bytree'],
            n_estimators=space['n_estimators']
        )
        accuracy = cross_val_score(model, standard_x, smote_y, cv=5).mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': 1 - accuracy, 'status': STATUS_OK}


    def start_hyperparameter_tunning(self):

        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': hp.choice('max_depth', range(3, 15)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'n_estimators': hp.choice('n_estimators', range(50, 500)),
            'k_neighbour': hp.choice('k_neighbour', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'n_components': hp.choice('n_components', [5, 10, 15, 20, 25, 30, 35])
        }

        trials = Trials()
        best = fmin(fn=self.objective_with_smote_pca,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=80,
                    trials=trials)
        best_params = space_eval(space, best)
        print("Best Hyperparameters:")
        print(best_params)
        logging.info(f"Best Hyperparameters: {best_params}")
        return best_params
    
    def start_training(self, best_params=None):
        if best_params is not None:
            self.best_params = best_params
        else:
            self.best_params = self.start_hyperparameter_tunning()
        
        xgb_hyp = XGBClassifier(
                learning_rate=self.best_params['learning_rate'],
                n_estimators =self.best_params['n_estimators'],
                max_depth=self.best_params['max_depth'],
                min_child_weight=self.best_params['min_child_weight'],
                subsample=self.best_params['subsample'],
                colsample_bytree=self.best_params['colsample_bytree']
            )

        xgb_hyp.fit(self.X, self.y)
        joblib.dump(xgb_hyp,self.dir.model_train_dir)

        evalulate_train_data(self.X,self.y,xgb_hyp)

        evalulate_test_data(self.test_data,xgb_hyp)

        start_validating_data()

        return self.dir.model_train_dir 
    
