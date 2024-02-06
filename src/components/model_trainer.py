from dataclasses import dataclass
import os
import joblib
from src.logger import logging
from hyperopt import tpe
from src.utils.model_trainer_utils import start_validating_data,evalulate_train_data,evalulate_test_data
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import tpe
from hpsklearn import HyperoptEstimator,any_preprocessing,k_neighbors_classifier

@dataclass
class Model_Training_files_dir():
    os.makedirs('artifacts/model',exist_ok=True)
    model_train_dir = 'artifacts/model/model.joblib'


class Model_Training():
    def __init__(self, X=None, y=None,test_data =None,cleaned_data = None):
        self.X = X
        self.y = y 
        self.test_data = test_data
        self.cleaned_data= cleaned_data
        self.dir = Model_Training_files_dir()


    def start_hyperparameter_tunning(self):
        logging.info('Initiated Hyper Parameter Tunning')

        estim = HyperoptEstimator(
            classifier=k_neighbors_classifier('my_knn'),
            preprocessing=any_preprocessing('my_pre'),
            algo=tpe.suggest,
            max_evals=100,
            trial_timeout=120,
        )

        # Fit the estimator
        estim.fit(self.X, self.y)

        # Get the best parameters and score
        best_params = estim._best_learner
        best_score = estim._best_loss
        logging.info(f'Completed Hyper Parameter Tunning Best Parameter Achieved is {best_params}')
        return best_params

    def start_training(self,model):
        logging.info('Initiated Model Training')
        if model == None:
            logging.info('Model Parameter Not found')
            knn_model = KNeighborsClassifier(algorithm='kd_tree', leaf_size=22, n_jobs=1, n_neighbors=2,
                        p=2.372428226692859, weights='distance')

            knn_model.fit(self.X, self.y)

        else:
            logging.info('Model Parameter Found')
            knn_model = model
            knn_model.fit(self.X, self.y)

        joblib.dump(knn_model,self.dir.model_train_dir)

        evalulate_train_data(self.X,self.y,knn_model)

        evalulate_test_data(self.test_data,knn_model)

        start_validating_data()
        logging.info(f'Completed Model Training')
        return self.dir.model_train_dir 
    
