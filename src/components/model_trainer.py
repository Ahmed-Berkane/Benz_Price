import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import  save_object, evaluate_models, adj_r2

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor




@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test data')
            X_train, y_train, X_test, y_test = (
                                                train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                                  )
            # Creating dictionary of the models to run and their hyperparameters 
            models = {
                        "Linear Regression": LinearRegression(),
                        "Elastic net Regression": ElasticNet(),
                        "Random Forest": RandomForestRegressor(),
                        "Gradient Boosting": GradientBoostingRegressor(),
                        "XGBRegressor": XGBRegressor()
                            }

            params = {
                        "Linear Regression":{
                            
                                                },
                
                        "Elastic net Regression":{
                                                    'alpha':[0.1,0.5,1,2,5],
                                                    'l1_ratio':[0.1,0.5,1]
                                                    },

                        "Random Forest":{
                                            'n_estimators': [3, 10, 30],
                                            'max_features': [2, 4, 6, 8]
                                            },
                
                        "Gradient Boosting":{
                                                'n_estimators': [100, 200, 300],        # Number of trees
                                                'learning_rate': [0.01, 0.05, 0.1]     # Step size shrinkage
                                                },
                
                        "XGBRegressor":{
                                            'n_estimators': [100, 200, 300],        # Number of trees
                                            'learning_rate': [0.01, 0.05, 0.1]     # Step size shrinkage
                                            },
                        }

            model_report = evaluate_models(X_train = X_train, 
                                           y_train = y_train, 
                                           X_test = X_test, 
                                           y_test = y_test, 
                                           models = models, 
                                           param = params)
            
            best_model_name = model_report['test', 'Adj_R2'].idxmax()
            
            best_model = models[best_model_name]
            
            logging.info("Best Model Found")
            best_model.fit(X_train, y_train)
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
                    )
        
            predicted = best_model.predict(X_test)
            ADJ_R2 = adj_r2(X_test, y_test, predicted)
            
            return ADJ_R2
            


            
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
        
