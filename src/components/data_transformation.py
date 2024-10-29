import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import data_prep, AddNumFeatures, AddCatFeatures, save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
    
class DataTransformation:
    
    '''
    This function is responsible for data transformation in pipelines
    
    '''
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
              
    def get_data_transformer_object(self, mapping_dict):
        try:
            num_attribs = ["Mileage", "Rating", "Review Count", "Year"]
            cat_attribs = ['Name']
            
            logging.info("Numerical Columns Standard Scaling Started")
            num_pipeline = Pipeline(
                         steps =[
                         ('new_features', AddNumFeatures(add_New_Features = False)),
                         ('imputer', SimpleImputer(strategy="median")),
                         ('std_scaler', StandardScaler())
                                 ]
                           )
            logging.info("Numerical Columns Standard Scaling Completed")
            
            cat_pipeline = Pipeline(
                            steps = [
                            ('adding_new_name', AddCatFeatures(mapping_dict = mapping_dict, add_Name = False)), # False when training model, and True when scoring
                            ('onehot', OneHotEncoder(drop='first'))      # One-hot encode with drop first column
                                      ]
                              )
            logging.info("Catigorical Columns Encoding Completed")
            
            preprocessor = ColumnTransformer(
                                    [
                                    ("num", num_pipeline, num_attribs),
                                    ("cat", cat_pipeline, cat_attribs)
                                        ]
                                    )
            
            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path, dict_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # Load the mapping dictionary from the CSV file into a DataFrame
            mapping_df = pd.read_csv(dict_path)
            # Convert the DataFrame back into a dictionary
            mapping_dict = dict(zip(mapping_df['Name'], mapping_df['MappedName']))
            
            logging.info("Read the train and test data completed")
            
            logging.info(f"Mapping dictionary loaded from {dict_path}")
            
            
            
            logging.info("Obtaining preprocessor object")
            
            preprocessing_obj = self.get_data_transformer_object(mapping_dict)
            
            target_column_name = 'Log_Price'
            num_attribs = ["Mileage", "Rating", "Review Count", "Year"]
            cat_attribs = ['Name']
            
            input_feature_train_df = train_df.drop(columns = ['Log_Price', 'Price'], axis = 1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns = ['Log_Price', 'Price'], axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
             # Check the shapes before transformation
            logging.info(f"Train input shape: {input_feature_train_df.shape}")
            logging.info(f"Train target shape: {target_feature_train_df.shape}")
            logging.info(f"Test input shape: {input_feature_test_df.shape}")
            logging.info(f"Test target shape: {target_feature_test_df.shape}")
            
            
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Train Preprocessed shape: {input_feature_train_arr.shape}")
            logging.info(f"Test Preprocessed  shape: {input_feature_test_arr.shape}")
            
            
            logging.info("Combining transformed features with the target feature.")
            
            # Convert the sparse arrays to dense to be abale to concatenated with the target
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = input_feature_test_arr.toarray()
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df).reshape(-1,1)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df).reshape(-1,1)]
            
            logging.info("Saved preprocessing object.")  
            
            save_object(
                        file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj
                            )
            
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                )                    

        
        except Exception as e:
            raise CustomException(e, sys)