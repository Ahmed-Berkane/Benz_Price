import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException

from src.utils import load_object, app_data_prep



class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            dict_path = 'artifacts\mapping_dict.csv'
            
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            
            # Load the mapping dictionary from the CSV file into a DataFrame
            mapping_df = pd.read_csv(dict_path)
            # Convert the DataFrame back into a dictionary
            mapping_dict = dict(zip(mapping_df['Name'], mapping_df['MappedName']))
            
            features = app_data_prep(df = features, 
                                    cat_feature = 'Name',
                                    mapping_dict  = mapping_dict)
            
            data_scaled = preprocessor.transform(features)
            
            preds = model.predict(data_scaled)
            
            preds = np.exp(preds)
            
            return preds
        

        except Exception as e:
            raise CustomException(e, sys)
    
    
    
class CustomData:
    def __init__(self,
                 name: str, 
                 mileage: int, 
                 rating: float,  
                 review_count: int
                  ):
        
        self.name = name
        self.mileage = mileage
        self.rating = rating
        self.review_count = review_count
        
        
        
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict = {
              
              "Name": [self.name], 
              "Mileage": [self.mileage],
              "Rating": [self.rating], 
              "Review Count": [self.review_count] 
                
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
        
        