import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.exception import CustomException
from src.logger import logging
from src.utils import data_prep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig





@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    dict_data_path: str=os.path.join('artifacts',"mapping_dict.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/usa_mercedes_benz_prices.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            
            df, mapping_dict = data_prep(df = df, 
                                         group_col = 'Name', 
                                         agg_col = 'Price',
                                         n_level = 2,
                                         low_threshold = 50000, 
                                         medium_threshold = 100000)
            
            # Group Name categories with one and two occurances into Other_low, Other_medium, and Other_high 
            df['Name'] = df['Name'].map(mapping_dict)
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Raw data saved successfully")
            
             # Convert mapping_dict to DataFrame
            mapping_df = pd.DataFrame(list(mapping_dict.items()), columns=['Name', 'MappedName'])
            
            # Save the DataFrame as a CSV file
            mapping_df.to_csv(self.ingestion_config.dict_data_path, index = False, header = True)
            
            logging.info(f"Mapping dictionary saved to {self.ingestion_config.dict_data_path}")
            
            # Train-test split
            logging.info("Train test split initiated")      
            train_set, test_set = train_test_split(df, 
                                                   stratify = df['Name'],
                                                   test_size = 0.3, 
                                                   random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path,
                self.ingestion_config.dict_data_path
            )
            
        except Exception as e:
           raise CustomException(e, sys)
       
       
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path, raw_path, dict_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path, dict_path)

        
