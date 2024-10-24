import os
import sys

from src.exception import CustomException
from src.logger import logging

import numpy as np 
import pandas as pd
import re
import dill  
import pickle
from sklearn.base import BaseEstimator, TransformerMixin




def data_prep(df, 
              group_col, 
              agg_col,
              n_level,
              low_threshold, 
              medium_threshold, 
              drop_na_col = True,
              drop_duplicates_col = True,
              log_transform = True,
              mileage_outliers = True):
    
  
    '''
    
    This function simplifies data preparation by:
    1. Extracting year from Name and updating the Name column.
    2. Converting specified columns to numeric, cleaning non-numeric characters.
    3. Droping any na's in the Target, duplcates rows, transforming the Target to Log, and removes potential outliers
    4. Grouping and categorizing rows based on aggregation statistics of the agg_col.
    It returns a clean data with the mapping_dict to group some levels if needed as the DataPrepResult class.
    
    '''
    try:
        #Step1:
        df['Year'] = df[group_col].str[:4]
        df[group_col] = df[group_col].str[5:]
        
        #Step2:
        col_name_convert = ['Mileage', 'Review Count', 'Price', 'Year']
        
        for col in col_name_convert:
            df[col] = pd.to_numeric(df[col].str.replace(r'\D', '', regex = True), errors = 'coerce')
        

        #Step3:
        
        # Drop rows with missing values in the agg_col column, drop_na_col = True
        if drop_na_col:
            df = df.dropna(subset = agg_col)
        
        
        if drop_duplicates_col:
            df = df.drop_duplicates()
            
        # Apply log transformation to the target variable
        if log_transform:
            df['Log_Price'] = np.log(df[agg_col])
            
        # Removing the potential outliers by setting Mileage less than 125,000
        if mileage_outliers:
            df = df[df['Mileage'] < 125000]
        
        
        #Step4:

        # Group by the specified column and calculate count and average of the aggregation column
        df_group = df.groupby(group_col).agg(
            Name_Count=(group_col, 'count'),
            Avg_Agg_Value=(agg_col, 'mean')
        ).reset_index()

        # Categorize based on thresholds
        df_group.loc[(df_group['Name_Count'] <= n_level) & (df_group['Avg_Agg_Value'] <= low_threshold), 'Adj_Name'] = 'Other_Low'
        df_group.loc[(df_group['Name_Count'] <= n_level) & (df_group['Avg_Agg_Value'] > low_threshold) & 
                    (df_group['Avg_Agg_Value'] <= medium_threshold), 'Adj_Name'] = 'Other_Medium'
        df_group.loc[(df_group['Name_Count'] <= n_level) & (df_group['Avg_Agg_Value'] > medium_threshold), 'Adj_Name'] = 'Other_High'
        df_group.loc[df_group['Name_Count'] > n_level, 'Adj_Name'] = df_group[group_col]

        # Merge the aggregated results back to the original DataFrame
        df = df.merge(df_group, on = group_col)
        
        # Create a mapping dictionary for future use when we will have new unseen data.
        mapping_dict = dict(zip(df['Name'], df['Adj_Name']))
        
        # Dropping the newly created columns that we don't need any more
        df = df.drop(columns = ['Name_Count', 'Avg_Agg_Value', 'Adj_Name'], axis = 1)

        # Return the result as a custom object
        return df, mapping_dict


    except Exception as e:
        raise CustomException(e, sys)
    
   
   
    
    

class AddCatFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict, add_Name = True):
        self.mapping_dict = mapping_dict
        self.add_Name = add_Name

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        
        try:
        
            Name_ix = 0
        
            if isinstance(X, pd.Series):
                X = X.to_frame()  # Convert Series to DataFrame if necessary

            if self.add_Name:
                # Map the 'Name' column to 'New_Name' using the mapping dictionary
                X.iloc[:, Name_ix] = X.iloc[:, Name_ix].map(self.mapping_dict)

                # Handle missing values, if required
                X.iloc[:, Name_ix] = X.iloc[:, Name_ix].fillna('Not Mapped')

            return X  # Return the transformed DataFrame
    
        except Exception as e:
            raise CustomException(e, sys)
    
    
 
 
    
class AddNumFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, add_New_Features = True):
        self.add_New_Features = add_New_Features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        try:
            
            Mileage_ix, Rating_ix, Review_Count_ix, Year_ix = 0, 1, 2, 3
            
            if self.add_New_Features:
            
                # Set Review Count to 1 if it is 0 to avoid division by zero
                X.iloc[X.iloc[:, Review_Count_ix] == 0, Review_Count_ix] = 1

                # Threshold for minimum number of reviews
                m = 100
                # Global average rating across all products
                C = X.iloc[:, Rating_ix].mean()

                Credibility = ((X.iloc[:, Rating_ix] * X.iloc[:, Review_Count_ix]) + (C * m)) / (X.iloc[:, Review_Count_ix] + m)

                # Add Year Mileage interaction
                Mil_Year = X.iloc[:, Mileage_ix] * X.iloc[:, Year_ix]

                X = np.c_[X, Credibility, Mil_Year]

                X = np.delete(X, [Mileage_ix, Rating_ix, Review_Count_ix, Year_ix], axis=1)
            
            return X
    
        except Exception as e:
            raise CustomException(e, sys)
        
        
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)