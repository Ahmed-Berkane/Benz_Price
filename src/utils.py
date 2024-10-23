import os
import sys

from src.exception import CustomException
from src.logger import logging

import numpy as np 
import pandas as pd
import re




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