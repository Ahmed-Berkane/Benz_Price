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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV




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
    


def adj_r2(X, y, pred):
    
    '''
    
    This function calculates the adjusted r_2
    
    '''
    try:
        # calculate R^2
        r2 = r2_score(y, pred)
        
        # Get the number of samples (n) and number of features (k)
        n, k = X.shape[0], X.shape[1]  
        # Calculate adjusted R-squared
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        return adj_r2

    except Exception as e:
            raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    
    '''
    
    This function calculates the MAE, RMSE, and Adj_R2 for both Train and Test sets and for each model, 
    then return the results in Pandas DataFrame
    
    '''
    try:
        data = {}

        # Loop through models and their corresponding hyperparameters
        for model_name, model in models.items():
            # Get the hyperparameter grid for the current model
            para = param[model_name]

            # Perform GridSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Make predictions using the best model from GridSearchCV
            y_train_pred = gs.predict(X_train)
            y_test_pred = gs.predict(X_test)
            
            
            # Calculate the Mean Absolute Error
            train_MAE = mean_absolute_error(y_train, y_train_pred)
            test_MAE = mean_absolute_error(y_test, y_test_pred)
            
            # Calculate the Root Mean Square
            train_RMSE = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_RMSE = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # calculate Adjusted R^2
            train_adj_r2 = adj_r2(X_train, y_train, y_train_pred)
            test_adj_r2 = adj_r2(X_test, y_test, y_test_pred)
            
            # Add the results to the report dictionary for this model
            data[model_name] = {
                                    'train_mae': train_MAE,
                                    'train_rmse': train_RMSE,
                                    'train_adj_r2': train_adj_r2,
                                    'test_mae': test_MAE,
                                    'test_rmse': test_RMSE,
                                    'test_adj_r2': test_adj_r2
                                    }
            
            # Now we take the dictionnary and create a nice data frame 
            # Create a MultiIndex for columns
            columns = pd.MultiIndex.from_tuples(
                [('train', 'MAE'), ('train', 'RMSE'), ('train', 'Adj_R2'), 
                ('test', 'MAE'), ('test', 'RMSE'), ('test', 'Adj_R2')],
                names = ['set', 'metric']
            )
            
            # Create a DataFrame from the data
            report = pd.DataFrame(
                {('train', 'MAE'): [data[model]['train_mae'] for model in data],
                ('train', 'RMSE'): [data[model]['train_rmse'] for model in data],
                ('train', 'Adj_R2'): [data[model]['train_adj_r2'] for model in data],
                ('test', 'MAE'): [data[model]['test_mae'] for model in data],
                ('test', 'RMSE'): [data[model]['test_rmse'] for model in data],
                ('test', 'Adj_R2'): [data[model]['test_adj_r2'] for model in data]
                },
                index = data.keys(),
                columns = columns
            )
            

        return report 

    
    except Exception as e:
            raise CustomException(e, sys)
    


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
    



def app_data_prep(df, 
              cat_feature, 
              mapping_dict):
    '''
    This function clean the data for the web app:

    '''
    try:

        #Step1:
        df['Year'] = df[cat_feature].str[:4]
        df[cat_feature] = df[cat_feature].str[5:]
            
        #Step2:
        df['Year'] = pd.to_numeric(df['Year'])
            
        # Step3: Apply tge groupping of thin levels
        df[cat_feature] = df[cat_feature].map(mapping_dict)

        return df
        
    except Exception as e:
        raise CustomException(e, sys) 
    
  
        