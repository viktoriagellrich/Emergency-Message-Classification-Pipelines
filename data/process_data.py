import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Description: Load the data and merge it into one dataframe
    
    Arguments:
        message_filepath - the filepath where to find the data containing emergency messages
        categories_filepath  - the filepath where to find the data containing ctegories for emergency messages
    
    Returns:
        df - dataframe containing all the (merged) data
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    categories["id"].drop_duplicates(inplace=True)
    df = pd.merge(messages, categories, how="inner", on="id")
    
    return df

def clean_data(df):
    """
    Description: Transform the category information into boolean columns so that it can be used for modeling
    
    Arguments:
        df - dataframe to be transformed
        
    Returns:
        df - dataframe transformed for modeling
    """
    
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[slice(-2)])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    categories["related"] = categories["related"].replace(2,1)
    
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace = True) 
    
    return df


def save_data(df, database_filename):
    """
    Description: Save the dataframe in a sqlite database
    
    Arguments:
        df - dataframe to be saved
        database_filename: name of the file where the dataframe shall be saved
        
    Returns:
        None
    """
    
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('df', engine, index=False) 


def main():
    """
    Description: Load the data with the function load_data,
                 clean the data with the function clean_data
                 save the data with the function save_data.
                 Print an informative message in either case, if the filepath is right or wrong.
                 
    Arguments:
        None
    
    Returns:
        None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
