import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories datasets
    
    Input:
    messages_filepath: --string. Filepath for csv file containing messages dataset with id as unique identifier.
    categories_filepath: --string. Filepath for csv file containing categories dataset with id as unique identifier.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df =pd.merge(messages,categories,how='inner',on='id')
    
    return df


def clean_data(df):
    """
    Loads df and cleans the data-removing duplicates,converting to categories from strings..
    
    Input: Loads df dataset which is a join of messages and categories
    
    Return: Clean dataframe free from duplicates
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    # https://stackoverflow.com/questions/25254016/pandas-get-first-row-value-of-a-given-column
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] =  categories[column].apply(lambda x: x[-1:])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int') 
        
    # Drop child_alone from categories dataframe because it contains only 0.
    categories.drop('child_alone', axis = 1, inplace = True)
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True) 
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # Remove rows with a related value of 2 from the dataset
    df = df[df['related'] != 2]
    return df


def save_data(df, database_filename):
    """Save cleaned data into an SQLite database.
    
    Input:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    Returns:
    None
    """
    engine = create_engine('sqlite:///disaster.db')
    df.to_sql('my_disaster_response_table', engine, index=False)


def main():
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