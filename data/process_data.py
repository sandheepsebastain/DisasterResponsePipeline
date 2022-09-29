# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,how="inner",on=['id'])
    return df

def clean_data(df):
    categories=df[['id','categories']].copy()
    categories.set_index('id',inplace=True)
    # create a dataframe of the 36 individual category columns
    categories=categories['categories'].str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = list(categories.iloc[0])
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames =[ x[0:len(x)-2] for x in row]
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1:])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    #Merging the categories dataframe back to the original dataframe
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    categories.reset_index(inplace=True)
    df = df.merge(categories,how="inner",on=['id'])

    # dropping duplicate rows in the dataset
    df.drop_duplicates(subset=['id'], keep="first", inplace=True)
    
    #return cleaned dataframe
    return df
    

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)
    pass  


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