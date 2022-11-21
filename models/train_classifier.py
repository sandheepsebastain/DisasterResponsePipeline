import sys
# import libraries
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier 
import nltk
import numpy as np
import pickle
import pdb

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def load_data(database_filepath):
    '''
    INPUT
    database_filepath - pandas dataframe

    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector
    category_names - gives all the names of the dependent variables

    This function cleans df using the following steps to produce X and y:
    1. reads the Messages table from the sqlite database
    2. Create X as the actual message text
    3. Create y as all the tags that message has been classified as
    4. Drop columns without any variability
    5. Get names of all y columns as category_names
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_query("select * from Messages", engine)
    #removing columns that dont have any variabilty so isnt really useful in prediction 
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df.drop(cols_to_drop, axis=1,inplace=True)
    X = df['message']
    y = df.iloc[:,4:]
    category_names=y.columns.tolist()
    return X,y,category_names

def tokenize(text):
    '''
    INPUT
    text - input string

    OUTPUT
    clean_tokens - A list containing tokens of lemmatized and lower cased words in the string

    This function does text transformations on the input string:
    1. reads the input stringMessages table from the sqlite database
    2. Tokenize the string
    3. Lemmatizes the string and then converts them into lower case
    4. Appends each of these to a list
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT
    None

    OUTPUT
    dictModels - Dictionary containing Different pipelines

    This function creates dictionary containing pipelines having different parameters:

    '''
    dictModels={}
    modelsparams = {\
         'LogisticRegression':
                  {   'model': [LogisticRegression(solver='liblinear')],
                      'parameters':{
                                    'clf__estimator__penalty': ['l1','l2'],
                                    'vect__ngram_range':[(1,1),(1,2),(1,3)],
                                    'tfidf__use_idf': (True, False)
                                }
                },
        'RandomForest':
                 {   'model': [RandomForestClassifier()],
                     'parameters':{
                                    'clf__estimator__n_estimators': [i for i in range(2,5,10)],
                                    'clf__estimator__max_depth': [10, 20],
                                    'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
                                    'vect__ngram_range':[(1,1),(1,2),(1,3)],
                                    'tfidf__use_idf': (True, False)
                               }
                }

        }
            
            
    for modeltype in modelsparams:          
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(modelsparams[modeltype]['model'][0]))
        ])
        

        parameters=modelsparams[modeltype]['parameters']
        
        cv = GridSearchCV(pipeline, param_grid=parameters)
        
        dictModels[modeltype]=cv
        
    return dictModels
        
def display_results(y_test, y_pred):
    '''
    INPUT
    y_test: List of dependent variables values from the testing dataset
    y_pred: List of predicted answers

    OUTPUT
    accuracy - Returns how close the predicted answers were to the test dataset actual values

    This function displays the classification report of the different models and return the accuracy of the models:
    1. reads the input stringMessages table from the sqlite database
    2. Tokenize the string
    3. Lemmatizes the string and then converts them into lower case
    4. Appends each of these to a list
    '''
    
    class_report = classification_report(y_test, y_pred)
    accuracy = (y_pred == y_test).mean()
    print("Classification Report:\n", class_report)
    print("Accuracy:", accuracy)
    return float(accuracy)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model: classifier model
    X_test: List of independent variables values from the testing dataset
    Y_test: List of dependent variables values from the testing dataset
    category_names - gives all the names of the dependent variables

    OUTPUT
    accuracy - Returns how close the predicted answers were to the test dataset actual values

    This function predicts the message category form the messages in the test dataset:
    1. predict the message categories from the messages in the test dataset for the model passed to the function 
    2. Get accuracy of the predicted category by calling the display_results function
    3. Append accuracy to list
    4. Get average accuracy of the model
    '''
    
    liAccuracy=[]
    
    Y_pred = model.predict(X_test)
    for index, item in enumerate(category_names):
        print("________________________________________________________")
        print("\nCategory: "+item)
        liAccuracy.append(display_results(Y_test.iloc[:,index],Y_pred[:,index]))
        
    print("==============================================================")
    accuracy=sum(liAccuracy)/len(liAccuracy)
    print("Average Model Accuracy: " +str(accuracy ))
    return accuracy


def save_model(model, model_filepath):
    '''
    INPUT
    model: classifier model
    model_filepath: path to store the model

    This function stores the model as a pickle file in the specified path:
    '''
    pickle.dump(model, open(model_filepath, "wb"))
    return True


def main():
    '''
    This function is the starting point for training the classifier:
    1. reads arguments provided via the command line. Arguments are path of the database and path to store the classifier model
    2. Reads data from the database
    3. splits the dataset into training and test datasets
    4. calls the build_model functions to get a dictionary of viable models for the classifier
    5. Trains each of the models using the training dataset
    6. Evaluates each model and selects the best model based on it's accuracy on the test dataset
    7. Saves the best model/classifer to a pickle file
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        dictModels = build_model()
        
        dictmodelEvalReport={}
        bestModel=""
        bestAccuracy=0
        for model in dictModels:
            temp_acc=0
            print('Training model:'+model)
            dictModels[model].fit(X_train, Y_train)
            
            print('Evaluating model:'+model)
            temp_acc=evaluate_model(dictModels[model], X_test, Y_test, category_names)
            dictmodelEvalReport[model]=temp_acc
            if temp_acc>bestAccuracy:
                bestAccuracy=temp_acc
                bestModel=model
        
        print("The best model is the "+bestModel+" with an average accuracy of "+str(bestAccuracy))
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(dictModels[bestModel], model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()