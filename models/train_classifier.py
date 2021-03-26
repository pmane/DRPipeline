# import libraries
import sys
import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """
    Load data from DB
    
    Arguments:
        database_filepath - Path to db file
    Output:
        X -> features datab base data frame
        Y -> labale data base data frame
        category_names -> List of categories name
    """
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    table_name = (database_filepath).replace(".db","") + "_table"
    df =  pd.read_sql_table(table_name, engine)
    
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y, y.keys()

def tokenize(text):
    '''
    Arguments 
        text: Text to be processed   
    Output
        Returns a processed text variable that was tokenized, stripped and lemmatized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model(X_train,y_train):
    '''
    Arguments 
        X_Train: Training features for GridSearchCV
        y_train: Training labels for GridSearchCV
    Output
        Returns a pipeline model 
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    return cv

def evaluate_model(pipeline, X_test, Y_test, category_names):
    '''
    Arguments 
        pipeline: The model to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    Output
        Print evaluation results
    '''
    # predict on test data
    y_pred = pipeline.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))

def save_model(model, model_filepath):
    '''
    Saves the model to disk
    Arguments 
        model: Model to be save
        model_filepath: location to save
    Output
        save model as a pickle file.
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
