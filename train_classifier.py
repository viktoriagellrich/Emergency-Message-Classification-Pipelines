import sys

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
import pickle
import joblib

import nltk
nltk.download(["punkt", "wordnet", "stopwords"])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


def load_data(database_filepath):
    """ 
    Description: Load the data from a SQL database and split it into features X and aimed categories Y.
                 Store the column names of the categories in a variable.
                 
    Arguments: 
        database_filepath - the filepath to the database
        
    Returns: 
        X - a dataframe with the features
        Y - a dataframe with the aimed categories
        category_names - a list of the categories in Y
    """
    
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql("SELECT * FROM df", engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Description: Execute a NLP tansformation of a text.
    
    Arguments:
        text - string to be transformed
        
    Returns:
        clean tokens - the transformed text
    """
    
    #text = re.sub(r"[^a-zA-z0-9]", " ", text.lower().strip()) # normalize
    tokens = word_tokenize(text) # tokenize
    
    lemmatizer = WordNetLemmatizer()
    #stemmer = PorterStemmer()
    
    clean_tokens = []
    for tok in tokens:
        #if tok not in stopwords.words("english"):
            
            #clean_tok = stemmer.stem(lemmatizer.lemmatize(lemmatizer.lemmatize(tok, pos="v")))
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok, pos="v")).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Description: Build a pipeline that treats text data and applies a multioutput classification afterwards.
                 With GridSearchCV the pipeline is optimized by finetuning the paramters.
    
    Arguments:
        None
        
    Returns:
        cv - a GridSearchCV that uses the best parameters for the pipeline.
    """
    
    pipeline = Pipeline([
    ("vect", CountVectorizer(tokenizer=tokenize)),
    ("tfidf", TfidfTransformer()),
    ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1),(1, 2)),
        'vect__max_df': (0.75, 1),
        'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv
    
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description: Execute the model prediction.
                 Print the f1 score, precision and recall of the model.
    
    Arguments:
        model - the ML model to be evaluated
        X_test - the features of the test data
        Y_test - the aimed values of the test data
        category_names - a list of aimed categories
        
    Returns:
        None
    """

    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    
    for i, var in enumerate(category_names):
        print(var)
        print(classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i]))


def save_model(model, model_filepath):
    """
    Description: Save the executed model
    
    Arguments: 
        model - the model to be saved
        model_filepath  - the filepath to save the model at
        
    Returns:
        None
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))
    #joblib.dump(model, model_filepath)


def main():
    """
    Description: Load a database by calling the function load_data, 
                 build a model by calling the function build_model,
                 train the model,
                 evaluate the model by calling the function evaluate_model,
                 save the model.
                 Print an informative message in both cases: if the filepath is right or wrong.
                 
    Arguments:
        None
    
    Returns:
        None
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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