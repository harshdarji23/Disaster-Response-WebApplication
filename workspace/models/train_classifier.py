import sys
# import libraries
import sys
import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sqlalchemy import create_engine

# For machine learning
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import warnings

warnings.simplefilter('ignore')

# For nlp
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
import pickle

def load_data(database_filepath):
    """
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('my_disaster_response_table',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """Normalize, tokenize and lemmatize text string
    
    Input:
    text: string- String containing message for processing
       
    Returns:
    stemmed: list of strings- List containing normalized and lemmatize word tokens
    """
    # Detect URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize and tokenize and remove punctuation
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Lemmatize
    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return tokens

def build_model():
    """Build model.
    Returns:
        pipline: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
    """
    
    #https://medium.com/swlh/the-hyperparameter-cheat-sheet-770f1fed32ff
    pipeline_ada = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'))
    ))
    ])

    parameters_ada = {
    'clf__estimator__learning_rate': [0.1, 0.3],
    'clf__estimator__n_estimators': [100, 200]
    }

    cv_ada = GridSearchCV(estimator=pipeline_ada, param_grid=parameters_ada, cv=3, scoring='f1_weighted', verbose=3)
    model=cv_ada
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model
    Input:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in Y_pred]), target_names = category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
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