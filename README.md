# Emergency-Message-Classification-Pipelines

## Questions to be Answered
With this model the following question will be answered:
In which categories can a received emergency message be classified?

Aim:
An effective processing of emergency messages with different aid organizations involved.

## Files of this Repository
* Input Data
  ** disaster_messages.csv: emergency messages
  ** disaster_categories.csv: categories for emergency messages
* ETL Pipeline:
  ** process_data.py: python script with ETL pipeline for data cleaning
  ** ETL Pipeline Preparation_.ipynb: Jupiter Notebook in Python containing the preparation for process_data.py
  ** DisasterResponse.db: output database of process_data.py
* ML Pipeline:
  ** train_classifier.py: python script with machine learning pipeline to build a model with GridSearchCV
  ** ML Pipeline Preparation.ipynb: Jupiter Notebook in Python containing the preparation for train_classifier.py
  ** classifier.pkl: output model of train_classifier.py
* Web App:
  ** master.html: basic web app template
  ** go.html: web app template to process new emergency messages
  ** run.py: creates the vizualisations on the web app
* README.md

## Libraries
The following libraries are used:
* numpy
* pandas
* sqlalchemy
* re
* sklearn
* nltk
* pickle

## The Underlying Data
Real 26.248 emergengy messages classified in 36 categories provided by Figure Eight Inc. 

## The model
The ML pipeline includes some NLP techniques. A MultiOutputClassifier is used with a RandomForestClassifier optimized by GridSearchCV.

## Results
Jokes that are funny to everyone just do not exist. Nor jokes that everyone regards as terrible. The second worst rated joke has also been the most controversial one.
Whether a joke is funny or not remains very subjective. So the best you can do is respond to the preferences of your listeners.
A first clue is to tell a story in the joke with more than 600 characters.
