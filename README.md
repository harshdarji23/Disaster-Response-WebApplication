# Disaster-Response-WebApplication

## Project Overview
In this project I have built up a Web application which takes input messages during calamity and direct to a particular alleviation organization that can give speedy help
<br> The application utilizes a ML model to sort any new messages got, and the store additionally contains the code used to train the model and to set up any new datasets for model training purposes
<br>Data Source: [Figure Eight](https://www.figure-eight.com/data-for-everyone/)
<br>
## File Description
* Workspace: This is the main folder containing all the sub folders and files.
* Data: This folder contains all the .csv files, .db file and .py file
* Data-> disaster_categories.csv/disaster_messages.csv: These files inside the data folder contains messages, their genres and different categories they beong to.
* Data-> process_data.py: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data
* Data-> disaster.db: This file is the database which is used to fetch data whenever needed
* Models: This folder contains the ML pipeline and the pickle object
* Models-> train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
* Models->classifier.pkl: This file contains the fitted model so we do not need to fit the model again
* App: This folder contains run.py and templates which are used to run the main web application

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or localhost:3001
