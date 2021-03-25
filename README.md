# Disaster Response Pipeline Project
Based on prelabbed tweets and text messages in csv file, first build ETL pipe line. Destination will be DB. Use this DB as a source for ML pipeline and build superwise learning model. Post model is created and trained use resulting trained model with in flask webbase app. Such app will be useful following diasterand we need to classify new messages using our model in various categories. This approch will be much better than simple keyword search.


### List of scripts
1. data/process_data.py

     - Loads the messages and categories datasets
     - Merges the two datasets
     - Cleans the data
     - Stores it in a SQLite database
     
2. models/train_classifier.py

     - Loads data from the SQLite database
     - Splits the dataset into training and test sets
     - Builds a text processing and machine learning pipeline
     - Trains and tunes a model using GridSearchCV
     - Outputs results on the test set
     - Exports the final model as a pickle file

3. app/run.py

	 - flask web

4. app/go.html / master.html

	- html, css and javascript for flask app
	

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
