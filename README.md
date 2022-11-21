# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### The project consists of 3 main parts
1. Cleaning up the data set.
	Files used:
	a. data\disaster_categories.csv: Contains what each message is classified as.
	b. data\disaster_messages.csv.csv: Contains the actual messages.
	c. data\DisasterResponse.db: SQLite database where the cleased data is stored.
	d. data\process.py: contains the script for cleansing the raw data.
	
2. Training the classifier model.
	Files used:
	a. models\train_classifier.py: Contains the script to train a MultiOutputClassifier model.
	b. models\classifier.pkl: The model stored as a pickle file.
	
3. Frount End visualization of the classifier output.
	Files used:
	a. app\run.py: Used to spin up a flask server to display a frount end user interface to interact with the model.
	b. app\templates\master.html: The html page that serves up the homepage. Contains visualizations of the overall data trends in the data.
	c. app\templates\go.html: Webpage that shows the output of the model when an input message is entered in the homepage.
	
