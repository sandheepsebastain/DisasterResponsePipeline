import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
import pdb


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    liColumns=df.columns.tolist()
    liColumns.remove('id')
    liColumns.remove('message')
    liColumns.remove('original')
    type_counts = df.groupby('genre')[liColumns].sum()

    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                  x= type_counts.columns.tolist(),
                  y= type_counts[type_counts.index=='direct'].values[0],
                  name= 'direct'
                ),
                Bar(
                  x= type_counts.columns.tolist(),
                  y= type_counts[type_counts.index=='news'].values[0],
                  name= 'news'
                ),
                Bar(
                  x= type_counts.columns.tolist(),
                  y= type_counts[type_counts.index=='social'].values[0],
                  name= 'social'
                ),
            ],

            'layout': {
                'barmode':'stack',
                'title': 'Distribution of Message Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Classification"
                }
            }
            
            
        },
        {
            'data': [
                Bar(
                  x= type_counts.columns.tolist(),
                  y= type_counts[type_counts.index=='direct'].values[0],
                  name= 'direct'
                ),
                Bar(
                  x= type_counts.columns.tolist(),
                  y= type_counts[type_counts.index=='news'].values[0],
                  name= 'news'
                ),
                Bar(
                  x= type_counts.columns.tolist(),
                  y= type_counts[type_counts.index=='social'].values[0],
                  name= 'social'
                ),
            ],

            'layout': {
                'barmode':'stack',
                'barnorm': 'percent',
                'title': 'Share of Message genres within Classifications',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Message Classification"
                }
            }
            
            
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()