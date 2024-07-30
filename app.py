from flask import Flask, request, redirect, url_for, render_template
from flask_socketio import SocketIO, emit
import numpy as np
import pandas as pd
import os
from bs4 import BeautifulSoup
import threading
import billinfo
from numba import cuda



def update_website(congress, bill_type, bill_number):
    committees, cosponsors, subjects, text_bert, titles_bert, summaries_bert = billinfo.get_bill_info(congress,bill_type.lower(),bill_number)
    with open('congressman.html', 'r', encoding='utf-8') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'lxml')
    h1_tag = soup.find('h1')
    if h1_tag:
        h1_tag.string = 'Prediction:'
    cards = soup.find_all('div', class_='card')
    iteration = 1
    percentage = round((iteration/len(cards))*100, 2)
    corrupt = pd.DataFrame(columns=['name'])
    bill_df=pd.read_json('bills_categorical.json')
    i = 1
    files = os.listdir('congressman_votes_data')
    for card in cards:
        bioguide_id = card.find('img')['src'][38:45]
        for file in files:
            if not file.endswith('.csv'):
                continue
            if bioguide_id in file:
                file_name = file
                break
        try:
            new_bill = billinfo.get_congressman_specific_encoding(pd.read_csv('congressman_votes_data/' + file_name), committees, cosponsors, subjects, text_bert, titles_bert, summaries_bert, bill_df=bill_df)
            model_location = 'DistilBERT/' + file_name.strip('.csv') + '.keras'
            prediction_percentage = billinfo.predict(model_location, new_bill)*100
        except:
            corrupt.loc[len(corrupt)] = file_name
            print(corrupt)
            print(len(corrupt))
            prediction_percentage = 0.5
        
        #try:
        
        #except:
        #    print(f'{i}. Error on {file_name}')
        #    i+=1
        #    prediction_percentage = 0.5
        

        
        details = card.find('div', class_='details')
        vote = details.find_all('p', class_='vote')[0]
        yea_prob = details.find_all('p', class_='vote')[1]
        nay_prob = details.find_all('p', class_='vote')[2]
        stats = details.find_all('p', class_='vote')[3]
        #yea_prob_test = random.uniform(0,1)
        vote.string = f"Vote: {'Yea' if prediction_percentage<= 0.5 else 'Nay'}"
        yea_prob.string = f"Yea Probability: {round(1-prediction_percentage,2)}%"
        nay_prob.string = f"Nay Probability: {round(prediction_percentage,2)}%"
        stats.string = "Accuracy: None"
        socketio.emit('update_progress', {'progress': percentage})
        iteration += 1
        percentage = round((iteration/len(cards))*100, 2)

    corrupt.to_csv('corruptv2.csv')

    with open('templates/congressman.html', 'w', encoding='utf-8') as file:
        file.write(str(soup))

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)


@app.after_request
def add_ngrok_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response


# Route to serve index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and redirect to loading.html
@app.route('/predict', methods=['POST'])
def predict():
    congress = request.form['congress']
    billType = request.form['billType']
    billNumber = request.form['billNumber']
    
    # Start the background task to update the progress bar
    threading.Thread(target=background_task, args=(congress, billType, billNumber)).start()
    
    # Redirect to loading.html with parameters in query string
    return redirect(url_for('loading', congress=congress, billType=billType, billNumber=billNumber))

# Route to serve loading.html
@app.route('/loading')
def loading():
    return render_template('loading.html')

# Route to serve congressman.html
@app.route('/congressman')
def congressman():
    return render_template('congressman.html')

def background_task(congress, billType, billNumber):
    #socketio.emit('update_progress', {'progress': 100})
    update_website(congress, billType, billNumber)
    socketio.emit('redirect', {'url': 'congressman'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
