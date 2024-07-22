import pandas as pd
import numpy as np
import requests #to get the info from the congress website
from bs4 import BeautifulSoup #to convert the html to text
from transformers import DistilBertTokenizer, TFDistilBertModel #Distilbert embeddings
from transformers import BertTokenizer, TFBertModel #Bert large embeddings
import tensorflow as tf
import preprocess #custom preprocessing library/file



api_key = ['0ndxcfLK5ncYlIqoibGzK8QmjgFpjK0zwo4dSFFA', 'mMecbd2llzpRG17qGE2QwbSZwPGoCaM06aEqeITx', 'h310atobblXmPgMqe48lYXYh14cDWc7wSN08VIHE', 'FeRN4xyXbhbkvKvbLbqJhm6uBoyo7a1QJuSwIibA', 'ufICLByVrzRqz1y9SrFLI6IAGhF9KEf1v7cckfvB'] #a bunch of api keys for the congress website(each key is limited to 5000 requests)

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0' #checks if a gpu is present and uses it if there is one

#tokenizer+model for DistilBert
model_name = 'distilbert/distilbert-base-uncased'
tokenizer_d = DistilBertTokenizer.from_pretrained(model_name)
model_d = TFDistilBertModel.from_pretrained(model_name)

#tokenizer+model for Bert Large
model_name = 'bert-large-uncased'
tokenizer_l = BertTokenizer.from_pretrained(model_name)
model_l = TFBertModel.from_pretrained(model_name)


#Function to split the text into chunks of 512. Default overlap is 50

def split_text(text, overlap=50): 
    words = text.split()
    chunks = []
    for i in range(0, len(words), 512 - overlap):
        chunk = ' '.join(words[i:i + 512])
        chunks.append(chunk)
    return chunks

"""
Function to get the embeddings for each chunk
chunk: chunk to get the embeddings for
large: whether to use bert large(true) or distilbert(false)
"""
def get_chunk_embeddings(chunk, large):
    if not large:
        inputs = tokenizer_d(chunk, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
        with tf.device(device):
            outputs = model_d(inputs)
        last_hidden_state = outputs.last_hidden_state
    else:
        inputs = tokenizer_l(chunk, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
        with tf.device(device):
            outputs = model_l(inputs)
        last_hidden_state = outputs.last_hidden_state
    # Mean pooling
    mean_embedding = tf.reduce_mean(last_hidden_state, axis=1)
    return mean_embedding.numpy()

# Function to get the aggregated embedding for the entire text. Works by taking the mean of each 512 word chunks
def get_text_embedding(text, large):
    chunks = split_text(text)
    chunk_embeddings = [get_chunk_embeddings(chunk, large) for chunk in chunks]
    aggregated_embedding = np.mean(chunk_embeddings, axis=0)
    return aggregated_embedding

# Function to get titles of the bill(iteration is a remnant from the data mining phase, it is used to prevent oversaturation of one api key)
def get_bill_titles(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/titles?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        titles_data = response.json()
        titles = [title['title'] for title in titles_data.get('titles', [])]
        return titles
    return []

# Function to get summaries of the bill, only takes the first summary
def get_bill_summaries(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/summaries?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        summaries_data = response.json()
        if summaries_data.get('summaries'):
            return summaries_data['summaries'][0]['text']
    return ""

# Function to get committees of the bill
def get_bill_committees(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/committees?api_key={api_key[0]}'
    response = requests.get(url)
    if response.status_code == 200:
        committees_data = response.json()
        committees = []
        for committee in committees_data.get('committees', []):
            committees.append(tuple([
                committee.get('chamber'),
                committee.get('name')
            ]))
        return committees
    return []

# Function to get cosponsors of the bill
def get_bill_cosponsors(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/cosponsors?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}?api_key={api_key[iteration % len(api_key)]}'
    response2 = requests.get(url)
    cosponsors = []
    if response2.status_code == 200:
        bill_data = response2.json().get('bill', {})
        for sponsor in bill_data.get('sponsors', []):
            cosponsors.append(tuple([
                sponsor.get('bioguideId', ''),
                sponsor.get('fullName', ''),
                sponsor.get('party', ''),
                sponsor.get('state', ''),
                sponsor.get('firstName', ''),
                sponsor.get('lastName', '')
            ]))
    if response.status_code == 200:
        cosponsors_data = response.json()
        for cosponsor in cosponsors_data.get('cosponsors', []):
            cosponsors.append(tuple([
                cosponsor.get('bioguideId', ''),
                cosponsor.get('fullName', ''),
                cosponsor.get('party', ''),
                cosponsor.get('state', ''),
                cosponsor.get('firstName', ''),
                cosponsor.get('lastName', '')
            ]))
            return cosponsors
    return []

# Function to get subjects of the bill
def get_bill_subjects(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/subjects?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        subjects_data = response.json()
        subjects = [subject['name'] for subject in subjects_data.get('subjects', {}).get('legislativeSubjects', [])]
        policy_area = subjects_data.get('subjects', {}).get('policyArea', {}).get('name')
        if policy_area:
            subjects.append(tuple(policy_area))
        return subjects
    return []

# Function to get text of the bill
def get_bill_text(congress, bill_type, bill_number, iteration):
    url = f'https://api.congress.gov/v3/bill/{congress}/{bill_type}/{int(bill_number)}/text?api_key={api_key[iteration % len(api_key)]}'
    response = requests.get(url)
    if response.status_code == 200:
        text_data = response.json()
        if text_data.get('textVersions'):
            earliest_version = text_data['textVersions'][-1]  # Get the earliest version
            for format in earliest_version['formats']:
                if format['type'] == 'Formatted Text':
                    text_url = format['url']
                    text_response = requests.get(text_url)
                    if text_response.status_code == 200:
                        soup = BeautifulSoup(text_response.content, 'html.parser')
                        return soup.get_text()
    return ""


"""
#function to get bill as a df with all the feature columns to feed into the model
congress: an integer representing the congress number
bill_type: a bill type string(e.g 's', 'hr', 'hres', 'sres', etc)
bill_number: the number of the bill
congressman: location of the congressman csv
large: false means using distilbert, true means using bert large
"""
def get_bill_as_df(congress, bill_type, bill_number, congressman, large):
    #get all the billinfo
    text = get_bill_text(congress, bill_type, bill_number, 1)
    titles = get_bill_titles(congress, bill_type, bill_number, 1)
    subjects = get_bill_subjects(congress, bill_type, bill_number, 1)
    cosponsors = get_bill_cosponsors(congress, bill_type, bill_number, 1)
    committees = get_bill_committees(congress, bill_type, bill_number, 1)
    summaries = get_bill_summaries(congress, bill_type, bill_number, 1)
    
    #check if anything is not available
    if not type(summaries) == str or len(summaries) <=0:
        summaries = "A summary is in progress."
    if not type(titles) == str or len(titles) <=0:
        titles = "A title is in progress."
    if not type(text) == str or len(text) <=0:
        text = "A summary is in progress."
    
    #get the text embeddings
    text_bert = get_text_embedding(text, large)
    titles_bert = get_text_embedding(titles, large)
    summaries_bert = get_text_embedding(summaries, large)
    data = {
        'committees': committees,
        'cosponsors': cosponsors,
        'subjects': subjects,
        'text_bert': list(text_bert[0]),
        'titles_bert': list(titles_bert[0]),
        'summaries_bert':list(summaries_bert[0])
    }   
    record = []
    record.append(data)
    bill_as_df_unprocessed = pd.DataFrame(record)
    congressman_df = pd.read_csv(congressman)
    if large:
        bill_data_df = pd.read_json('bert_large_final_json.json')
    else:
        bill_data_df = pd.read_csv('distilbert_final_csv.csv')
    
    bill_as_df_semiprocessed = preprocess.process_new_bill(bill_as_df_unprocessed, congressman_df, bill_data_df) # use the preprocess file
    bill_as_df_processed = pd.DataFrame(bill_as_df_semiprocessed['combined'].tolist(), columns=[f'x{i}' for i in range(bill_as_df_semiprocessed['combined'].iloc[0].size)]) #extract the features as x0, x1, x2...
    return bill_as_df_processed
    
        
        
    
    
    