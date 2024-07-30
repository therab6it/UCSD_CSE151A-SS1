import pandas as pd
import numpy as np
import requests #to get the info from the congress website
from bs4 import BeautifulSoup #to convert the html to text
from transformers import DistilBertTokenizer, TFDistilBertModel #Distilbert embeddings
from transformers import BertTokenizer, TFBertModel #Bert large embeddings
import tensorflow as tf
import ast

api_key = ['0ndxcfLK5ncYlIqoibGzK8QmjgFpjK0zwo4dSFFA', 'mMecbd2llzpRG17qGE2QwbSZwPGoCaM06aEqeITx', 'h310atobblXmPgMqe48lYXYh14cDWc7wSN08VIHE', 'FeRN4xyXbhbkvKvbLbqJhm6uBoyo7a1QJuSwIibA', 'ufICLByVrzRqz1y9SrFLI6IAGhF9KEf1v7cckfvB'] #a bunch of api keys for the congress website(each key is limited to 5000 requests)

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0' #checks if a gpu is present and uses it if there is one

#tokenizer+model for DistilBert
model_name = 'distilbert/distilbert-base-uncased'
tokenizer_d = DistilBertTokenizer.from_pretrained(model_name)
model_d = TFDistilBertModel.from_pretrained(model_name)

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
def get_chunk_embeddings(chunk, large=False):
    if not large:
        inputs = tokenizer_d(chunk, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
        with tf.device(device):
            outputs = model_d(inputs)
        last_hidden_state = outputs.last_hidden_state
    # Mean pooling
    mean_embedding = tf.reduce_mean(last_hidden_state, axis=1)
    return mean_embedding.numpy()

# Function to get the aggregated embedding for the entire text. Works by taking the mean of each 512 word chunks
def get_text_embedding(text, large=False):
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

#Function to extract the cosponsor IDs
def extract_cosponsor_ids(cosponsors_str):
    if type(cosponsors_str) == str:
        cosponsors_list = ast.literal_eval(cosponsors_str)
    else: #if else to differentiate between JSON vs CSV
        cosponsors_list = cosponsors_str
    return [cosponsor[0] for cosponsor in cosponsors_list]

#Function to extract the committees
def extract_committees(committee_str):
    if type(committee_str) == str:
        committee_list = ast.literal_eval(committee_str)
    else:#if else to differentiate between JSON vs CSV
        committee_list = committee_str
    return [(committee[0] + ' ' + committee[1]) for committee in committee_list] #combines chamber + committee

#function to extract the sujbects
def extract_subjects(subjects_str):
    if type(subjects_str) == str:
        subjects_list = ast.literal_eval(subjects_str)
    else:#if else to differentiate between JSON vs CSV
        subjects_list = subjects_str
    r = [subject for subject in subjects_list]
    #last subject is always just 'congress' which is unnecessary
    return r[0:len(r)-1]


#function to extract the unique elements in a column and convert each row into a one hot key encoding
def encode_categorical(data_df, modify_df, column, mapping = []):
    if len(mapping) <= 0: #check if the user provided a mapping
        unique_categories =  np.unique(sum(data_df[column], []))
    else:
        unique_categories = mapping
    
    #subfunction to use dataframe parallelization using the apply() function
    def create_mapping_array(array):
        return_array = np.empty(len(unique_categories))
        iteration = 0
        for category in unique_categories:
            if category in array:
                return_array[iteration] = 1
            else:
                return_array[iteration] = 0
            iteration += 1
        return return_array
    return modify_df[column].apply(create_mapping_array), unique_categories
                

def get_bill_info(congress, bill_type, bill_number):
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
    text_bert = get_text_embedding(text)
    titles_bert = get_text_embedding(titles)
    summaries_bert = get_text_embedding(summaries)
    return committees, cosponsors, subjects, text_bert, titles_bert, summaries_bert

def get_congressman_specific_encoding(congressman_df, committees, cosponsors, subjects, text_bert, titles_bert, summaries_bert, bill_df=pd.read_json('bills_categorical.json')):
    congressman_df.rename(columns={
            'Congress': 'congress',
            'Bill Type': 'bill_type',
            'Bill Number': 'bill_number',
            'Vote Position': 'vote'
        }, inplace=True)
    congressman_df = congressman_df[congressman_df['vote'].isin(['Yea', 'Nay'])]
    merged = pd.merge(congressman_df, bill_df, on=['congress', 'bill_type', 'bill_number'])
    data_df = merged.drop(columns=['Category:', 'congress', 'bill_type', 'bill_number'])
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
    
    bill_as_df_unprocessed['committees'] = bill_as_df_unprocessed['committees'].apply(extract_committees)
    bill_as_df_unprocessed['cosponsors'] = bill_as_df_unprocessed['cosponsors'].apply(extract_cosponsor_ids)
    bill_as_df_unprocessed['subjects'] = bill_as_df_unprocessed['subjects'].apply(extract_subjects)
    
    data_df['cosponsors'] = data_df['cosponsors'].apply(extract_cosponsor_ids)
    data_df['committees'] = data_df['committees'].apply(extract_committees)
    data_df['subjects'] = data_df['subjects'].apply(extract_subjects)
    
    
    bill_as_df_unprocessed['cosponsors'], cosponsor_categories = encode_categorical(data_df, bill_as_df_unprocessed, 'cosponsors')
    bill_as_df_unprocessed['committees'], committee_categories = encode_categorical(data_df, bill_as_df_unprocessed, 'committees')
    bill_as_df_unprocessed['subjects'], subject_categories = encode_categorical(data_df, bill_as_df_unprocessed, 'subjects')
    
    
    concatenated = np.concatenate((bill_as_df_unprocessed.loc[0, 'committees'], bill_as_df_unprocessed.loc[0, 'cosponsors'], bill_as_df_unprocessed.loc[0, 'subjects'], bill_as_df_unprocessed.loc[0, 'text_bert'], bill_as_df_unprocessed.loc[0, 'titles_bert'], bill_as_df_unprocessed.loc[0, 'summaries_bert']))
    column = []
    column.append(concatenated)
    bill_as_df_unprocessed['combined'] = column
    bill_as_df = pd.DataFrame(bill_as_df_unprocessed['combined'].tolist(), columns=[f'x{i}' for i in range(bill_as_df_unprocessed['combined'].iloc[0].size)])
    return bill_as_df




def predict(location, df):
    device = '/CPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'
    with tf.device(device):
        model = tf.keras.models.load_model(location)
        r = model.predict(df)[0][0]
        return r