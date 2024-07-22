from transformers import BertTokenizer, TFBertModel, DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import pandas as pd
import numpy as np

print(tf.config.experimental.list_physical_devices())


device = '/GPU:0' if tf.config.list_physical_devices('GPU') else 'CPU:0'


model_name = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

def split_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_chunk_embeddings(chunk):
    inputs = tokenizer(chunk, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
    with tf.device(device):
        outputs = model(inputs)
    last_hidden_state = outputs.last_hidden_state
    mean_embedding = tf.reduce_mean(last_hidden_state, axis=1)
    return mean_embedding.numpy()

def get_text_embedding(text):
    chunks = split_text(text)
    chunk_embeddings = [get_chunk_embeddings(chunk) for chunk in chunks]
    aggregated_embedding = np.mean(chunk_embeddings, axis=0)
    return aggregated_embedding


csv_without = pd.read_csv('/Users/suraj/Desktop/Classes/2024 Summer/Summer Session 1/CSE 151A/Project/Data Collection/billinfocombined.csv')
csv_without.dropna(ignore_index = True)

csv_with = pd.DataFrame(columns=['congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'committees', 'cosponsors', 'subjects', 'text', 'text_bert', 'titles_bert', 'summaries_bert'])
iteration = 1

records = []



for index, row in csv_without.iterrows():
    if len(row) <= 0:
        continue
    
    bill_text = row['text']
    bill_title = row['titles']
    bill_summaries = row['summaries']
    
    if not type(bill_summaries) == str:
        bill_summaries = "A summary is in progress."
    if not type(bill_title) == str:
        bill_title = "A title is in progress."
    if not type(bill_text) == str:
        bill_text = "A text is in progress."
    
    text_embedding = get_text_embedding(bill_text)
    title_embedding = get_text_embedding(bill_title)
    summary_embedding = get_text_embedding(bill_summaries)
    
    if iteration <= len(records):
        continue
    
    #if iteration == 1:
    #csv_with = {'congress': row['congress'], 'bill_type': row['bill_type'], 'bill_number': row['bill_number'], 'titles': bill_title, 'summaries': bill_summaries, 'committees': row['committees'], 'cosponsors': row['cosponsors'], 'subjects': row['subjects'], 'text':row['text'], 'text_bert':text_embedding, 'titles_bert':title_embedding, 'summaries_bert':summary_embedding}
    #else:
    new_entry = {'congress': row['congress'], 'bill_type': row['bill_type'], 'bill_number': row['bill_number'], 'titles': bill_title, 'summaries': bill_summaries, 'committees': row['committees'], 'cosponsors': row['cosponsors'], 'subjects': row['subjects'], 'text':row['text'], 'text_bert': text_embedding[0], 'titles_bert':title_embedding[0], 'summaries_bert':summary_embedding[0]}
    records.append(new_entry)
    print(f"{iteration}. Finished bill {row['congress']} {row['bill_type']} {row['bill_number']}")
    if iteration % 500 == 0:
        csv_with = pd.DataFrame(records)
        csv_with.to_json(f"/Users/suraj/Desktop/Classes/2024 Summer/Summer Session 1/CSE 151A/Project/bert_large_{iteration/500}_json.json")
        print("Saved to file")
    iteration = iteration+1


csv_with.to_json('/Users/suraj/Desktop/Classes/2024 Summer/Summer Session 1/CSE 151A/Project/bert_large_final_json.json')
print("Finished!!!")
