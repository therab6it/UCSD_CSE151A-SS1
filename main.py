import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')

model.eval()

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    return inputs

def get_cls_embedding(text):
    if isinstance(text, str):
        text = [text]
    inputs = encode_text(text)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.numpy()

# congressst = set()
# Category: {'veto-override', 'passage-suspension', 'passage'}
# congress: {113, 114, 115, 116, 117, 118}
# bill_type: {'hr', 'hres', 'hconres', 'sconres', 's', 'hjres', 'sjres'}
# bill_number: 450 float variables, roughly range 0 - 5000.0
# Vote Position: {'Not Voting', 'Yea', 'Nay', 'Present'}
# titles: embedding, shape of (x, 768)
# summaries:
# committees:
# cosponsors:
# subjects:
# text:

merged_df = pd.read_csv('data/Scott_Peters_CA_Democrat_rep_P000608_merged.csv')
print(merged_df.columns)
# exit()
##### one-hot #####
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
categoryencoder = OneHotEncoder(sparse=False)
categoryencoder.fit([['veto-override'], ['passage-suspension'], ['passage']])

congressencoder = OneHotEncoder(sparse=False)
congressencoder.fit([[113], [114], [115], [116], [117], [118]])

billtypeencoder = OneHotEncoder(sparse=False)
billtypeencoder.fit([['hr'], ['hres'], ['hconres'], ['sconres'], ['s'], ['hjres'], ['sjres']])

resencoder = OneHotEncoder(sparse=False)
resencoder.fit([['Not Voting'], ['Yea'], ['Nay'], ['Present']])

scaler = MinMaxScaler()
mx_len = 768
mx_title_len = 20
# for index, row in merged_df.iterrows():
#     titles_list = row['titles'].split('|')
#     mx_title_len = len(titles_list) if len(titles_list) > mx_title_len else mx_title_len
# print("mx_title_len:",mx_title_len)
# exit()

from mymodel import SimpleNN
input_size = 768
hidden_size = 512
output_size = 4

mymodel = SimpleNN(input_size, hidden_size, output_size)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mymodel.parameters(), lr=0.001)

mymodel.eval()

loss_list = []

import random
random_subset = random.sample(range(merged_df.shape[0] + 1), int(0.2*(merged_df.shape[0]+1)))

import time
start = time.time()
correct = 0
for _ in range(100):
    sub_loss_list = []
    for index, row in merged_df.iterrows():
        if index in random_subset:
            continue
        category_encoded = categoryencoder.transform([[row['Category:']]])[0]
        # print("category_encoded:", category_encoded)
        category_encoded = np.pad(category_encoded,(0,mx_len - len(category_encoded)),'constant').reshape(1,mx_len)
        # print(type(category_encoded))

        congress_encoded = congressencoder.transform([[row['congress']]])[0]

        # print("congress_encoded:", category_encoded)
        congress_encoded = np.pad(congress_encoded, (0, mx_len - len(congress_encoded)), 'constant').reshape(1, mx_len)
        billtype_encoded = billtypeencoder.transform([[row['bill_type']]])[0]
        # print(row['bill_type'])
        # print("billtype_encoded:", billtype_encoded)
        billtype_encoded = np.pad(billtype_encoded, (0, mx_len - len(billtype_encoded)), 'constant').reshape(1, mx_len)

        titles_list = row['titles'].split('|')
        cls_embedding = get_cls_embedding(titles_list)
        # print(cls_embedding.shape)
        padded_titles = np.zeros((mx_title_len, mx_len))
        padded_titles[:cls_embedding.shape[0],:] = cls_embedding
        input_matrix = np.vstack((category_encoded, congress_encoded, billtype_encoded, padded_titles))
        # print(input_matrix.shape)
        res_encoded = resencoder.transform([[row['Vote Position']]])[0]
        # print("res_encoded:", res_encoded)
        # print("!!",res_encoded.shape)
        input_matrix = np.expand_dims(input_matrix, axis=0)
        input_matrix = np.expand_dims(input_matrix, axis=0)
        outputs = mymodel(torch.tensor(input_matrix.astype(np.float32)))
        # print("outputs:",outputs)
        # one_hot_predictions = F.one_hot(outputs, num_classes=4)
        # print("pred:",one_hot_predictions)
        if np.argmax(outputs.detach().numpy()) == np.argmax(res_encoded):
            correct += 1
        loss = criterion(outputs, torch.tensor([np.argmax(res_encoded)]))
        sub_loss_list.append(loss.item())
        # print("loss:", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break
        # a=input()
    loss_list.append(sub_loss_list)
print("total time:", time.time()-start)
print("loss:", loss_list)
print("acc for training data:",correct / int(8*(merged_df.shape[0]+1)))
mymodel.eval()
correct = 0
sub_loss_list = []
for index, row in merged_df.iterrows():
    if index in random_subset:
        if index in random_subset:
            category_encoded = categoryencoder.transform([[row['Category:']]])[0]
            # print("category_encoded:", category_encoded)
            category_encoded = np.pad(category_encoded, (0, mx_len - len(category_encoded)), 'constant').reshape(1, mx_len)
            # print(type(category_encoded))

            congress_encoded = congressencoder.transform([[row['congress']]])[0]

            # print("congress_encoded:", category_encoded)
            congress_encoded = np.pad(congress_encoded, (0, mx_len - len(congress_encoded)), 'constant').reshape(1, mx_len)
            billtype_encoded = billtypeencoder.transform([[row['bill_type']]])[0]
            # print(row['bill_type'])
            # print("billtype_encoded:", billtype_encoded)
            billtype_encoded = np.pad(billtype_encoded, (0, mx_len - len(billtype_encoded)), 'constant').reshape(1, mx_len)

            titles_list = row['titles'].split('|')
            cls_embedding = get_cls_embedding(titles_list)
            # print(cls_embedding.shape)
            padded_titles = np.zeros((mx_title_len, mx_len))
            padded_titles[:cls_embedding.shape[0], :] = cls_embedding
            input_matrix = np.vstack((category_encoded, congress_encoded, billtype_encoded, padded_titles))
            # print(input_matrix.shape)
            res_encoded = resencoder.transform([[row['Vote Position']]])[0]
            # print("res_encoded:", res_encoded)
            # print("!!",res_encoded.shape)
            input_matrix = np.expand_dims(input_matrix, axis=0)
            input_matrix = np.expand_dims(input_matrix, axis=0)
            outputs = mymodel(torch.tensor(input_matrix.astype(np.float32)))
            # print("outputs:",outputs)
            # one_hot_predictions = F.one_hot(outputs, num_classes=4)
            # print("pred:",one_hot_predictions)
            if np.argmax(outputs.detach().numpy()) == np.argmax(res_encoded):
                correct += 1
            loss = criterion(outputs, torch.tensor([np.argmax(res_encoded)]))
            sub_loss_list.append(loss.item())
print("loss:", sub_loss_list)
print("acc for testing data:", correct / int(0.2 * (merged_df.shape[0] + 1)))