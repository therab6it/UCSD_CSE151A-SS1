import pandas as pd
import numpy as np
import ast #used to convert strings to lists

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

#function to extract arrays from a string
def extract_number_arrays(number_str):
    number_list = number_str.strip('[]').split()
    number_array = np.array(number_list, dtype=float)
    return number_array

#function to encode the votes as a 1(Nay) or 0(Yea)
def encode_votes(vote):
    #Nay is a 1 since tensorflow trains on 1's(positives), so treating nay's as positives improves results
    if vote == "Yea":
        return 0
    else:
        return 1

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
                

#function to combine all the number columns into 1 mega list
def combined_columns(data_df):
    column = []
    for i in range(len(data_df)):
        concatenated = np.concatenate((data_df.loc[i, 'committees'], data_df.loc[i, 'cosponsors'], data_df.loc[i, 'subjects'], data_df.loc[i, 'text_bert'], data_df.loc[i, 'titles_bert'], data_df.loc[i, 'summaries_bert']))
        column.append(concatenated)
    data_df['combined'] = column
    return column

#function to get the features raised to exponents for polynomial models
def polynomial_features(data_df, degree = 1):
    def num_exponential(num, exp):
        return num ** exp
    if degree <= 1:
        return
    cols = data_df.columns
    for i in range(degree+1):
        if i == 0 or i == 1:
            continue
        for j in range(len(cols)):
            col_new = f'x{len(cols)+j}'
            col_old = f'x{j}'
            data_df[col_new] = data_df[col_old].apply(lambda x: num_exponential(x, i))
        
    
#function to get the finished df from a csv
def get_finished_df(voting_record, new_data = None, bill_data = pd.read_csv('distilbert_final_csv.csv'), new_bill = False): #was originally intended to be able to preprocess the data as well as a new bill, but was changed, so the new_bill parameter is deprecated
    if not new_bill:
        #rename the congressman columns to match the bill columns
        voting_record.rename(columns={
            'Congress': 'congress',
            'Bill Type': 'bill_type',
            'Bill Number': 'bill_number',
            'Vote Position': 'vote'
        }, inplace=True)
        voting_record = voting_record[voting_record['vote'].isin(['Yea', 'Nay'])] #only care about the yea's and nays
        merged = pd.merge(voting_record, bill_data, on=['congress', 'bill_type', 'bill_number']) #merge the dataframes
        data_df = merged.drop(columns=['Category:', 'congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'text']) #drop the non-numerical features
        data_df['vote'] = data_df['vote'].apply(encode_votes)
        data_df['cosponsors'] = data_df['cosponsors'].apply(extract_cosponsor_ids)
        data_df['cosponsors'], cosponsor_categories = encode_categorical(data_df, data_df, 'cosponsors')
    
        data_df['committees'] = data_df['committees'].apply(extract_committees)
        data_df['committees'], committee_categories = encode_categorical(data_df, data_df, 'committees')
    
    
        data_df['subjects'] = data_df['subjects'].apply(extract_subjects)    
        data_df['subjects'], subject_categories = encode_categorical(data_df, data_df, 'subjects')
    else:
        voting_record.rename(columns={
            'Congress': 'congress',
            'Bill Type': 'bill_type',
            'Bill Number': 'bill_number',
            'Vote Position': 'vote'
        }, inplace=True)
        voting_record = voting_record[voting_record['vote'].isin(['Yea', 'Nay'])]
        merged = pd.merge(voting_record, bill_data, on=['congress', 'bill_type', 'bill_number'])
        merged = merged.drop(columns=['Category:', 'congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'text'])
        
        
        data_df = new_data
        data_df['cosponsors'] = data_df['cosponsors'].apply(extract_cosponsor_ids)
        merged['cosponsors'] = merged['cosponsors'].apply(extract_cosponsor_ids)
        data_df['cosponsors'], cosponsor_categories = encode_categorical(merged, data_df, 'cosponsors')
    
        data_df['committees'] = data_df['committees'].apply(extract_committees)
        merged['committees'] = merged['committees'].apply(extract_committees)
        data_df['committees'], committee_categories = encode_categorical(merged, data_df, 'committees')
    
    
        data_df['subjects'] = data_df['subjects'].apply(extract_subjects)    
        merged['subjects'] = merged['subjects'].apply(extract_subjects)    
        data_df['subjects'], subject_categories = encode_categorical(merged, data_df, 'subjects')
    
    data_df['text_bert'] = data_df['text_bert'].apply(extract_number_arrays)
    data_df['titles_bert'] = data_df['titles_bert'].apply(extract_number_arrays)
    data_df['summaries_bert'] = data_df['summaries_bert'].apply(extract_number_arrays)
    combined_columns(data_df)

    return data_df, cosponsor_categories, committee_categories, subject_categories



#function nearly identical to the csv function, but JSONs don't save the arrays as strings so don't need that extra layer of processing
def get_finished_df_large(voting_record, bill_data = pd.read_json('bert_large_final_json.json')):
    voting_record.rename(columns={
        'Congress': 'congress',
        'Bill Type': 'bill_type',
        'Bill Number': 'bill_number',
        'Vote Position': 'vote'
    }, inplace=True)
    voting_record = voting_record[voting_record['vote'].isin(['Yea', 'Nay'])]
    merged = pd.merge(voting_record, bill_data, on=['congress', 'bill_type', 'bill_number'])
    data_df = merged.drop(columns=['Category:', 'congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'text'])
    data_df['vote'] = data_df['vote'].apply(encode_votes)
    data_df['cosponsors'] = data_df['cosponsors'].apply(extract_cosponsor_ids)
    data_df['cosponsors'], cosponsor_categories = encode_categorical(data_df, data_df, 'cosponsors')
    
    data_df['committees'] = data_df['committees'].apply(extract_committees)
    data_df['committees'], committee_categories = encode_categorical(data_df, data_df, 'committees')
    
    
    data_df['subjects'] = data_df['subjects'].apply(extract_subjects)    
    data_df['subjects'], subject_categories = encode_categorical(data_df, data_df, 'subjects')
    
    combined_columns(data_df)
    
    return data_df, cosponsor_categories, committee_categories, subject_categories


#function to process a new bill. Extremely similar to the JSON preprocessing function
def process_new_bill(bill_df, voting_record, bill_data_df):
    voting_record.rename(columns={
            'Congress': 'congress',
            'Bill Type': 'bill_type',
            'Bill Number': 'bill_number',
            'Vote Position': 'vote'
        }, inplace=True)
    voting_record = voting_record[voting_record['vote'].isin(['Yea', 'Nay'])]
    merged = pd.merge(voting_record, bill_data_df, on=['congress', 'bill_type', 'bill_number'])
    data_df = merged.drop(columns=['Category:', 'congress', 'bill_type', 'bill_number', 'titles', 'summaries', 'text'])
    
    bill_df['cosponsors'] = bill_df['cosponsors'].apply(extract_cosponsor_ids)
    data_df['cosponsors'] = data_df['cosponsors'].apply(extract_cosponsor_ids)
    bill_df['cosponsors'], cosponsor_categories = encode_categorical(data_df, bill_df, 'cosponsors')
    
    
    bill_df['committees'] = bill_df['committees'].apply(extract_committees)
    data_df['committees'] = data_df['committees'].apply(extract_committees)
    bill_df['committees'], committee_categories = encode_categorical(data_df, bill_df, 'committees')
    
    
    bill_df['subjects'] = bill_df['subjects'].apply(extract_subjects)  
    data_df['subjects'] = data_df['subjects'].apply(extract_subjects)   
    bill_df['subjects'], subject_categories = encode_categorical(data_df, bill_df, 'subjects')
    
    combined_columns(bill_df)
    return bill_df
    