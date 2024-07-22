# Bill Prediction Project

## AI Disclaimer
Partially written with the help of AI. Prompt: I am going to show you the code for 3 python files that I made for a project. All these files are brand new. I need you to create a README for the project, particularly the updates/functions of each file. It needs to follow this guideline:

Update your README.md to include your new work and updates you have all added. Make sure to upload all code and notebooks. Provide links in your README.md
followed by uploading the 3 python files

## Project Abstract

Congressmen are notorious when it comes to deciding the laws of this country – a simple yea/nay vote by less than five hundred individuals can potentially decide the lives of millions of Americans. In this project, we create a machine learning model to predict how a congressman is likely to vote given a particular bill. A large database of bills and the corresponding votes by each congressman are gathered and processed. Various features of the bills, such as their content, sponsors, and historical voting patterns, are taken into account. A sophisticated neural network model is trained using supervised learning on the processed data. Given a new bill, our model predicts how each congressman is likely to vote, which can inform whether a bill is likely to pass. This prediction can be valuable for political analysts, lawmakers, and the general public in understanding legislative dynamics and potential outcomes.
Keywords: Congressional bills and voting, neural network, supervised learning, visualisation

## Project Overview

This project aims to predict the voting behavior of congress members on bills using natural language processing techniques and machine learning models. The project utilizes BERT and DistilBERT embeddings to process bill text, titles, and summaries, while using One Hot Encoding committees, cosponsors, and subjects, and employs neural networks and logistic regression models for prediction.

## Project Structure

The project consists of three main Python files:

1. **preprocess.py**
2. **billinfo.py**
3. **model.py**

## preprocess.py

This file contains various functions to preprocess and encode bill data for machine learning models. Key functions include:

- `extract_cosponsor_ids`: Extracts cosponsor IDs from a string or list.
- `extract_committees`: Extracts committee information from a string or list.
- `extract_subjects`: Extracts subjects from a string or list.
- `extract_number_arrays`: Converts a string of numbers into a NumPy array.
- `encode_votes`: Encodes votes as 1 (Nay) or 0 (Yea).
- `encode_categorical`: Encodes categorical data into one-hot key encoding.
- `combined_columns`: Combines multiple columns into a single mega list.
- `polynomial_features`: Generates polynomial features for the data.
- `get_finished_df`: Merges voting record and bill data, preprocesses, and returns the finalized dataframe.
- `get_finished_df_large`: Similar to `get_finished_df` but for JSON data.
- `process_new_bill`: Processes a new bill and returns a dataframe.

## billinfo.py

This file is responsible for retrieving bill information from the Congress API and generating embeddings using BERT and DistilBERT models. Key functions include:

- `split_text`: Splits text into chunks of 512 words with overlap.
- `get_chunk_embeddings`: Generates embeddings for text chunks.
- `get_text_embedding`: Aggregates embeddings for the entire text.
- `get_bill_titles`: Retrieves bill titles.
- `get_bill_summaries`: Retrieves bill summaries.
- `get_bill_committees`: Retrieves bill committees.
- `get_bill_cosponsors`: Retrieves bill cosponsors.
- `get_bill_subjects`: Retrieves bill subjects.
- `get_bill_text`: Retrieves bill text.
- `get_bill_as_df`: Combines all bill information into a dataframe.

## model.py

This file includes methods to train and evaluate machine learning models. Key functions include:

- `load_data`: Loads and preprocesses data for training.
- `create_neural_network`: Creates and trains a neural network model.
- `create_logistic_regression`: Creates and trains a logistic regression model.
- `train_neural_network`: Trains a neural network model with given parameters.
- `train_logistic_regression`: Trains a logistic regression model with given parameters.
- `load_model`: Loads a saved model from memory.
- `predict`: Predicts the outcome of a bill using a trained model.

## datascanning.py

This file includes the methods to go through every single JSON file we scraped from the House and Senate websites to create CSV files of every single congressman's vote position

## addbilldata.py

This file includes methods to retrieve bill information from the congress website through it's publically available API. Many of these methods have been included in `billinfo.py` and have better documentation in that file

## bert.py

This file includes methods to convert text into BERT and/or DistilBERT embeddings. Many of these methods have been included in `billinfo.py` and have better documentation in that file.

## Usage 

### Preprocessing Data

Use the functions in `preprocess.py` to preprocess the bill data and voting records. The preprocessing steps include extracting relevant information, encoding categorical data, and combining features. Use `get_finished_df` for for the DistilBERT CSV data, and `get_finished_df_large` for the BERT JSON data.

### Retrieving Bill Information

Use the functions in `billinfo.py` to retrieve detailed information about a bill from the Congress API and generate embeddings using BERT or DistilBERT models.

### Training Models

Use the functions in `model.py` to train and evaluate machine learning models. You can choose between a neural network and logistic regression model for prediction.

### Example

```python
congressman = 'congressman_votes_data/Scott_Peters_CA_Democrat_rep_P000608_nan.csv'
large = False #Set to True if you want to use BERT Large embeddings, False for DistilBert embeddings

# Train the neural network model
neural_model = train_neural_network(congressman, large=large, epochs=10, metric=['Recall'])

# Predict the outcome for a specific bill
predict(118, 's', 1667, neural_model, congressman, large=large)
predict(118, 'hr', 3442, neural_model, congressman, large=large)
```

# Data Downloads:

[BERT Large Embeddings](https://drive.google.com/file/d/1SIXCe2fGVnLYC062aPLHVIHMLs7zYksE/view?usp=sharing)\
[DistilBERT Embeddings](https://drive.google.com/file/d/1Mpab1Mc6JTlcQokTGGY3-1169RgY_okD/view?usp=sharing)\
[Congressman Data](https://drive.google.com/drive/folders/1trQ2IgKjsJwroj9lQ55R9QbZYUVTQ1rg?usp=sharing)\
[Official Congress Website Scraper For Scraping Vote Data](https://github.com/unitedstates/congress)


## Remarks
1. Nays were set to 1 to ensure we train on the recall of nays, as it seems to yield the best results
2. We excluded not voting and present to ensure we get a clean Yay/Nay results as the other 2 categories cannot be indicative of the congressman's position. We aim to predict a congressman's position if he/she were forced to vote, not necessarily whether they would vote or not to begin with.
3. Model file sizes are 0.5-1GB, so they take up a lot of space, hence we need to either find a way to store/retrieve the models, or find a way to reduce filesize
4. When we got the DistilBERT embeddings, we saved it as a CSV which unfortunately converted all our lists/arrays into strings and we had to convert it back when importing. After some testing, when saved as a JSON, it doesn't need this extra preprocessing step. Furthermore, BERT Large resulted in 1024 embeddings which got truncated with ... when exported to a CSV. JSON did not have this issue
5. Senators vote on significantly less legislation(due to the filibuster), thus resulting in very poor metrics for senators compared to representatives
6. One Hot Key encoding only assigns 1's and 0's to categories present on the specific bills the congressman voted on, not every single bill since it significantly reduces the number of features at the cost of a very minimal hit to accuracy when predicting bills not in the dataset.
7. The dataset only contains data from the 102nd congress(1991-1993) to the present since the vote data before that is not available on the website
8. We only included passage and passage under the suspension of rules and veto overrides. Other types of votes include starting/ending sessions, certifying elections, convictions, etc. The other categories do not include bill text and as such we decided not to include them. Regular passage involves getting a majority of votes in the House and Senate(although due to the filibuster, you really need 60 votes aka a 3/5 majority). Passage under the suspension of rules is a procedure in the house where debate can be skipped and the bill can directly be voted on in exchange for requiring a 2/3 majority. Veto overrides are when the president vetoes a bill, and is retured to congress, where if both houses pass the bill again with 2/3 majorities in both chambers, the bill becomes law, thus overriding the president's veto. 


## To Do
1. Hyperparameter Tuning
2. GPU Acceleration for model training
3. More robust saving/loading system
4. Training all congressmen
5. Visualization website frontend + backend server setup
