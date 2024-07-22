# Bill Prediction Project

## AI Disclaimer
Partially written with the help of AI. Prompt: I am going to show you the code for 3 python files that I made for a project. All these files are brand new. I need you to create a README for the project, particularly the updates/functions of each file. It needs to follow this guideline:

Update your README.md to include your new work and updates you have all added. Make sure to upload all code and notebooks. Provide links in your README.md
followed by uploading the 3 python files


## Project Overview

This project aims to predict the voting behavior of congress members on bills using natural language processing (NLP) techniques and machine learning models. The project utilizes BERT and DistilBERT embeddings to process bill text, titles, summaries, committees, cosponsors, and subjects, and employs neural networks and logistic regression models for prediction.

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

## Usage

### Preprocessing Data

Use the functions in `preprocess.py` to preprocess the bill data and voting records. The preprocessing steps include extracting relevant information, encoding categorical data, and combining features.

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

[BERT Large Embeddings](https://drive.google.com/file/d/1SIXCe2fGVnLYC062aPLHVIHMLs7zYksE/view?usp=sharing)
[DistilBERT Embeddings](https://drive.google.com/file/d/1Mpab1Mc6JTlcQokTGGY3-1169RgY_okD/view?usp=sharing)
[Congressman Data](https://drive.google.com/drive/folders/1trQ2IgKjsJwroj9lQ55R9QbZYUVTQ1rg?usp=sharing)



## To Do
1. Hyperparameter Tuning
2. GPU Acceleration for model training
3. More robust saving/loading system
4. Training all congressmen
