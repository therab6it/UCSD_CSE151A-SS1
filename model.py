import pandas as pd
import numpy as np
import tensorflow as tf
import preprocess
import billinfo
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision # type: ignore
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression



"""
method to load the data
congressman: location of the congressman csv
large: whether to use the bert large(True) or distilbert(False) embeddings

"""
def load_data(congressman, large, degree, random_state=42, test_size = 0.2):
    voting_record = pd.read_csv(congressman)
    if not large:
        bill_data = pd.read_csv('distilbert_final_csv.csv')
        finalized_data, cosponsor_categories, committee_categories, subject_categories = preprocess.get_finished_df(voting_record, bill_data)
    else:
        bill_data = pd.read_json('bert_large_final_json.json')
        finalized_data, cosponsor_categories, committee_categories, subject_categories = preprocess.get_finished_df_large(voting_record, bill_data)
    
    #turn the combined column(which is an array in each row) into a bunch of features
    X = pd.DataFrame(finalized_data['combined'].tolist(), columns=[f'x{i}' for i in range(finalized_data['combined'].iloc[0].size)])
    preprocess.polynomial_features(X, degree=degree)
    y = finalized_data['vote']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return finalized_data, cosponsor_categories, committee_categories, subject_categories, X, y, X_train, X_test, y_train, y_test


#Method to create the neural netwoek, Recall yields the best results
def create_neural_network(X_train, X_test, y_train, y_test, epochs = 20, metric = ['Recall']):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8192, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(4096, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2048, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1024, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metric)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=['Yea', 'Nay'])
    print(report)
    model.save('Trained/Scott_Peters_CA_Democrat_rep_P000608_2.keras') 
    return model

#method the create a logistic regression model
def create_logistic_regression(X_train, X_test, y_train, y_test, max_iter):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    yhat_test = model.predict(X_test)
    yhat_train = model.predict(X_train)
    print(classification_report(y_test, yhat_test, target_names=['Yay', 'Nay']))
    return model
    

#method to train the neural network model
def train_neural_network(congressman,large, epochs = 15, metric = ['Recall'], degree=1):
    finalized_data, cosponsor_categories, committee_categories, subject_categories, X, y, X_train, X_test, y_train, y_test = load_data(congressman, large, degree=degree)
    model = create_neural_network(X_train, X_test, y_train, y_test, epochs=epochs, metric=metric)
    return model

#method to train the logistic_regression model
def train_logistic_regression(congressman, large, max_iter = 50, degree=1):
    finalized_data, cosponsor_categories, committee_categories, subject_categories, X, y, X_train, X_test, y_train, y_test = load_data(congressman, large, degree=degree)
    model = create_logistic_regression(X_train, X_test, y_train, y_test, max_iter)
    return model

#method to load a model from memory
def load_model(location):
    return tf.keras.models.load_model(location)


#function to get a bils info and provide a prediction
def predict(congress, bill_type, bill_number, model, congressman, large):

    bill = billinfo.get_bill_as_df(congress, bill_type, bill_number, congressman, large)

    predicted = model.predict(bill)
    #logistic regression returns a 1d array, neural network returns a 2d array, so this try except needs to be done to get the info
    try:
        if predicted[0,0] < 0.5:
            print(F"Prediction: Yea \n Probability of Yea: {(1-predicted[0,0])*100}% \n Probability of Nay: {predicted[0,0]*100}%")
        else:
            print(F"Prediction: Nay \n Probability of Yea: {(1-predicted[0,0])*100}% \n Probability of Nay: {predicted[0,0]*100}%")
    except:
        if predicted[0] < 0.5:
            print("Prediction: Yea")
        else:
            print("Prediction: Nay")


congressman = 'congressman_votes_data/Scott_Peters_CA_Democrat_rep_P000608_nan.csv' #change this to train rhe model on a different congressman
large = False #change this to change whether to use bert large(True) or distilbert(False)

#model = train_neural_network(congressman, metric= ['Recall'])
#print("Logistic Regression:")
#logistic_model = train_logistic_regression(congressman, large=large, max_iter=100000, degree=3)
print("Neural Network: ")
neural_model = train_neural_network(congressman, large=large, epochs=10, metric=['Recall'])
#predict(118, 's', 1667, logistic_model, congressman, large=large)
#predict(118, 'hr', 3442, logistic_model, congressman, large=large)
predict(118, 's', 1667, neural_model, congressman, large=large)
predict(118, 'hr', 3442, neural_model, congressman, large=large)


# Columns = vote, committees, cosponsors, subjects, text_bert, titles_bert, summaries_bert, combined


#print(finalized_data)
#print(type(finalized_data['vote'].values))
#print(finalized_data.columns)


#cogressman_votes_data/Scott_Peters_CA_Democrat_rep_P000608_nan.csv