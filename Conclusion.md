## **1. Conclusion of the First Model**

The model is trained for 10 epochs on a designated congressman, and the train and test set are split as 8:2. Below are the overall performances:

1. Accuracy:

   Overall accuracy is 0.91, indicating that the model correctly predicts the voting decision ("Yea" or "Nay") 91% of the time.

2. Precision, Recall, and F1-Score:

   - For the "Yea" class:

     Precision: 0.96

     Recall: 0.94

     F1-Score: 0.95

   - For the "Nay" class:

     Precision: 0.74

     Recall: 0.80

     F1-Score: 0.77

   - The macro average Precision is 0.85, Recall is 0.87, and F1-score is 0.86
   - The weighted average Precision is 0.92, Recall is 0.91, and F1-score is 0.91.

3. Loss:

   Training loss: 0.0468

   Validation loss: 0.0991

In conclusion, the model reaches high a high Precision, Recall and F1-Score. It still has some drawbacks in predicting the ‘Nay’ class since the performance is not as good as predicting the ‘Yea’ class. We think the reason for this is that the training data for ‘Nay’ class is much less than the data for ‘Yea’ class, which cause the insufficient learning of the ‘Nay’ class. More data would be helpful to increase the performance on ‘Nay’ class.

## **2. Possible Improvements**

Below are the possible improvements that could enhance the performance of the model.

1. Address Class Imbalance:

   The "Nay" class has lower precision, recall, and F1-score compared to the "Yea" class. This indicates a potential class imbalance issue. There are mainly 2 ways to solve this problem.

   - Resampling Methods: Use techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes.

   - Class Weights: Assign higher weights to the minority class ("Nay") when compiling the model.

2. Hyper parameter Tuning:

   We can use techniques like Grid Search or Random Search to find optimal hyper parameters, such as learning rate, batch size, and number of epochs.

3. Model Complexity:

   The model has a large number of neurons in each layer. We can consider reducing the number of neurons to simplify the model and prevent overfitting. A possible solution is adding L2 regularization to the Dense layers to penalize large weights and reduce overfitting.

4. Early Stopping:

   We can implement early stopping to prevent overfitting by monitoring the validation loss and stopping training when it stops improving.

5. Feature Engineering:

   We can also perform additional feature engineering to create more meaningful input features for the model by considering domain-specific knowledge to derive new features.