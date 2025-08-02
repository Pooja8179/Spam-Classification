This project is a Spam Detection System built using Python and scikit-learn. It utilizes basic Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier to determine whether a given email message is Spam or Not Spam.

Table of Contents
Project Overview

Dataset

Tech Stack

How It Works

Model Performance

Running the Code

Example Usage

Future Improvements

Contributing

License

Author

Project Overview
The main goal of this project is to classify emails as Spam or Not Spam based on their content using Machine Learning. This type of text classification is a classic NLP problem and has many real-world applications such as:

Email spam filtering

SMS classification

Content moderation

Dataset
The dataset used is a CSV file named emails.csv, which contains:

Columns:

text: The content of the email

spam: Label (1 = spam, 0 = not spam)

Make sure your dataset follows this structure. The model uses the text column as input and learns to predict the spam column.

Tech Stack
Python 3

Pandas: Data manipulation and reading CSV

scikit-learn:

CountVectorizer – for text feature extraction

train_test_split – to split data

MultinomialNB – for classification

accuracy_score – to evaluate model performance

How It Works
Data Loading
The CSV file is read using pandas.

Text Vectorization
CountVectorizer converts the text into numerical vectors using a bag-of-words approach.

Model Training
The dataset is split into training and testing sets (80% training, 20% testing). A MultinomialNB classifier is trained on the training data.

Model Evaluation
The model is tested on unseen data and accuracy is calculated.

Prediction Function
A custom function allows users to enter new email messages and receive real-time spam predictions.

Model Performance
After training the model, we evaluate its accuracy on the test set:

Accuracy: 0.98 
