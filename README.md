📩 Spam Message Classifier

A machine learning project to classify SMS messages as Spam or Ham using different models such as Random Forest Classifier (RFC), Decision Tree Classifier (DTC), and Multinomial Naïve Bayes (MNB).

🚀 Features

Preprocesses raw text messages (lowercasing, removing special characters, stopwords removal, stemming).

Converts text into numerical features using CountVectorizer.

Trains and evaluates 3 models:

🌲 Random Forest Classifier (RFC)

🌳 Decision Tree Classifier (DTC)

📊 Multinomial Naïve Bayes (MNB)

Provides accuracy, confusion matrix, and classification reports.

Saves trained models using Pickle for future predictions.

Includes a function predictMessage() to classify any new SMS.

📂 Dataset

The dataset used is spam.csv, which contains SMS labeled as ham (not spam) or spam.

Example:

ham, Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam, Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...


📦 Requirements

Python 3.x

numpy

pandas

matplotlib

seaborn

nltk

scikit-learn

(You can install them using pip install -r requirements.txt)

🔑 Usage
1. Training the Model

Run the training script (spam_classifier.py or notebook). This will:

Preprocess the dataset

Train all three models

Save them as .pkl files

2. Predicting a Message

Use the predictMessage() function to test a new message:

usermessage = "Congratulations! You have won a free lottery ticket."
result = predictMessage(usermessage)
print(f"The message is: {result}")


Output:

The message is: Spam

📊 Model Performance
Model	Accuracy
Random Forest Classifier	~97%
Decision Tree Classifier	~95%
Multinomial Naïve Bayes	~98%
📌 Project Structure
📂 spam-classifier
 ┣ 📜 spam_classifier.py        # Main training + prediction code
 ┣ 📜 spam.csv                  # Dataset
 ┣ 📜 RFC.pkl                   # Saved Random Forest Model
 ┣ 📜 DTC.pkl                   # Saved Decision Tree Model
 ┣ 📜 MNB.pkl                   # Saved Naïve Bayes Model
 ┣ 📜 vectorizer.pkl            # Saved CountVectorizer
 ┣ 📜 requirements.txt          # Dependencies
 ┗ 📜 README.md                 # Documentation

✨ Future Improvements

Add support for deep learning models (LSTMs, Transformers).

Deploy as a web app using Flask/Django.

Create a Streamlit app for easy testing.
