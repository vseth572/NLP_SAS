import spacy
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model, model_selection, preprocessing, naive_bayes, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

PATH_TRAIN = "C:/Users/vs_00/Desktop/Internship/SAS/NLP_Disaster_Tweets/train.csv"
PATH_TEST = "C:/Users/vs_00/Desktop/Internship/SAS/NLP_Disaster_Tweets/test.csv"
PATH_SUBMISSION = "C:/Users/vs_00/Desktop/Internship/SAS/NLP_Disaster_Tweets/sample_submission.csv"
train_df = pd.read_csv(PATH_TRAIN, usecols=['text', 'target'])
test_df = pd.read_csv(PATH_TEST)
sample_submission = pd.read_csv(PATH_SUBMISSION)
nlp = spacy.load("en_core_web_sm")
stops = stopwords.words("english")

""" This function does the follwoing 
1. HTML decoding,
2. Removes @mentions, URL Links, all the numbers and replaces hashtags with empty space
3  Removes stopwords
4. Lemmatizes: convert the word to its root form
"""


def normalize(comment, lowercase, remove_stopwords):
    if lowercase:
        comment = comment.lower()
    soup = BeautifulSoup(comment, 'lxml')
    comment = soup.get_text()
    comment = re.sub(combined_pat, '', str(comment))
    try:
        clean = comment.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = comment
    letter_only = re.sub("[^a-zA-Z]", ' ', clean)
    letter_only = nlp(letter_only)

    lemmatized = list()
    for word in letter_only:
        lemma = word.lemma_.strip()
        if lemma:
            if not remove_stopwords or (remove_stopwords and lemma not in stops):
                lemmatized.append(lemma)
    return " ".join(lemmatized)


train_x, test_x, train_y, test_y = train_test_split(train_df["text"], train_df["target"], test_size=0.3)

train_df['text'] = train_df['text'].apply(normalize, lowercase=True, remove_stopwords=True)
test_df['text'] = test_df['text'].apply(normalize, lowercase=True, remove_stopwords=True)

# # Making WorCloud
# wordcloud = WordCloud(background_color="white", width=3000, height=2000, max_words=300).generate(' '.join(train_df['text'].transpose()))
# # set the figsize
# plt.figure(figsize=[15,10])
# # plot the wordcloud
# plt.imshow(wordcloud, interpolation="bilinear")
# # remove plot axes
# plt.axis("off")
# plt.show()

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(train_df['text'])
train_x_tfidf = tfidf_vectorizer.transform(train_x)
test_x_tfidf = tfidf_vectorizer.transform(test_x)

# Ridge Classifier
clf = linear_model.RidgeClassifier()
clf.fit(train_x_tfidf, train_y)
predictions_ridge = clf.predict(test_x_tfidf)
confusion_matrix_ridge = confusion_matrix(test_y, predictions_ridge)
print("confusion matrix for Ridge classifier ", confusion_matrix_ridge)
print("Ridge Classifier Accuracy Score ->", accuracy_score(predictions_ridge, test_y)*100)

# Naive Bayes Classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(train_x_tfidf, train_y)

# predicting the labels on the validation set
predictions_NB = Naive.predict(test_x_tfidf)

confusion_matrix_nb = confusion_matrix(test_y, predictions_NB)
print("Confusion Matrix for Naive Bayes ",confusion_matrix_nb)

# Using the accuracy score function to get the accuracy score
print("Naive Bayes Accuracy Score ->", accuracy_score(predictions_NB, test_y)*100)

# SVM Classifier
SVM = svm.SVC()
SVM.fit(train_x_tfidf, train_y)
predictions_svm = SVM.predict(test_x_tfidf)

confusion_matrix_svm = confusion_matrix(test_y, predictions_svm)
print("confusion matrix for svm ", confusion_matrix_svm)
print("SVM Accuracy Score ->", accuracy_score(predictions_svm, test_y)*100)

# Logistic Classifier
logreg = LogisticRegression()
logreg.fit(train_x_tfidf, train_y)
predictions_logistic = logreg.predict(test_x_tfidf)
confusion_matrix_log = confusion_matrix(test_y, predictions_logistic)
print("Confusion Matrix for logistic", confusion_matrix_log)
print("Logistic Accuracy Score ->", accuracy_score(predictions_logistic, test_y)*100)


