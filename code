###Pharma Sentiment Analysis Master Code###

#Import the necessary libraries
import nltk
import sklearn
import re

#Convert database into usable format
import pandas as pd
reviews = pd.read_csv(r'C:\Users\fj142\Dropbox\FJR 2019 Research Project\Airline Sentiment ML Example Project\TRAINING80.csv', encoding='latin-1')

#Convert ratings on a 1-10 scale to a sentiment: positive, negative, or neutral
def sentiment_calc(rating):
    if rating <= 4:
        return 'negative'
    elif 5 <= rating <= 6:
        return 'neutral'
    else:
        return 'positive'
reviews['drug_sentiment'] = reviews.RATING.apply(sentiment_calc)

drug_counts = reviews.drug_sentiment.value_counts()
number_of_reviews = reviews.USERID.count()
print(drug_counts)



###Preprocessing###



#Importing libraries and combing through stop words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
dont_exclude = ["couldn't", 'most', "weren't", "shouldn't", 'same', 'few', "mightn't", "shan't", 'after', 'not', 'below', 'at', "aren't", "didn't", "hadn't", 'but', 'just', 'any', "haven't", 'no', "don't", "should've", 'off', "hasn't", "wouldn't", "needn't", "wasn't", "isn't", "won't", "doesn't", 'all']
[stop_words.remove(x) for x in dont_exclude]
wordnet_lemmatizer = WordNetLemmatizer()

#Input a review as a string and outputs the normalized review parsed as a list
def normalizer(review):
    only_letters = re.sub("[^a-zA-Z]", " ",review)
    tokens = nltk.word_tokenize(only_letters)[:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

#Create a new column in the database with the normalized review
reviews['normalized_review'] = reviews.REVIEW.apply(normalizer)

#Imput a review that has been put through the normalizer function and output a list of unigrams and bigrams
from nltk import ngrams
def ngrams(input_list):
    unigrams = input_list
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    return unigrams+bigrams

#Makes a list back into a string
def joiner(gram):
    return " ".join(gram)

#Create a new column in the database with the ngrams
reviews['grams'] = reviews.normalized_review.apply(ngrams)

#Counter function
import collections
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

#20 most commonly used unigrmas/bigrams associated with positive, negative, and neutral reviews
reviews[(reviews.drug_sentiment == 'positive')][['grams']].apply(count_words)['grams'].most_common(20)
reviews[(reviews.drug_sentiment == 'negative')][['grams']].apply(count_words)['grams'].most_common(20)
reviews[(reviews.drug_sentiment == 'neutral')][['grams']].apply(count_words)['grams'].most_common(20)



###Preparing the Data###



#Vectorizing the Data
import numpy as np
from scipy.sparse import hstack

#Using Count Vectorization
#from sklearn.feature_extraction.text import CountVectorizer
#count_vectorizer = CountVectorizer(ngram_range=(1,2))

#Using TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=8500)

#Enter the pre-processed reviews to be vectorized
reviews['normalized_review2'] = reviews.normalized_review.apply(joiner)
vectorized_data = count_vectorizer.fit_transform(reviews.normalized_review2)
features = count_vectorizer.get_feature_names()

#Convert sentiments into targets that can be used by the classifier
def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]
targets = reviews.drug_sentiment.apply(sentiment2target)

#Split the data into training and test data

#Data is stratified based on T-Area and split randomly in Excel and exported as two files for use: Training and Test
test_datafile = pd.read_csv(r'C:\Users\fj142\Dropbox\FJR 2019 Research Project\Airline Sentiment ML Example Project\TEST20.csv', encoding='latin-1')
test_datafile['normalized_review'] = test_datafile.REVIEW.apply(normalizer).apply(joiner)
data_train = vectorized_data
data_test = count_vectorizer.transform(test_datafile.normalized_review)
targets_train = targets
targets_test = test_datafile.RATING.apply(sentiment_calc).apply(sentiment2target)



###Fitting The Classifiers###



#SVM as Classifier
from sklearn import svm
clf = svm.SVC(gamma=0.01, C=0.25, probability=True, class_weight='balanced', kernel='linear')

#SVM + OneVsRest as Classifier
#from sklearn.multiclass import OneVsRestClassifier
#clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=0.25, probability=True, class_weight='balanced', kernel='linear'))
#0.05, 'POLY', degree=2

#Random Forest as Classifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
#clf = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=0)

#Multinomial Naive Bayes as Classifier
#import numpy as np
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)

#Fitting the Training Data
clf_output = clf.fit(data_train, targets_train)



###Evaluating the Results###



#K-Fold Cross-Validation
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y, K):
    scores = cross_val_score(clf, X, y, cv=K)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))

#Cross-Validation with five folds
evaluate_cross_validation(clf, data_train, targets_train, 5)

#Averaged Accuracy Score: Training and Test Data
print(clf.score(data_train, targets_train))
print(clf.score(data_test, targets_test))

#Saving the Classifier
import pickle
#Exporting
save = pickle.dumps(clf)
#Importing
clf2 = pickle.loads(save)

#Helper Functions for Calculating Metrics

#Predicts if a given review is Negative(0), Neutral(1), or Positive(2)
def predictor(review):
    return clf2.predict(count_vectorizer.transform([review]))[0]
    
#Returns true if a given review is in the inputted therapeutic area.
def is_area(area):
    return test_datafile['THERAPEUTICAREA'] == area

#Uses is_area to filter out all reviews that are not in the inputted therapeutic area.
def specific_area(area):
    return test_datafile[is_area(area)]


#Exporting all of the predicted polarities to Excel
test_datafile['PREDICTED'] = test_datafile.REVIEW.apply(predictor)


#Confusion Matrix Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_true = targets_test
y_pred = clf2.predict(data_test)

#Data by Therapeutic Area
def metrics_by_area(area):
    data = specific_area(area)
    y_true = data.SENTIMENT  
    y_pred = data.REVIEW.apply(predictor)
    print((accuracy_score(y_true, y_pred)))
    return classification_report(y_true, y_pred, target_names=reviews['drug_sentiment'].unique())

#All Test Data
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=reviews['drug_sentiment'].unique()))

#By Therapeutic Area
print(metrics_by_area('Sexual Health/Eurological'))



###Chart Representations###



#All Data
import matplotlib.pyplot as plt
marginal_probs = list(map(lambda p : p[0], margin))
n, bins, patches = plt.hist(marginal_probs, 25, facecolor='blue', alpha=0.75)
plt.title('Marginal confidence histogram - All data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()

#Positive Data
positive_test_data = list(filter(lambda row : row[0]==2, hstack((targets_test[:,None], data_test)).toarray()))
positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))
marginal_positive_probs = list(map(lambda p : marginal(p), positive_probs))
n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='green', alpha=0.75)
plt.title('Marginal confidence histogram - Positive data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()

#Neutral Data
positive_test_data = list(filter(lambda row : row[0]==1, hstack((targets_test[:,None], data_test)).toarray()))
positive_probs = clf.predict_proba(list(map(lambda r : r[1:], positive_test_data)))
marginal_positive_probs = list(map(lambda p : marginal(p), positive_probs))
n, bins, patches = plt.hist(marginal_positive_probs, 25, facecolor='gold', alpha=0.75)
plt.title('Marginal confidence histogram - Neutral data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()

#Negative Data
negative_test_data = list(filter(lambda row : row[0]==0, hstack((targets_test[:,None], data_test)).toarray()))
negative_probs = clf.predict_proba(list(map(lambda r : r[1:], negative_test_data)))
marginal_negative_probs = list(map(lambda p : marginal(p), negative_probs))
n, bins, patches = plt.hist(marginal_negative_probs, 25, facecolor='red', alpha=0.75)
plt.title('Marginal confidence histogram - Negative data')
plt.ylabel('Count')
plt.xlabel('Marginal confidence')
plt.show()
