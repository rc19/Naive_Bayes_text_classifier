import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from pprint import pprint


# Function to print the most informative features of all classes
def show_top10(classifier, vectorizer, categories):
    print("\n Category \t\t\t Top 10 informative words\n")
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s \t\t %s" % (category, " ".join(feature_names[top10])))


# store the directory path in the data_home variable to store the downloaded dataset
train_set = fetch_20newsgroups(subset = 'train')
test_set = fetch_20newsgroups(subset = 'test')

#print names of all classes
pprint(list(train_set.target_names))

print("\n Vectorizing the data!")
tfidf_vect = TfidfVectorizer()
train_vector = tfidf_vect.fit_transform(train_set.data)
test_vector = tfidf_vect.transform(test_set.data)

#==============================================================================
# another alternative for vectorizing the test data is to directly use the function 
# "sklearn.datasets.fetch_20newsgroups_vectorized" which returns ready-to-use tfidf
# features instead of file names.
# 
#==============================================================================
print("\n Training started!")
clf = MultinomialNB(alpha = 1, fit_prior = 'false')
clf.fit(train_vector, train_set.target)
pred = clf.predict(test_vector)

#Evaluation of the model

acc = np.mean(pred == test_set.target)   #accuracy
f1 = metrics.f1_score(pred, test_set.target, average = 'macro')    #F1 score
print("\n Accuracy of the Naive Bayes model is: ", acc)
print("\n F1 score of the Naive Bayes model is: ", f1)
show_top10(clf, tfidf_vect, train_set.target_names)