#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:20:41 2020

@author: cecilialu
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os

pd.options.display.max_columns = None
pd.options.display.max_rows = None

reviews = pd.read_csv('reviews.csv', delimiter='\t')


#keeping a record of original labels
original_columns = ['Name','RatingValue','DatePublished','Review']


#binning rating values to positive, neutral and negative
def labelling(df):
    
    buckets = [0,2.5,3.5,5.5]
    df['Sentiment'] = pd.cut(df['RatingValue'], bins = buckets, labels=["negative","neutral","positive"], right = False)

 
    return df

#encoding binned sentiment labels
def encoding(df):
    
    le = preprocessing.LabelEncoder()
    df['Sentiment_Encoded'] = le.fit_transform(df['Sentiment'])
    
    
    return df, le

def print_confusion_matrix(matrix, labels):
    df = pd.DataFrame(data = matrix, index = labels, columns = labels)
    
    return df



    
#Ratings binned in to the 3 required bins and encode the labels
reviews_labelled = labelling(reviews)


"""
reduction method 1:
    reduce # of positive based on word counts of reviews: select by quantile/mean
"""



#reducing number of positives
pos_review = reviews_labelled[reviews_labelled['Sentiment'] == 'positive']['Review']
pos_review_wordcount = pos_review.apply(lambda x: len(x.split()))


# print(pos_review_wordcount[pos_review_wordcount > pos_review_wordcount.quantile(q = 0.75)].count())



below_75 = pos_review_wordcount[pos_review_wordcount < pos_review_wordcount.quantile(q = 0.75)].index
below_median = pos_review_wordcount[pos_review_wordcount < pos_review_wordcount.median()].index
below_mean = pos_review_wordcount[pos_review_wordcount < pos_review_wordcount.mean()].index


"""
reduction method 2: 
    make sure all three categories have the same number of obs
    randomly select obs from categories that have more obs
    
    all negative, neutral, and positive will have 158 obs each
"""

negative_count = reviews_labelled.groupby('Sentiment').count()['Review'].loc['negative']
neutral_count = reviews_labelled.groupby('Sentiment').count()['Review'].loc['neutral']
positive_count = reviews_labelled.groupby('Sentiment').count()['Review'].loc['positive']

count_reduced = min(min(negative_count, neutral_count), positive_count)

neg_raw = reviews_labelled[reviews_labelled['Sentiment'] == 'negative']
neu_raw = reviews_labelled[reviews_labelled['Sentiment'] == 'neutral']
pos_raw = reviews_labelled[reviews_labelled['Sentiment'] == 'positive']

neg_sampled = neg_raw.sample(n = count_reduced, random_state = 42)
neu_sampled = neu_raw.sample(n = count_reduced, random_state = 42)
pos_sampled = pos_raw.sample(n = count_reduced, random_state = 42)

selected = neg_sampled.index.copy()
selected = selected.union(neu_sampled.index)
selected = selected.union(pos_sampled.index)





#reduced using method 1
# reviews_reduced = reviews_labelled.drop(axis = 0, index = below_median)

#reduced using mehtod 2
reviews_reduced = reviews_labelled.iloc[selected]



# print('before reduction')
# print(reviews_labelled.groupby('Sentiment').count()['Review'],'\n')

# print('after reduction')
# print(reviews_reduced.groupby('Sentiment').count()['Review'],'\n')



#splitting training and validating set
train, valid = train_test_split(reviews_reduced, test_size = 0.33, random_state = 42)

#outputing train and valid to csv files
train[original_columns].to_csv('train.csv', index = False, sep ='\t')
valid[original_columns].to_csv('valid.csv', index = False, sep ='\t')



training_set = pd.read_csv('train.csv', sep ='\t')
validating_set = pd.read_csv('valid.csv', sep ='\t')

train_labelled = labelling(training_set)
valid_labelled = labelling(validating_set)

train_encoded, train_encoder = encoding(train_labelled)
valid_encoded, valid_encoder = encoding(valid_labelled)


X_train = train_encoded['Review']
y_train = train_encoded['Sentiment_Encoded']

X_valid = valid_encoded['Review']
y_valid = valid_encoded['Sentiment_Encoded']





#tokenizing text with CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
# print('# of reviews and size of vocabulary:')
# print(X_train_counts.shape,'\n')
# print('count of most frequent word:')
# print(X_train_counts.sum(axis=0).max(),'\n')

#TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print('traiing TF-IDF Shape')
# print(X_train_tfidf.shape, '\n')


#importing models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier



#Naive Bayes Pipline
from sklearn.pipeline import Pipeline
text_clf_NB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

"""
#training the model - NB
text_clf_NB.fit(X_train, y_train)

predicted_NB = text_clf_NB.predict(X_valid)

print('Naive Bayes accuracy:')
print(np.mean(predicted_NB == y_valid),'\n')
"""

#SVM
text_clf_SVM = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

"""
#training the model - SVM
text_clf_SVM.fit(X_train, y_train)

predicted_SVM = text_clf_SVM.predict(X_valid)
print('SVM accuracy:')
print(np.mean(predicted_SVM == y_valid),'\n')
"""
#hyperparameter tuning
from sklearn.model_selection import GridSearchCV

#Naive Bayes hyperparameter tuning
parameters_NB = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1, 1e-5),
}


gs_clf_NB = GridSearchCV(text_clf_NB, parameters_NB, cv=5, n_jobs=-1)


gs_clf_NB = gs_clf_NB.fit(X_train, y_train)

predicted_gs_NB = gs_clf_NB.predict(X_valid)

# print('GS Naive Bayes accuracy:')
# print(np.mean(predicted_gs_NB == y_valid),'\n')


# print('grid search Naive Bayes best score')
# print(gs_clf_NB.best_score_,'\n')

# print('grid search Naive Bayes best parameters')
# for param_name in sorted(parameters_NB.keys()):
#     print("%s: %r" % (param_name, gs_clf_NB.best_params_[param_name]))




#SVM Hyperparameter Tuning
parameters_SVM = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-1, 1e-7),
}


gs_clf_SVM = GridSearchCV(text_clf_SVM, parameters_SVM, cv=5, n_jobs=-1)


gs_clf_SVM = gs_clf_SVM.fit(X_train, y_train)

predicted_gs_SVM = gs_clf_SVM.predict(X_valid)

# print('GS SVM accuracy:')
# print(np.mean(predicted_gs_SVM == y_valid),'\n')

# print('grid search SVM best score')
# print(gs_clf_SVM.best_score_,'\n')

# print('grid search SVM best parameters')
# for param_name in sorted(parameters_SVM.keys()):
#     print("%s: %r" % (param_name, gs_clf_SVM.best_params_[param_name]))
    
print('===========================================')
print('\n')



#selecting the best model

if gs_clf_NB.best_score_ > gs_clf_SVM.best_score_:
    print('Naive Bayes model is selected','\n')
    # print(gs_clf_NB.best_score_)
    final_model_param = gs_clf_NB.best_params_
    final_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range = final_model_param['vect__ngram_range'])),
        ('tfidf', TfidfTransformer(use_idf=final_model_param['tfidf__use_idf'])),
        ('clf', MultinomialNB(alpha = final_model_param['clf__alpha'])),
])
else:
    print('SVM is selected','\n')
    # print(gs_clf_SVM.best_score_)
    final_model_param = gs_clf_SVM.best_params_
    final_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range = final_model_param['vect__ngram_range'])),
        ('tfidf', TfidfTransformer(use_idf=final_model_param['tfidf__use_idf'])),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=final_model_param['clf__alpha'], random_state=42,
                          max_iter=5, tol=None)),
])

print('===========================================')
print('\n')

final_clf.fit(X_train, y_train)

predicted = final_clf.predict(X_valid)



print('performace metric: (valid) ', '\n')
print('-------------------------------------------')

from sklearn import metrics

performance_report = metrics.classification_report(y_valid, predicted, target_names = valid_encoder.classes_)


print(performance_report)
print('-------------------------------------------','\n')

from sklearn.metrics import confusion_matrix

confusion_matrix_raw = metrics.confusion_matrix(valid_encoder.inverse_transform(y_valid), valid_encoder.inverse_transform(predicted), labels = valid_encoder.classes_)

confusion_matrix_printed = print_confusion_matrix(confusion_matrix_raw, valid_encoder.classes_)

print(confusion_matrix_printed,'\n')

print('===========================================')


if os.path.exists('test.csv'):
    print('\n')
    print('test.csv is imported...','\n')
    
    print('===========================================','\n')
    
    test_set = pd.read_csv('test.csv', sep ='\t')
    test_labelled = labelling(test_set)
    test_encoded, test_encoder = encoding(test_labelled)
    
    X_test = test_encoded['Review']
    y_test = test_encoded['Sentiment_Encoded']
    

    predicted_test = final_clf.predict(X_test)
    
    
    performance_report_test = metrics.classification_report(y_test, predicted_test, target_names = test_encoder.classes_)
    
    confusion_matrix_raw_test = metrics.confusion_matrix(test_encoder.inverse_transform(y_test), test_encoder.inverse_transform(predicted_test), labels = test_encoder.classes_)

    confusion_matrix_test_printed = print_confusion_matrix(confusion_matrix_raw_test, test_encoder.classes_)
    
    print('performace metric: (test) ', '\n')
    print('-------------------------------------------')
    print(performance_report_test)
    print('-------------------------------------------','\n')
    print(confusion_matrix_test_printed,'\n')
    
    print('===========================================')
    





