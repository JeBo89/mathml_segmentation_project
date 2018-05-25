from SPARQLWrapper import SPARQLWrapper, JSON
import pprint
from lxml import etree, objectify
from StringIO import StringIO
from lxml import etree
import regex as re
import json
from pprint import pprint
import uuid
from iribaker import to_iri
from django.utils.encoding import smart_str, smart_unicode
import pandas as pd
from rdflib import Dataset, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

from nltk.corpus import stopwords
from string import ascii_lowercase
import pandas as pd
import gensim, os, re, pymongo, itertools, nltk, snowballstemmer

from numpy import array
import numpy as np

import gensim
from gensim.models.doc2vec import TaggedDocument, LabeledSentence, Doc2Vec

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




df = pd.read_csv('training.csv',index_col=0).dropna(axis=0)
# print df.count()
# print(df.head())


# df['count'] = df.groupby('subject').transform('sum').prod(1)

# print  df.groupby(['subject']).agg(['count'])


# df_grouped = pd.DataFrame( df.groupby(['subject']).size().reset_index(name='counts').sort_values(by='counts', ascending=False))
groups = df.groupby(['subject']).size()
df_grouped = pd.DataFrame( groups )
# print df_grouped.head()

# print '----------------------------------------------------------------'

# exit()

# df = df.combine_first(df_grouped)
# df = df_grouped.update(df)
df = df.join(df_grouped,  on='subject', how='right')


df.columns = ['formula_label', 'symbol_set', 'subject', 'count']

df = df[df['count'] > 200]
# print df['subject'].unique()
# exit()
print df.groupby(['subject']).size()

symbol_set_all = set()
for symbol_set  in (df['symbol_set'].tolist()):
    for symbol in symbol_set:
        symbol_set_all.add(symbol)

symbol_set_list = df['symbol_set'].tolist()
symbol_list = []
for symbol_set in symbol_set_list:
    symbol_list.append(list(set(symbol_set)))

# print symbol_set_all
# print len(word_set)

# training_df = pd.get_dummies(df['symbol_set'],drop_first=True)

# print(list(training_df))
# exit()



def tag_row(row):
    symbols_ = list(set(row['symbol_set']))
    tags_ = [row['subject']]
    tagged_row =  { 'words' : symbols_, 'tags'  : tags_
    }
    return tagged_row

df['tagged'] = df.apply(tag_row,axis=1)


tagged_document = df['tagged'].tolist()



# model = Doc2vec(tagged_document )




# word 2 vec experiment

from gensim.models import Word2Vec
# build vocabulary and train model
Word2Vec_model = gensim.models.Word2Vec(
    symbol_list,
    size=100,
    window=2,
    min_count=0,
    workers=4)
Word2Vec_model.train(symbol_list, total_examples=len(symbol_list))


d = []

for t in tagged_document:
    v = []
    for s in t['words']:
        v.append(Word2Vec_model[s])
        np.array(v)
    meanv_v =    np.mean(v, axis=0)
    # print  meanv_v
    t['features'] = meanv_v
    if t['tags'][0] == 'Life sciences':
        t['tag'] = 10
    if t['tags'][0] == 'Fields of mathematics':
        t['tag'] = 200
    if t['tags'][0] == 'Applied sciences':
        t['tag'] = 3000
    if t['tags'][0] == 'Subfields of physics':
        t['tag'] = 40000

    # ['Life sciences' 'Fields of mathematics' 'Physical sciences'
    #  'Applied sciences' 'Subfields of physics']
    # t['tag'] = t['tags'][0]
    d.append(t)

class_names = ['Life sciences' ,'Fields of mathematics' ,'Applied sciences', 'Subfields of physics']


df =  pd.DataFrame(d)
# print df.head()
# df = pd.get_dummies(df,drop_first=True)


X = list(df['features'])
X = np.array(X)
from scipy import sparse
X=sparse.csr_matrix(X)
# print(b)

# X = X.reshape(-1, 1)

Y = df['tag']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.1,  random_state=42,stratify=Y)



# exit()
print 'RandomForestClassifier'
from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
rf = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

from sklearn.metrics import r2_score,f1_score,roc_auc_score
from scipy.stats import spearmanr, pearsonr
# predicted_train = rf.predict(X_train)
predicted_test = rf.predict(X_test)
# f1_score = f1_score(y_test, predicted_test,average=None)
# roc_auc_score_test = roc_auc_score(y_test, predicted_test, average=None, sample_weight=None)

# print('Out-of-bag R-2 score estimate: ' , rf.oob_score_ )
# print('F1 Score: ', f1_score )
# print('roc_auc_score: ',roc_auc_score_test)



from sklearn.model_selection import cross_val_score
# 10-Fold Cross validation
print 'cross_val_score: ', np.mean(cross_val_score(rf, X_train, y_train, cv=2))
from sklearn.metrics import confusion_matrix
#
# print 'confusion_matrix',confusion_matrix(y_test, predicted_test)
#
#
# print rf.predict([Word2Vec_model['m'],Word2Vec_model['V']])
#
# print rf.predict([Word2Vec_model['a'],Word2Vec_model['b']])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('plot_confusion_matrix_RandomForestClassifier')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()
plt.savefig('plot_confusion_matrix_normal_RandomForestClassifier')




print '-------------------------------------------------------------------'
print 'DecisionTreeClassifier'
from sklearn.tree import DecisionTreeClassifier
# rf = RandomForestClassifier()
dt = DecisionTreeClassifier( random_state=42)
dt.fit(X_train, y_train)

# predicted_train = dt.predict(X_train)
predicted_test = dt.predict(X_test)

print 'cross_val_score: ', np.mean(cross_val_score(dt, X_train, y_train, cv=10))

cnf_matrix = confusion_matrix(y_test, predicted_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('plot_confusion_matrix_DecisionTreeClassifier')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()
plt.savefig('plot_confusion_matrix_normal_DecisionTreeClassifier')



print '-------------------------------------------------------------------'

print 'MLPClassifier'
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100,13),max_iter=500)

mlp.fit(X_train, y_train)
#
predicted_train = mlp.predict(X_train)
predicted_test = mlp.predict(X_test)

# f1_score = f1_score(y_test, predicted_test,average=None)
# spearman = spearmanr(y_test, predicted_test)
# pearson = pearsonr(y_test, predicted_test)
# # print('Out-of-bag R-2 score estimate: ' , mlp.oob_score_ )
# print('Test data Spearman correlation: ' ,   spearman[0] )
# print('Test data Pearson correlation: ' ,  pearson[0] )
# print('F1 Score: ', f1_score )
print 'cross_val_score: ', np.mean(cross_val_score(mlp, X_train, y_train, cv=10))


cnf_matrix = confusion_matrix(y_test, predicted_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('plot_confusion_matrix_MLPClassifier')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()
plt.savefig('plot_confusion_matrix_normal_MLPClassifier')


print '-------------------------------------------------------------------'

print 'LinearSVC'
from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=42)

svc.fit(X_train, y_train)
#
# predicted_train = svc.predict(X_train)
predicted_test = svc.predict(X_test)

# f1_score = f1_score(y_test, predicted_test,average=None)
# spearman = spearmanr(y_test, predicted_test)
# pearson = pearsonr(y_test, predicted_test)
# # print('Out-of-bag R-2 score estimate: ' , mlp.oob_score_ )
# print('Test data Spearman correlation: ' ,   spearman[0] )
# print('Test data Pearson correlation: ' ,  pearson[0] )
# print('F1 Score: ', f1_score )
print 'cross_val_score: ', np.mean(cross_val_score(svc, X_train, y_train, cv=10))


cnf_matrix = confusion_matrix(y_test, predicted_test)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('plot_confusion_matrix_LinearSVC')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

# plt.show()
plt.savefig('plot_confusion_matrix_normal_LinearSVC')


exit()



# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()
# mlb.fit(df['symbol_set'])
# def b(row):
#     return mlb.fit_transform(row)
#
# df['features'] = df['symbol_set'].apply(b)
# print df.head()
# df.to_csv('training_binary.csv')
# exit()
# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()
# symbol_set_list = df['symbol_set'].tolist()
# symbol_list = []
# for symbol_set in symbol_set_list:
#     symbol_list.append(list(set(symbol_set)))

# print list(set(symbol_set_all))
# exit()

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = list(set(symbol_set_all))
values = array(data)

# integer encode

label_encoder = LabelEncoder()
label_encoder_model = label_encoder.fit(values)
integer_encoded = label_encoder.transform(values)
# print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder_model =onehot_encoder.fit(integer_encoded)

onehot_encoded = onehot_encoder_model.transform(integer_encoded)

# print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted)


def encode_transform(data):
    values = array(data)
    integer_encoded = label_encoder.transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder_model.transform(integer_encoded)
    return onehot_encoded

# print encode_transform(['n'])

# df_2 = pd.get_dummies(df,drop_first=True)

# print(df_2)
# exit()
def encode_set(row):

    symbols_ = list(set(row['symbol_set']))

    return list(encode_transform(symbols_))

df['features'] = df.apply(encode_set,axis=1)

# print df['features'].tolist()[0]






# enc.fit(symbol_list)
# OneHotEncoder(categorical_features='all', dtype='numpy.string',
#        handle_unknown='error', n_values='auto', sparse=True)
# enc.n_values_
# # array([2, 3, 4])
# enc.feature_indices_
# # array([0, 2, 5, 9])
# enc.transform([[0, 1, 1]]).toarray()
# array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.]])
#
