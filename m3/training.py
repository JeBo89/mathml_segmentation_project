from __future__ import print_function
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
from sklearn.metrics import confusion_matrix, roc_curve, pairwise_distances_argmin, silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score



from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import matplotlib.cm as cm
from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

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

# df = df[df['count'] > 100]
# print df['subject'].unique()
# exit()
print (df.groupby(['subject']).size())

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
integer_encoded = label_encoder_model.transform(values)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder_model =onehot_encoder.fit(integer_encoded)



def encode_transform(data):
    values = array(data)
    integer_encoded = label_encoder_model.transform(values)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder_model.transform(integer_encoded)
    return onehot_encoded

# print encode_transform(['a'])[0]
#
# exit()
# word 2 vec experiment

from gensim.models import Word2Vec
# build vocabulary and train model
Word2Vec_model = gensim.models.Word2Vec(
    symbol_list,
    size=100,
    window=4,
    min_count=1,
    workers=4)
Word2Vec_model.train(symbol_list, total_examples=len(symbol_list))


def vectorize(s):
    v = Word2Vec_model[s]
    # v = encode_transform([s])[0]
    return v


# print vectorize('a')


d = []

# class_names = ['Life sciences', 'Physical sciences', 'Applied sciences']
class_names = ['Life sciences', 'Physical sciences', 'Applied sciences', 'Society', 'Professional studies', 'Auxiliary sciences of history', 'Humanities', 'Environmental studies', 'Categories by parameter']

for t in tagged_document:
    v = []
    for s in t['words']:
        v.append(vectorize(s))
        np.array(v)
    meanv_v = np.mean(v, axis=0)
    # print  meanv_v
    t['features'] = meanv_v
    if t['tags'][0] == class_names[0]:
        t['tag'] = 0
    if t['tags'][0] == class_names[1]:
        t['tag'] = 1
    if t['tags'][0] == class_names[2]:
        t['tag'] = 2
    if t['tags'][0] == class_names[3]:
        t['tag'] = 3
    if t['tags'][0] == class_names[4]:
        t['tag'] = 4
    if t['tags'][0] == class_names[5]:
        t['tag'] = 5
    if t['tags'][0] == class_names[6]:
        t['tag'] = 6
    if t['tags'][0] == class_names[7]:
        t['tag'] = 7
    if t['tags'][0] == class_names[8]:
        t['tag'] = 8


    d.append(t)




df =  pd.DataFrame(d)
# print df.head()
# df = pd.get_dummies(df,drop_first=True)


X = list(df['features'])
X = np.array(X)
from scipy import sparse
# X=sparse.csr_matrix(X)

# print(b)
from sklearn.decomposition.truncated_svd import TruncatedSVD
from sklearn.decomposition.sparse_pca import SparsePCA
from sklearn.decomposition import  dict_learning_online


sparsepca =SparsePCA(n_components=200)
X= sparsepca.fit_transform(X)

pca = TruncatedSVD(n_components=2)
# X = pca.fit_transform(X)


# X = X.reshape(-1, 1)

Y = df['tag']

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.2,  random_state=42,stratify=Y)
X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.2,  random_state=42)
#


print ('Clustering')


km = KMeans(n_clusters=len(class_names), init='k-means++', max_iter=100, n_init=1,
                verbose=True)
range_n_clusters = [2, 3, 4, 5, 6 , 9 ]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    print (len(X))
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.savefig('Silhouette'+ str(n_clusters))


#
# print("Clustering sparse data with %s" % km)
# t0 = time()
# km.fit(X)
# t_batch = time() - t0
# print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(Y, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(Y, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(Y, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(Y, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))
#
# print()
#
# # KMeans
#
# fig = plt.figure(figsize=(8, 3))
# fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
# colors = ['#4EACC5', '#FF9C34', '#4E9A06']
#
# # We want to have the same colors for the same cluster from the
# # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# # closest one.
# k_means_cluster_centers = np.sort(km.cluster_centers_, axis=0)
# k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
#
# ax = fig.add_subplot(1, 3, 1)
# for k, col in zip(range(len(class_names)), colors):
#     my_members = k_means_labels == k
#     cluster_center = k_means_cluster_centers[k]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w',
#             markerfacecolor=col, marker='.')
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=6)
# ax.set_title('KMeans')
# ax.set_xticks(())
# ax.set_yticks(())
# plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
#     t_batch, km.inertia_))
#
# plt.savefig('k_means_cluster_centers')

print ('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

from sklearn.metrics import r2_score,f1_score,roc_auc_score
from scipy.stats import spearmanr, pearsonr
# predicted_train = rf.predict(X_train)
predicted_test = rf.predict(X_test)
# f1_score = f1_score(y_test, predicted_test,average=None)

# roc_curve_score_test = roc_curve(y_test, predicted_test,pos_label=0)
# print('roc_auc_score: ',roc_curve_score_test)

# print('Out-of-bag R-2 score estimate: ' , rf.oob_score_ )
# print('F1 Score: ', f1_score )





# 10-Fold Cross validation
print ('cross_val_score: ', np.mean(cross_val_score(rf, X_train, y_train, cv=10)))

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




print ('-------------------------------------------------------------------')
print ('DecisionTreeClassifier')
from sklearn.tree import DecisionTreeClassifier
# rf = RandomForestClassifier()
dt = DecisionTreeClassifier( random_state=42 ,criterion = 'gini')

dt.fit(X_train, y_train)

# predicted_train = dt.predict(X_train)
predicted_test = dt.predict(X_test)

print ('cross_val_score: ', np.mean(cross_val_score(dt, X_train, y_train, cv=10)))

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



print ('-------------------------------------------------------------------')

print ('MLPClassifier')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100),max_iter=100)

mlp.fit(X_train, y_train)
#
# predicted_train = mlp.predict(X_train)
predicted_test = mlp.predict(X_test)

# f1_score = f1_score(y_test, predicted_test,average=None)
# spearman = spearmanr(y_test, predicted_test)
# pearson = pearsonr(y_test, predicted_test)
# # print('Out-of-bag R-2 score estimate: ' , mlp.oob_score_ )
# print('Test data Spearman correlation: ' ,   spearman[0] )
# print('Test data Pearson correlation: ' ,  pearson[0] )
# print('F1 Score: ', f1_score )
print ('cross_val_score: ', np.mean(cross_val_score(mlp, X_train, y_train, cv=10)))


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


print ('-------------------------------------------------------------------')

print ('LinearSVC')
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
print ('cross_val_score: ', np.mean(cross_val_score(svc, X_train, y_train, cv=10)))


cnf_matrix = confusion_matrix(y_test ,predicted_test)
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

exit()
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
use_hashing = True
use_idf = True
if use_hashing:
    if use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=df.n_features,
                                   stop_words='english', alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=df.n_features,
                                       stop_words='english',
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=df.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=df.use_idf)
X = vectorizer.fit_transform(df.data)
