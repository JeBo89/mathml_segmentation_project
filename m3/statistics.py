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



df = pd.read_csv('formula_with_subject.csv',index_col=0).dropna(axis=0)
# print df.count()
# print(df.head())



# df_grouped = df.groupby('subject').agg('count')
df_grouped = df.groupby(['subject']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
print(df_grouped.head())

# exit()
f  = df_grouped[df_grouped['counts']>10]['subject'].tolist()
print(f)

df = df.loc[df['subject'].isin(f)]


# print(df.head())
print(df.count())

df.to_csv('training.csv')


#                           subject  counts
# 6           Fields of mathematics     891
# 7                   Life sciences     263
# 0                Applied sciences     231
# 12           Subfields of physics     217
# 8               Physical sciences     110
# 10           Professional studies      49
# 1   Auxiliary sciences of history      47
# 9      Political science theories      34
# 3          Branches of philosophy      27
# 11                        Society      25
# 2             Branches of biology      20
# 5           Environmental studies      19
# 4         Categories by parameter      17