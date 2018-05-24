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



df = pd.read_csv('training.csv',index_col=0).dropna(axis=0)
print df.count()
print(df.head())

print(df.groupby(['subject']).size().reset_index(name='counts').sort_values(by='counts', ascending=False))

