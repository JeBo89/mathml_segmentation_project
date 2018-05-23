
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



sparql = SPARQLWrapper("http://localhost:5820/MATH/query")
sparql.setQuery("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT distinct ?formula_Label ?symbol_label

WHERE {
  GRAPH ?g {
  ?s <http://localhost:5820/MATH/vocab/partOf> ?o;
   	a <http://localhost:5820/MATH/vocab/Symbol>;
    <http://localhost:5820/MATH/vocab/label> ?symbol_label.
   
  ?o <http://localhost:5820/MATH/vocab/label> ?formula_Label . 
  }   
}
group by ?formula_Label  ?symbol_label
order by ?formula_Label

""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

l = []
print (len(l))
for result in results["results"]["bindings"]:
    # print result
    d = {}
    d["formula_label"] = result["formula_Label"]["value"].encode("utf8")
    d["symbol_label"] = result["symbol_label"]["value"].encode("utf8")
    # get_def_list(result["label"]["value"].encode("utf8"))
    l.append(d)

# pprint(l)
def make_set(x):
    # x is a DataFrame of group values
    x = set(x['symbol_label'])
    return x

df = pd.DataFrame.from_dict(l)
df['formula_set'] = df.groupby('formula_label').apply(make_set)

# print (df.groupby('formula_label').agg('count').sort_values(by=['symbol_label'], ascending=False)) #shows how many symbols in each group
# print(df)

def get_subject(row):
    print row
    # sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # sparql.setQuery("""
    #    prefix dbc: <http://dbpedia.org/resource/Category:>
    # prefix dct: <http://purl.org/dc/terms/>
    # prefix owl: <http://www.w3.org/2002/07/owl#>
    # prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    # prefix skos: <http://www.w3.org/2004/02/skos/core#>
    # prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    # prefix dbo: <http://dbpedia.org/ontology/>
    # select ?subject_Label WHERE
    # {
    #   	?s rdfs:label "Stokes' law"@en;
    #   		dct:subject  ?subject .
    #
    #   ?subject rdfs:label ?subject_Label . filter(lang(?subject_Label) = "en")
    # }limit 10
    # """)
    # sparql.setReturnFormat(JSON)
    # results = sparql.query().convert()
    #
    # l = []
    # for result in results["results"]["bindings"]:
    #     subject = result["subject_Label"]["value"]
    #     print subject
    #     return  subject

    return None








print df
# df['subject'] = df['formula_label'].apply(get_subject )