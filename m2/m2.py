import sys
from nltk.corpus import wordnet ,stopwords
from itertools import product
from urlparse import urlparse


def nltk_similarity_score(wordx,wordy):
    sem1, sem2 = wordnet.synsets(wordx.decode("utf8")), wordnet.synsets(wordy.decode("utf8"))
    maxscore = 0
    for i, j in list(product(*[sem1, sem2])):
        score = i.wup_similarity(j)  # Wu-Palmer Similarity
        maxscore = score if maxscore < score else maxscore
    return(maxscore)

def_dic = {}
import urllib
def get_def_list(name):
    if name not in def_dic:
        url = 'https://hdt.lod.labs.vu.nl/triple?g=%3Chttps%3A//hdt.lod.labs.vu.nl/graph/LOD-a-lot%3E&p=rdfs:label&o=%22'+name+'%22^^%3Chttp://www.w3.org/2001/XMLSchema%23string%3E'
        results = urllib.urlopen(url).read()


        results = results.replace(name,'')
        results = results.replace('""^^<http://www.w3.org/2001/XMLSchema#string>','')
        results = results.replace('<https://hdt.lod.labs.vu.nl/graph/LOD-a-lot>','')
        results = results.replace('<http://www.w3.org/2000/01/rdf-schema#label>','')
        results = results.replace('owl', '')
        results = results.replace('<http://lodlaundromat', '')
        results = results.replace('<http://www', '')
        results = results.replace('<http://', '')
        # results = results.replace('\n', '')
        results = results.replace(r'\s', '')
        results = results.replace('ontology', '')
        results = results.replace('ontologies', '')
        results = results.replace('Ontologie', '')
        results = results.replace('org', '')
        results = results.replace('edu', '')
        results = results.replace('\d', '')
        results = results.replace('/', ' ')
        results = results.replace('.', ' ')
        results = results.replace('#', ' ')

        results = results.split()
        results = [word for word in results if word not in stopwords.words('english')]
        def_dic[name]= results
        # print('query' ,name , results)
    else:
        results= def_dic[name]
    return results



from fuzzywuzzy import fuzz
from fuzzywuzzy import process
def fuzzy_similarity_score(wordx,wordy):
    return float(fuzz.ratio(wordx, wordy)/100)

def lod_similarity_score(wordx,wordy):
    maxscore = 0
    for l1 in get_def_list(wordx):
        for l2 in get_def_list(wordy):
            l1 = l1.lower()
            l2 = l2.lower()
            score = fuzzy_similarity_score(l1, l2)
            maxscore = score if maxscore < score else maxscore
    return float(maxscore)



# word1 = 'massive'
# word2 = 'mass'
#
# print(lod_similarity_score(word1,word2))
# print(nltk_similarity_score(word1,word2))
# print(fuzzy_similarity_score(word1,word2))

# exit(0)
mylabel = "m"

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql = SPARQLWrapper("http://localhost:5820/MATH/query")
sparql.setQuery("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT distinct *
    WHERE {
      GRAPH ?g {
        ?s a <http://localhost:5820/MATH/vocab/Formula> ;
        <http://localhost:5820/MATH/vocab/xml>  ?xml;
        <http://localhost:5820/MATH/vocab/label>  ?label.
      }   
      
    }
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()


l = []
for result in results["results"]["bindings"]:
    d = {}
    d["uri"] = result["s"]["value"].encode("utf8")
    d["label"] = result["label"]["value"].encode("utf8")
    d["xml"] = result["xml"]["value"].encode("utf8")
    l.append(d)


# import pandas as pd
#
# df = pd.DataFrame(l)
# print (df)
# exit(0)
#
# # print l
# import csv
# # toCSV = [{'name':'bob','age':25,'weight':200},
# #          {'name':'jim','age':31,'weight':180}]
# keys = l[0].keys()
# with open('all_data.csv', 'wb') as output_file:
#     dict_writer = csv.DictWriter(output_file, keys)
#     dict_writer.writeheader()
#     dict_writer.writerows(l)
#
# exit(0)



i=0

same_set = set()

def is_xml_in_list(uri1,uri2):
    for row1 in same_set:
        if (row1[0] == uri1 and row1[1] == uri2)  or  (row1[1] == uri2 and row1[0] == uri1) :
            return True
    return False

for row1 in l:
    for row2 in l:

        if row1['xml'] == row2['xml'] \
                and row1['uri'] != row2['uri'] \
                and (row2["uri"],row1["uri"]) not in same_set\
                and max(nltk_similarity_score(row1['label'] , row2['label']) , lod_similarity_score(row1['label'] , row2['label'])) > 0.5 \
                and not is_xml_in_list(row1["uri"],row2["uri"]):
            # print ("match" , row1["xml"] , row1["label"],row2["label"])
            same_set.add((row1["uri"],row2["uri"]))
            i=i+1


            sys.stdout.write('.')




# import pprint
# pprint.pprint(same_set)
print (i)
from rdflib import Dataset, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

host = "http://localhost:5820/MATH"
# A namespace for our resources
data = host + '/'  # + '/resource/'
DATA = Namespace(data)
# A namespace for our vocabulary items (schema information, RDFS, OWL classes and properties etc.)
vocab = host  # + '/vocab/'
VOCAB = Namespace(host + '/vocab/')

# The URI for our graph
graph_uri = URIRef(host)  # + '/graph')

# We initialize a dataset, and bind our namespaces
dataset = Dataset()
dataset.bind('data', DATA)
dataset.bind('vocab', VOCAB)

# We then get a new graph object with our URI from the dataset.
graph = dataset.graph(graph_uri)

dataset.default_context.parse("../vocab.ttl", format="turtle")
# IRI baker is a library that reliably creates valid (parts of) IRIs from strings (spaces are turned into underscores, etc.).


for row in same_set:
    # graph.add((row[0], RDF.type, VOCAB['Formula']))
    # graph.add((row[1], RDF.type, VOCAB['Formula']))
    graph.add((URIRef(row[0]), OWL.sameas ,URIRef(row[1])))


with open('same_formula_db.trig', 'w') as f:
    graph.serialize(f, format='trig')
    # print (dataset.serialize(format='trig'))


