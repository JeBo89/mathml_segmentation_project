import pprint
import sys
from nltk.corpus import wordnet, stopwords
from itertools import product
from urlparse import urlparse


def nltk_similarity_score(wordx, wordy):
    sem1, sem2 = wordnet.synsets(wordx.decode("utf8")), wordnet.synsets(wordy.decode("utf8"))
    maxscore = 0
    for i, j in list(product(*[sem1, sem2])):
        score = i.wup_similarity(j)  # Wu-Palmer Similarity
        maxscore = score if maxscore < score else maxscore
    return (maxscore)


def_dic = {}
import urllib


def get_def_list(name):
    if name not in def_dic:
        print '------------------------'
        print name
        url = 'https://hdt.lod.labs.vu.nl/triple?g=%3Chttps%3A//hdt.lod.labs.vu.nl/graph/LOD-a-lot%3E&p=rdfs:label&o=%22' + name + '%22^^%3Chttp://www.w3.org/2001/XMLSchema%23string%3E'
        results = urllib.urlopen(url).read()


        results = results.split()
        for uri_ in results:
            if (
                        ('<http://bag.kadaster.nl/def#huisnummertoevoeging> ' not in uri_) and \
                    ('"'+name+'"^^<http://www.w3.org/2001/XMLSchema#string>' not in uri_) and \
                            ('<https://hdt.lod.labs.vu.nl/graph/LOD-a-lot>' not in uri_) and \
                            ('<http://www.w3.org/2000/01/rdf-schema#label>' not in uri_) and\
                            ('.'!= uri_)
                ):


                print uri_
                print '---------------'
                uri_ = uri_.replace('#','%23')
                uri_ = uri_.replace(':','%3A')
                url = 'https://hdt.lod.labs.vu.nl/triple?g=%3Chttps%3A//hdt.lod.labs.vu.nl/graph/LOD-a-lot%3E&s=' + uri_
                results1 = urllib.urlopen(url).read()
                print results1
        # print (results)
        # results = [word for word in results if word not in stopwords.words('english')]
        def_dic[name] = results
        # print('query' ,name , results)
    else:
        results = def_dic[name]
    return results






from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
sparql = SPARQLWrapper("http://localhost:5820/MATH/query")
sparql.setQuery("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT distinct ?label
    WHERE {
      GRAPH ?g {
     ?s ?p ?o;
     a <http://localhost:5820/MATH/vocab/Symbol>;
     <http://localhost:5820/MATH/vocab/label> ?label.
      }   

    }limit 10
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

l = []
for result in results["results"]["bindings"]:
    # print result
    d = {}
    # d["uri"] = result["s"]["value"].encode("utf8")
    # d["label"] = result["label"]["value"].encode("utf8")
    get_def_list(result["label"]["value"].encode("utf8"))
    l.append(get_def_list(result["label"]["value"].encode("utf8")))

# pprint.pprint(l)
exit(0)












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


# for row in same_set:
#     # graph.add((row[0], RDF.type, VOCAB['Formula']))
#     # graph.add((row[1], RDF.type, VOCAB['Formula']))
#     graph.add((URIRef(row[0]), OWL.sameas, URIRef(row[1])))
#
# with open('same_formula_db.trig', 'w') as f:
#     graph.serialize(f, format='trig')
#     # print (dataset.serialize(format='trig'))
#
#
