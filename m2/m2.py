def duplicate_dictionary_check(d,specific_word=''):
    for key_a in d:
       for key_b in d:
           if key_a == key_b:
               break
           for item in d[key_a]:
               if (item in d[key_b]):
                   if specific_word:
                        if specific_word == item:
                            print key_a,key_b,"found specific word:", specific_word
                   print key_a,key_b,"found match:",item



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

def is_xml_in_list(l,uri1,uri2):
    for row1 in l:
        if (row1['uri1'] == uri1 and row1['uri2'] == uri2)  or  (row1['uri1'] == uri2 and row1['uri2'] == uri1) :
            return True
    return False

for row1 in l:
    for row2 in l:
        if row1['xml'] == row2['xml'] \
                and row1['uri'] != row2['uri'] \
                and (row2["uri"],row1["uri"]) not in same_set\
                and row1['label'] != row2['label']:
                # and not is_xml_in_list(same_set,row1["uri"],row2["uri"]):
            # print ("match" , row1["xml"] , row1["label"],row2["label"])

            # d = {}
            # d["uri1"] = row1["uri"]
            # d["uri2"] = row2["uri"]
            # d["label1"] = row1["label"]
            # d["label2"] = row2["label"]
            # d["xml"] = row1["xml"]
            same_set.add((row1["uri"],row2["uri"]))
            i=i+1




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


