import pprint

from lxml import etree
import regex as re
import json
from pprint import pprint
import uuid
from iribaker import to_iri

from rdflib import Dataset, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

class mathml_segmentation:
    NS = ["http: // www.w3.org / 1998 / Math / MathML"]
    current_id = 0
    current_formula_id = 0

    left_symbol = None
    right_symbol = None
    operator_add = None
    operator_subtract = None


    def __init__(self):
        self.new_id()
        return None
    def new_id(self):
        self.current_id = uuid.uuid1()
        return self.current_id
    def get_id(self):
        return self.current_id

    def reset_formula_id(self):
        self.current_formula_id = 0
    def new_formula_id(self):
        self.current_formula_id = uuid.uuid1()
        return self.current_formula_id
    def get_formula_id(self):
        return self.current_formula_id

    def parse_mathml(self,mathml_xml,label_value='',description_value=''):
        l =[]
        if (mathml_xml.getchildren() != []):
            d = {}
            s = etree.tostring(mathml_xml, encoding="unicode", method="xml",xml_declaration=False).replace(" ", "").replace("\n", "")
            # s = re.sub(r'([\\n\s]*?)','', s, flags=re.IGNORECASE)
            d["Formula"] = s
            d["label"] = label_value
            d["description"]=description_value
            d["id_"] = self.get_formula_id()
            d["id"] = self.new_formula_id()
            parent_id = d["id"]

            l.append(d)
        for children in mathml_xml.getchildren():

            if children.getchildren() != []:
                # HAS CHILDREN
                l = l + self.parse_mathml(children,label_value,description_value)
            else:
                # NO CHILDREN

                if children.tag == '{http://www.w3.org/1998/Math/MathML}mi':
                    d = {}
                    d["Symbol"] =  children.text
                    d["label"] = label_value
                    d["description"] = description_value
                    d["id_"] = self.get_id()
                    d["id"] = self.new_id()
                    d["parent_id"] = parent_id
                    l.append(d)

                    if self.left_symbol == None:
                        self.left_symbol = d["id"]
                    elif self.operator_add != None:
                        self.right_symbol = d["id"]
                        d["Function_add"] = self.operator_add
                        d["left"] = self.left_symbol
                        d["right"] = self.right_symbol
                        l.append(d)
                        self.left_symbol = self.right_symbol
                        self.right_symbol = None
                        self.operator_add = None
                    elif self.operator_subtract != None:
                        d["Function_subtract"] = self.operator_subtract
                        d["left"] = self.left_symbol
                        d["right"] = self.right_symbol
                        l.append(d)
                        self.left_symbol = self.right_symbol
                        self.right_symbol = None
                        self.operator_subtract = None




                elif children.tag == '{http://www.w3.org/1998/Math/MathML}mo':
                    d = {}
                    d['Operator'] =   children.text
                    d["label"] = label_value
                    d["description"] = description_value
                    d["id_"] = self.get_id()
                    d["id"] = self.new_id()
                    d["parent_id"] = parent_id
                    l.append(d)
                    if children.text == "+":
                        self.operator_add = d["id"]
                    elif children.text =="-":
                        self.operator_subtract = d["id"]

        return l



    def make_RDF(self,contents):

        host  = "http://localhost:5820/MATH"
        # A namespace for our resources
        data = host +'/'# + '/resource/'
        DATA = Namespace(data)
        # A namespace for our vocabulary items (schema information, RDFS, OWL classes and properties etc.)
        vocab = host #+ '/vocab/'
        VOCAB = Namespace(host + '/vocab/')

        # The URI for our graph
        graph_uri = URIRef(host)#+ '/graph')

        # We initialize a dataset, and bind our namespaces
        dataset = Dataset()
        dataset.bind('data', DATA)
        dataset.bind('vocab', VOCAB)

        # We then get a new graph object with our URI from the dataset.
        graph = dataset.graph(graph_uri)

        dataset.default_context.parse("vocab.ttl",format="turtle")
        # IRI baker is a library that reliably creates valid (parts of) IRIs from strings (spaces are turned into underscores, etc.).


        for row in contents:


            id = URIRef((data + str(row['id']))) # primary key for the object

            id_ = URIRef((data + str(row['id_'])))
            # graph.add((id, VOCAB['previous_id'] ,id_))
            # graph.add((id, RDF.type, OWL.NamedIndividual))


            if ('Formula' in row):

                graph.add((id,RDF.type,VOCAB['Formula']))
                formula_xml = Literal(row['Formula'], datatype=XSD['string'])
                graph.add((id, VOCAB['xml'], formula_xml))
                Description = Literal(row['description'], datatype=XSD['string'])
                graph.add((id,VOCAB['Description'],Description))
                label = Literal(row['label'], datatype=XSD['string'])
                graph.add((id,VOCAB['label'],label))
                if row['id_'] !=0:
                    parent_id = URIRef((data + str(row['id_'])))
                    graph.add((id, VOCAB['subFormulaOf'], parent_id))


            if ('Symbol' in row):
                graph.add((id, RDF.type, VOCAB['Symbol']))
                symbol = Literal(row['Symbol'], datatype=XSD['string'])
                graph.add((id, VOCAB['label'], symbol))
                parent_id = URIRef((data + str(row['parent_id'])))
                graph.add((id, VOCAB['partOf'], parent_id))


            if ('Operator' in row):
                graph.add((id, RDF.type, VOCAB['Operator']))
                operator = Literal(row['Operator'], datatype=XSD['string'])
                graph.add((id, VOCAB['label'], operator))
                parent_id = URIRef((data + str(row['parent_id'])))
                graph.add((id, VOCAB['partOf'], parent_id))


            if ('Function_add' in row):
                # print(row)
                id = URIRef((data + str(row['Function_add'])))
                graph.add((id, RDF.type, VOCAB['Operator']))
                left = URIRef((data + str(row['left'])))
                graph.add((id, VOCAB['left'], left))
                right = URIRef((data + str(row['right'])))
                graph.add((id, VOCAB['right'], right))

            if ('Function_subtract' in row):
                print(row)
                id = URIRef((data + str(row['Function_subtract'])))
                graph.add((id, RDF.type, VOCAB['Operator']))
                left = URIRef((data + str(row['left'])))
                graph.add((id, VOCAB['left'], left))
                right = URIRef((data + str(row['right'])))
                graph.add((id, VOCAB['right'], right))

        with open('math_db.trig','w') as f:
            graph.serialize(f, format='trig')
        # print (dataset.serialize(format='trig'))


def main():


    ms = mathml_segmentation()


    data = json.load(open('queryResults.json'))

    content = []
    # pprint ((data['results']['bindings'][0:1]))
    # exit(0)
    i = 0
    for data_d in data['results']['bindings'][0:1]:
        mathml_xml_string = (data_d['Formula']["value"])
        if (len(mathml_xml_string) > 0 ):
            i+=1;
            # print (len(mathml_xml_string))
        # print (mathml_xml_string)
            mathml_xml = etree.fromstring(mathml_xml_string)
            label_value = data_d['Label']['value']
            if 'Description' in data_d:
                description_value = data_d['Description']['value']
            else:
                description_value=""
            ms.reset_formula_id()
            content = content +  (ms.parse_mathml(mathml_xml,label_value,description_value))

    pprint(content)
    print(i)
    ms.make_RDF(content)



if __name__ == '__main__':
    main()
