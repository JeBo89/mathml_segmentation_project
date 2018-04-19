import pprint

from lxml import etree
import regex as re
import json
from pprint import pprint
import uuid


class mathml_segmentation:
    NS = ["http: // www.w3.org / 1998 / Math / MathML"]
    current_id = 0

    def __init__(self):
        self.new_id()
        return None
    def new_id(self):
        self.current_id = uuid.uuid1()
        return self.current_id
    def get_id(self):
        return self.current_id
    def parse_mathml(self,mathml_xml,math_context):
        l =[]
        if (mathml_xml.getchildren() != []):
            d = {}
            s = etree.tostring(mathml_xml, encoding="unicode", method="xml",xml_declaration=False).replace(" ", "").replace("\n", "")
            # s = re.sub(r'([\\n\s]*?)','', s, flags=re.IGNORECASE)
            d["Formula"] = s
            d["Mathematical_Context"] = math_context
            d["id_"] = self.get_id()
            d["id"] = self.new_id()
            parent_id = d["id"]

            l.append(d)
        for children in mathml_xml.getchildren():
            if children.getchildren() != []:
                # HAS CHILDREN
                l = l + self.parse_mathml(children,math_context)
            else:
                # NO CHILDREN
                if children.tag == '{http://www.w3.org/1998/Math/MathML}mi':
                    d = {}
                    d["Symbol"] =  children.text
                    d["Mathematical_Context"] = math_context
                    d["id_"] = self.get_id()
                    d["id"] = self.new_id()
                    d["parent_id"] = parent_id
                    l.append(d)
                elif children.tag == '{http://www.w3.org/1998/Math/MathML}mo':
                    d = {}
                    d['Operator'] =   children.text
                    d["Mathematical_Context"] = math_context
                    d["id_"] = self.get_id()
                    d["id"] = self.new_id()
                    d["parent_id"] = parent_id
                    l.append(d)

        return l



def main():

    mathml_xml_string = """<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math>"""

    # mathml_xml_string = """
    #        <math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\" alttext=\"{\\displaystyle f(x+T)=f(x)}\">
    #            <semantics>
    #                <mrow class=\"MJX-TeXAtom-ORD\">
    #                  <mstyle displaystyle=\"true\" scriptlevel=\"0\">
    #                    <mi>f</mi>
    #                    <mo stretchy=\"false\">(</mo>
    #                    <mi>x</mi>
    #                    <mo>+</mo>
    #                    <mi>T</mi>
    #                    <mo stretchy=\"false\">)</mo>
    #                    <mo>=</mo>
    #                    <mi>f</mi>
    #                    <mo stretchy=\"false\">(</mo>
    #                    <mi>x</mi>
    #                    <mo stretchy=\"false\">)</mo>
    #                  </mstyle>
    #                </mrow>
    #                <annotation encoding=\"application/x-tex\">{\\displaystyle f(x+T)=f(x)}</annotation>
    #            </semantics>
    #        </math>
    #        """

    mathml_xml_string = u"""
        <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
          <mrow>
            <mi>x</mi>
            <mo>=</mo>
            <mfrac>
              <mrow>
                <mrow>
                  <mo>-</mo>
                  <mi>b</mi>
                </mrow>
                <mo>&#xB1;</mo>
                <msqrt>
                  <mrow>
                    <msup>
                      <mi>b</mi>
                      <mn>2</mn>
                    </msup>
                    <mo>-</mo>
                    <mrow>
                      <mn>4</mn>
                      <mo>&#x2062;</mo>
                      <mi>a</mi>
                      <mo>&#x2062;</mo>
                      <mi>c</mi>
                    </mrow>
                  </mrow>
                </msqrt>
              </mrow>
              <mrow>
                <mn>2</mn>
                <mo>&#x2062;</mo>
                <mi>a</mi>
              </mrow>
            </mfrac>
          </mrow>
        </math>
           """
    ms = mathml_segmentation()

    # mathml_xml = etree.fromstring(mathml_xml_string)
    # pprint(ms.parse_mathml(mathml_xml))

    # print ("""@prefix mydb: <http://mydb.org/> .
    # @prefix schema: <http://schema.org/> .
    # @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    # @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    # """)


    data = json.load(open('queryResults.json'))

    # pprint(data['results']['bindings'][0]['Formula'])
    content = []
    # print ((data['results']['bindings'][0:2]))
    # exit(0)
    for data_d in data['results']['bindings']:
        mathml_xml_string = (data_d['Formula']["value"])
        # print (mathml_xml_string)
        mathml_xml = etree.fromstring(mathml_xml_string)
        math_context = data_d['Label']['value']
        content = content +  (ms.parse_mathml(mathml_xml,math_context))

    # pprint(content)
    make_RDF(content)


def make_RDF(contents):

    from rdflib import Dataset, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

    host  = "http://localhost:5820/MATH"
    # A namespace for our resources
    data = host + '/resource/'
    DATA = Namespace(data)
    # A namespace for our vocabulary items (schema information, RDFS, OWL classes and properties etc.)
    vocab = host + '/vocab/'
    VOCAB = Namespace(host + '/vocab/')

    # The URI for our graph
    graph_uri = URIRef(host+ '/resource/mathentitygraph')

    # We initialize a dataset, and bind our namespaces
    dataset = Dataset()
    dataset.bind('data', DATA)
    dataset.bind('vocab', VOCAB)

    # We then get a new graph object with our URI from the dataset.
    graph = dataset.graph(graph_uri)

    dataset.default_context.parse("vocab.ttl",format="turtle")
    # IRI baker is a library that reliably creates valid (parts of) IRIs from strings (spaces are turned into underscores, etc.).
    from iribaker import to_iri

    # Let's iterate over the dictionary, and create some triples
    # Let's pretend we know exactly what the 'schema' of our CSV file is
    for row in contents:
        # `id` is the primary key and we use it as our primary resource, but we'd also like to use it as a label
        # print (row)
        id = URIRef((data + str(row['id']))) # primary key for the object

        mathematical_context = Literal(row['Mathematical_Context'], datatype=XSD['string'])
        graph.add((id, VOCAB['mathematical_context'], mathematical_context))

        id_ = URIRef((data + str(row['id_'])))
        graph.add((id, VOCAB['previous_id'] ,id_))

        if ('Formula' in row):
            formula = Literal(row['Formula'], datatype=XSD['string'])
            dataset.add((id, VOCAB['formula'], formula))

        if ('Symbol' in row):
            symbol = Literal(row['Symbol'], datatype=XSD['string'])
            dataset.add((id, VOCAB['symbol'], symbol))
            parent_id = URIRef((data + str(row['parent_id'])))
            dataset.add((id, VOCAB['part_of'], parent_id))


        if ('Operator' in row):
            operator = Literal(row['Operator'], datatype=XSD['string'])
            dataset.add((id, VOCAB['operator'], operator))
            parent_id = URIRef((data + str(row['parent_id'])))
            dataset.add((id, VOCAB['part_of'], parent_id))


        # print
        # person = URIRef(to_iri(data + row['Name']))
        # name = Literal(row['Name'], datatype=XSD['string'])
        # # `Country` is a resource
        # country = URIRef(to_iri(data + row['Country']))
        # # But we'd also like to use the name as a label (with a language tag!)
        # country_name = Literal(row['Country'], lang='en')
        # # `Age` is a literal (an integer)
        # age = Literal(int(row['Age']), datatype=XSD['int'])
        # # `Favourite Colour` is a resource
        # colour = URIRef(to_iri(data + row['Favourite Colour']))
        # colour_name = Literal(row['Favourite Colour'], lang='en')
        # # `Place` is a resource
        # place = URIRef(to_iri(data + row['Place']))
        # place_name = Literal(row['Place'], lang='en')
        # # `Address` is a literal (a string)
        # address = Literal(row['Address'], datatype=XSD['string'])
        # # `Hobby` is a resource
        # hobby = URIRef(to_iri(data + row['Hobby']))
        # hobby_name = Literal(row['Hobby'], lang='en')
        #
        # # All set... we are now going to add the triples to our graph
        # graph.add((person, VOCAB['name'], name))
        # graph.add((person, VOCAB['age'], age))
        # graph.add((person, VOCAB['address'], address))
        #
        # # Add the place and its label
        # graph.add((person, VOCAB['place'], place))
        # graph.add((place, VOCAB['name'], place_name))
        #
        # # Add the country and its label
        # graph.add((person, VOCAB['country'], country))
        # graph.add((country, VOCAB['name'], country_name))
        #
        # # Add the favourite colour and its label
        # graph.add((person, VOCAB['favourite_colour'], colour))
        # graph.add((colour, VOCAB['name'], colour_name))
        #
        # # Add the hobby and its label
        # graph.add((person, VOCAB['hobby'], hobby))
        # graph.add((hobby, VOCAB['name'], hobby_name))

    with open('math_db.trig','w') as f:
        graph.serialize(f, format='trig')
    # print (dataset.serialize(format='trig'))

if __name__ == '__main__':
    main()
