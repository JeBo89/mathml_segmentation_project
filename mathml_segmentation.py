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
    for data_d in data['results']['bindings'][0:10]:
        mathml_xml_string = (data_d['Formula']["value"])
        # print (mathml_xml_string)
        mathml_xml = etree.fromstring(mathml_xml_string)
        math_context = data_d['Label']['value']
        content = content +  (ms.parse_mathml(mathml_xml,math_context))

    # pprint(content)
    make_RDF(content)


def make_RDF(contents):


    host  = "http://localhost:5820/MATH"
    # A namespace for our resources
    data = host + '/resource/'
    DATA = Namespace(data)
    # A namespace for our vocabulary items (schema information, RDFS, OWL classes and properties etc.)
    vocab = host + '/vocab/'
    VOCAB = Namespace(host + '/vocab/')

    # The URI for our graph
    graph_uri = URIRef(host+ '/resource/math_entity_graph')

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
        Mathematical_entity = Literal('Mathematical_entity', datatype=XSD['string'])
        graph.add((id, RDFS.label,Mathematical_entity))

        mathematical_context = Literal(row['Mathematical_Context'], datatype=XSD['string'])
        graph.add((id, VOCAB['mathematical_context'], mathematical_context))

        id_ = URIRef((data + str(row['id_'])))
        graph.add((id, DATA['previous_id'] ,id_))

        if ('Formula' in row):
            formula = Literal(row['Formula'], datatype=XSD['string'])
            graph.add((id, VOCAB['formula'], formula))

        if ('Symbol' in row):
            symbol = Literal(row['Symbol'], datatype=XSD['string'])
            graph.add((id, VOCAB['symbol'], symbol))
            parent_id = URIRef((data + str(row['parent_id'])))
            graph.add((id, VOCAB['part_of'], parent_id))


        if ('Operator' in row):
            operator = Literal(row['Operator'], datatype=XSD['string'])
            graph.add((id, VOCAB['operator'], operator))
            parent_id = URIRef((data + str(row['parent_id'])))
            graph.add((id, VOCAB['part_of'], parent_id))


    with open('math_db.trig','w') as f:
        graph.serialize(f, format='trig')
    print (dataset.serialize(format='trig'))

if __name__ == '__main__':
    main()
