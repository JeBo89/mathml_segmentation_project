import pprint

from lxml import etree
import regex as re
import json
from pprint import pprint


class mathml_segmentation:
    NS = ["http: // www.w3.org / 1998 / Math / MathML"]
    counter = 0

    def __init__(self):
        self.counter = 0
        return None

    def parse_mathml(self,mathml_xml):
        l =[]
        if (mathml_xml.getchildren() != []):
            d = {}
            s = etree.tostring(mathml_xml, encoding="unicode", method="xml",xml_declaration=False).replace(" ", "").replace("\n", "")
            # s = re.sub(r'([\\n\s]*?)','', s, flags=re.IGNORECASE)
            d["Formulas"] = s
            d["id"] = mathml_segmentation.counter
            mathml_segmentation.counter += 1
            l.append(d)
        for children in mathml_xml.getchildren():
            if children.getchildren() != []:
                # HAS CHILDREN
                l = l + self.parse_mathml(children)
            else:
                # NO CHILDREN
                if children.tag == '{http://www.w3.org/1998/Math/MathML}mi':
                    d = {}
                    d["Symbol"] =  children.text
                    d["id"] = mathml_segmentation.counter
                    mathml_segmentation.counter += 1
                    l.append(d)
                if children.tag == '{http://www.w3.org/1998/Math/MathML}mo':
                    d = {}
                    d['Operator'] =   children.text
                    d["id"] = mathml_segmentation.counter
                    mathml_segmentation.counter += 1
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
    for data_d in data['results']['bindings']:
        mathml_xml_string = (data_d['Formula']["value"])
        # print (mathml_xml_string)
        mathml_xml = etree.fromstring(mathml_xml_string)
        content.append (ms.parse_mathml(mathml_xml))

    pprint(content)
    # make_RDF(contents)


def make_RDF(contents):

    from rdflib import Dataset, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

    host  = ""
    # A namespace for our resources
    data = host + '/resource/'
    DATA = Namespace(data)
    # A namespace for our vocabulary items (schema information, RDFS, OWL classes and properties etc.)
    vocab = host + '/vocab/'
    VOCAB = Namespace(host + '/vocab/')

    # The URI for our graph
    graph_uri = URIRef(host+ '/resource/examplegraph')

    # We initialize a dataset, and bind our namespaces
    dataset = Dataset()
    dataset.bind('g20data', DATA)
    dataset.bind('g20vocab', VOCAB)

    # We then get a new graph object with our URI from the dataset.
    graph = dataset.graph(graph_uri)

    # IRI baker is a library that reliably creates valid (parts of) IRIs from strings (spaces are turned into underscores, etc.).
    from iribaker import to_iri

    # Let's iterate over the dictionary, and create some triples
    # Let's pretend we know exactly what the 'schema' of our CSV file is
    for row in contents:
        # `Name` is the primary key and we use it as our primary resource, but we'd also like to use it as a label
        person = URIRef(to_iri(data + row['Name']))
        name = Literal(row['Name'], datatype=XSD['string'])
        # `Country` is a resource
        country = URIRef(to_iri(data + row['Country']))
        # But we'd also like to use the name as a label (with a language tag!)
        country_name = Literal(row['Country'], lang='en')
        # `Age` is a literal (an integer)
        age = Literal(int(row['Age']), datatype=XSD['int'])
        # `Favourite Colour` is a resource
        colour = URIRef(to_iri(data + row['Favourite Colour']))
        colour_name = Literal(row['Favourite Colour'], lang='en')
        # `Place` is a resource
        place = URIRef(to_iri(data + row['Place']))
        place_name = Literal(row['Place'], lang='en')
        # `Address` is a literal (a string)
        address = Literal(row['Address'], datatype=XSD['string'])
        # `Hobby` is a resource
        hobby = URIRef(to_iri(data + row['Hobby']))
        hobby_name = Literal(row['Hobby'], lang='en')

        # All set... we are now going to add the triples to our graph
        graph.add((person, VOCAB['name'], name))
        graph.add((person, VOCAB['age'], age))
        graph.add((person, VOCAB['address'], address))

        # Add the place and its label
        graph.add((person, VOCAB['place'], place))
        graph.add((place, VOCAB['name'], place_name))

        # Add the country and its label
        graph.add((person, VOCAB['country'], country))
        graph.add((country, VOCAB['name'], country_name))

        # Add the favourite colour and its label
        graph.add((person, VOCAB['favourite_colour'], colour))
        graph.add((colour, VOCAB['name'], colour_name))

        # Add the hobby and its label
        graph.add((person, VOCAB['hobby'], hobby))
        graph.add((hobby, VOCAB['name'], hobby_name))

    # with open('example-simple.trig','w') as f:
    #     graph.serialize(f, format='trig')

if __name__ == '__main__':
    main()
