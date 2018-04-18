from lxml import etree




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
            d["Mathematical Entity"] = etree.tostring(mathml_xml, encoding="unicode", method="text")
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
                    d["symbol"] =  children.text
                    d["id"] = mathml_segmentation.counter
                    mathml_segmentation.counter += 1
                    l.append(d)
                if children.tag == '{http://www.w3.org/1998/Math/MathML}mo':
                    d = {}
                    d['operator'] =   children.text
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

    # mathml_xml_string = u"""
    #     <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
    #       <mrow>
    #         <mi>x</mi>
    #         <mo>=</mo>
    #         <mfrac>
    #           <mrow>
    #             <mrow>
    #               <mo>-</mo>
    #               <mi>b</mi>
    #             </mrow>
    #             <mo>&#xB1;</mo>
    #             <msqrt>
    #               <mrow>
    #                 <msup>
    #                   <mi>b</mi>
    #                   <mn>2</mn>
    #                 </msup>
    #                 <mo>-</mo>
    #                 <mrow>
    #                   <mn>4</mn>
    #                   <mo>&#x2062;</mo>
    #                   <mi>a</mi>
    #                   <mo>&#x2062;</mo>
    #                   <mi>c</mi>
    #                 </mrow>
    #               </mrow>
    #             </msqrt>
    #           </mrow>
    #           <mrow>
    #             <mn>2</mn>
    #             <mo>&#x2062;</mo>
    #             <mi>a</mi>
    #           </mrow>
    #         </mfrac>
    #       </mrow>
    #     </math>
    #        """
    ms = mathml_segmentation()

    mathml_xml = etree.fromstring(mathml_xml_string)
    print (ms.parse_mathml(mathml_xml))

    # print ("""@prefix mydb: <http://mydb.org/> .
    # @prefix schema: <http://schema.org/> .
    # @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    # @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    # """)



if __name__ == '__main__':
    main()
