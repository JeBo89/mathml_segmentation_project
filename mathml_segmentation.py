from lxml import etree




class mathml_segmentation:
    NS = ["http: // www.w3.org / 1998 / Math / MathML"]

    def __init__(self):
        return None

    def parse_mathml(self,mathml_xml):
        for children in mathml_xml.getchildren():
            if children.getchildren() != []:
                self.parse_mathml(children)

                # print children.text
            else:
                if children.tag == '{http://www.w3.org/1998/Math/MathML}mi':
                    print 'mi',  children.text

                if children.tag == '{http://www.w3.org/1998/Math/MathML}mo':
                    print 'mo',  children.text

                # print children.tag



def main():
    mathml_xml_string = """
           <math xmlns=\"http://www.w3.org/1998/Math/MathML\" display=\"block\" alttext=\"{\\displaystyle f(x+T)=f(x)}\">
               <semantics>
                   <mrow class=\"MJX-TeXAtom-ORD\">
                     <mstyle displaystyle=\"true\" scriptlevel=\"0\">
                       <mi>f</mi>
                       <mo stretchy=\"false\">(</mo>
                       <mi>x</mi>
                       <mo>+</mo>
                       <mi>T</mi>
                       <mo stretchy=\"false\">)</mo>
                       <mo>=</mo>
                       <mi>f</mi>
                       <mo stretchy=\"false\">(</mo>
                       <mi>x</mi>
                       <mo stretchy=\"false\">)</mo>
                     </mstyle>
                   </mrow>
                   <annotation encoding=\"application/x-tex\">{\\displaystyle f(x+T)=f(x)}</annotation>
               </semantics>
           </math>
           """

    ms = mathml_segmentation()
    NS = "http://www.w3.org/1998/Math/MathML"

    mathml_xml = etree.fromstring(mathml_xml_string)
    ms.parse_mathml(mathml_xml)


if __name__ == '__main__':
    main()
