
import xml.dom.minidom as xmldom


def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    # print(eles.tagName)
    xmin = eles.getElementsByTagName("xmin")[0].firstChild.data
    xmax = eles.getElementsByTagName("xmax")[0].firstChild.data
    ymin = eles.getElementsByTagName("ymin")[0].firstChild.data
    ymax = eles.getElementsByTagName("ymax")[0].firstChild.data
    print(xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax


def test_parse_xml(num):

    parse_xml('./'+'/cup_label/'+str(num)+'.xml')
        

if __name__ == "__main__":
    
    
    num = 1
    while True:
        if num<100:
            
            test_parse_xml(num)
            num += 1

