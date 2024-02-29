import xmltodict
import os
os.chdir("/home/kteacher/Desktop/Deep Learning Project/yolo")

with open("data/datasets/VOC2008/Annotations/2007_000027.xml") as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
    print(data_dict["annotation"])


    
