import xmltodict
import cv2
import os
os.chdir(os.getcwd())

class VOCDatasets:
    def __init__(self, path="data/datasets/VOC2008/"):
        self.path = path
        self.datasets = []
        self.classes = {
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20
        }
        self.__create_datasets__()

    def __create_datasets__(self):
        path_to_images = self.path + "JPEGImages"
        path_to_xml = self.path + "Annotations"
        for filename in os.listdir(path_to_xml):
            data_dict = {}
            image = cv2.imread(path_to_images + "/" + filename[0:11] + ".jpg")
            data_dict["image"] = image
            with open(path_to_xml + "/" + filename) as xml_file:
                dict_info = xmltodict.parse(xml_file.read())
                annotation = dict_info["annotation"]
                data_dict["width"] = annotation["size"]["width"]
                data_dict["height"] = annotation["size"]["height"]
                objects = []
                bbox = []
                if isinstance(annotation["object"], list):
                    for object in annotation["object"]:
                        objects.append(self.classes[object["name"]])
                        box = [float(object["bndbox"]["xmin"]), float(object["bndbox"]["ymin"]), float(object["bndbox"]["xmax"]), float(object["bndbox"]["ymax"])]
                        bbox.append(box)
                    data_dict["objects"] = objects
                    data_dict["bbox"] = bbox
                else:
                    objects.append(self.classes[annotation["object"]["name"]])
                    box = [float(annotation["object"]["bndbox"]["xmin"]), float(annotation["object"]["bndbox"]["ymin"]), float(annotation["object"]["bndbox"]["xmax"]), float(annotation["object"]["bndbox"]["ymax"])]
                    bbox.append(box)
                    data_dict["objects"] = objects
                    data_dict["bbox"] = bbox
            self.datasets.append(data_dict)    
            
    def display_image(self, index):
        item = self.datasets[index]
        image = item["image"].copy()
        for i in range(len(item["bbox"])):
            box = [int(coord) for coord in item["bbox"][i]]
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Object Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def get_data(self):
        return self.datasets
        
        
voc = VOCDatasets()
voc.display_image(15)
# with open("data/datasets/VOC2008/Annotations/2007_000033.xml") as xml_file:
#     data_dict = xmltodict.parse(xml_file.read())
#     for data in data_dict["annotation"]["object"]:
#         print(data["name"])
    #print(data_dict["annotation"]["size"])


    
