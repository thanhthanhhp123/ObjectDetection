from lib import *
from make_datapath import make_data_path_list

class Anno_xml(object):
    def __init__(self, classes):
        self.classes = classes
    
    def __call__(self, xml_path, width, height):
        ret = []
        #read xml file
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue
            #information for bouding box
            bndbox = []
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for pt in pts:
                pixel = int(bbox.find(pt).text) - 1
                
                if pt == 'xmin' or pt == 'xmax':
                    pixel /= width #RATIO OF WIDTH
                else:
                    pixel /= height #RATIO OF HEIGHT
                
                bndbox.append(pixel)
            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += [bndbox]
        return np.array(ret) #[[xmin, ymin, xmax, ymax, label_id], ...]

if __name__ == '__main__':
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    anno_xml = Anno_xml(classes)

    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)
    idx = 1
    img_file_path = val_img_list[idx]

    img = cv2.imread(img_file_path)
    h, w, _ = img.shape
    # print("Image size {}, {}".format(w, h))

    annotation_infor = anno_xml(val_annotation_list[idx], w, h)
    print(annotation_infor)
