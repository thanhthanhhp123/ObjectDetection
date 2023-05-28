from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, \
    PhotometricDistort, Expand, RandomSampleCrop, \
    ToPercentCoords, Resize, SubtractMeans, RandomMirror
from lib import *
from extract_infrom_annotation import Anno_xml
from make_datapath import make_data_path_list
class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(), #convert image from int to float
                ToAbsoluteCoords(), #convert bounding box from ratio to pixel
                PhotometricDistort(), #change brightness, contrast, saturation, hue
                Expand(color_mean), #expand image
                RandomSampleCrop(), #random crop image
                RandomMirror(), #random mirror image
                ToPercentCoords(), #convert bounding box from pixel to ratio
                Resize(input_size), #resize image
                SubtractMeans(color_mean) #subtract mean
            ]),
            "val": Compose([
                ConvertFromInts(), #convert image from int to float
                Resize(input_size), #resize image
                SubtractMeans(color_mean) #subtract mean
            ])
        }
    
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)

if __name__ == "__main__":
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)

    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)
    h, w, _ = img.shape

    #annotation information
    trans_anno = Anno_xml(classes)
    anno_infor_list = trans_anno(train_annotation_list[0], w, h)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    color_mean = (104, 117, 123)
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    #transform img
    phase = "train"
    img_transformed, boxes, labels = transform(img, phase, anno_infor_list[:, :4], anno_infor_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

    #transform val img
    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, anno_infor_list[:, :4], anno_infor_list[:, 4])

    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()
