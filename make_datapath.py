from lib import *
def make_data_path_list(root_path):
    image_path_template = os.path.join(root_path, "JPEGImages", "%s.jpg")
    annotation_path_template = os.path.join(root_path, "Annotations", "%s.xml")

    train_id_names = os.path.join(root_path, "ImageSets/Main/train.txt")
    val_id_names = os.path.join(root_path, "ImageSets/Main/val.txt")

    train_img_list = list()
    train_annotation_list = list()

    val_img_list = list()
    val_annotation_list = list()
    for line in open(train_id_names):
        file_id = line.strip() #xoá ký tự xuống dòng
        img_path = (image_path_template % file_id) 
        anno_path = (annotation_path_template % file_id)
        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)
    
    for line in open(val_id_names):
        file_id = line.strip() #xoá ký tự xuống dòng
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)

        val_img_list.append(img_path)
        val_annotation_list.append(anno_path)
    
    return train_img_list, train_annotation_list, val_img_list, val_annotation_list

if __name__ == "__main__":
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list = make_data_path_list(root_path)
    print(train_img_list[0])
    print(train_annotation_list[0])
    print(val_img_list[0])
    print(val_annotation_list[0])
    print(len(train_img_list))
    print(len(train_annotation_list))
    print(len(val_img_list))
    print(len(val_annotation_list))