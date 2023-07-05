from shutil import copyfile

import cv2
import os
import numpy as np
import json

from tqdm import tqdm


def read_label(lable_files_path, image_root=None):
    datasets = []
    datas = json.load(open(lable_files_path, "r"))
    images = datas["images"]
    categories = datas["categories"]
    annotations = datas["annotations"]
    image_all_labels = {}
    for annotation in tqdm(annotations):
        image_name = images[f"{annotation['image_id']}"]
        image_path = os.path.join(image_root, image_name)
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        xmin, ymin, xmax, ymax = annotation['bbox']
        keypoints = annotation['keypoints']
        category_id = annotation['category_id']
        if category_id >= len(categories):
            continue
        categorie = categories[category_id-1]
        if categorie["name"] != "dog":
            continue

        dataset = [0]  # 类比索引，只有一类，所以全是0

        cell_w = xmax-xmin
        cell_h = ymax-ymin
        cell_x = xmin + w/2
        cell_y = ymin + h/2
        cell_x, cell_y, cell_w, cell_h = cell_x/w, cell_y/h, cell_w/w, cell_h/h
        dataset.extend([cell_x, cell_y, cell_w, cell_h])  # 添加x,y,w,h

        for keypoint in keypoints:
            keypoint_x, keypoint_y, keypoint_visible = keypoint
            keypoint_x = keypoint_x / w
            keypoint_y = keypoint_y / h
            dataset.extend([keypoint_x, keypoint_y, keypoint_visible])

        image_all_labels.setdefault(image_name, [])
        image_all_labels[image_name].append(dataset)

    for image_name, label in image_all_labels.items():
        absolute_image_path = os.path.join(image_root, image_name)
        datasets.append({"absolute_image_path": absolute_image_path, "relative_image_path": os.path.join("images", image_name), "label": label})
    return datasets


def make(save_root, datasets, data_type):
    save_image_dir = os.path.join(save_root, "images")
    if not os.path.exists(save_image_dir):
        os.makedirs(save_image_dir)
    save_image_type_dir = os.path.join(save_image_dir, data_type)
    if not os.path.exists(save_image_type_dir):
        os.makedirs(save_image_type_dir)

    save_label_dir = os.path.join(save_root, "labels")
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    save_label_type_dir = os.path.join(save_label_dir, data_type)
    if not os.path.exists(save_label_type_dir):
        os.makedirs(save_label_type_dir)

    dataset_type_file = os.path.join(save_root, f"{data_type}.txt")
    dataset_type_datas = [f"./images/{data_type}/{os.path.split(dataset['relative_image_path'])[1]}" for dataset in datasets]
    dataset_type_datas = "\n".join(dataset_type_datas)
    with open(dataset_type_file, "w") as f:
        f.write(dataset_type_datas)
    for dataset in tqdm(datasets, desc=data_type):
        absolute_image_path = dataset['absolute_image_path']
        relative_image_path = dataset['relative_image_path']
        file_name = os.path.split(relative_image_path)[1]
        dataset_type_label_file = os.path.join(save_label_type_dir, f"{os.path.splitext(file_name)[0]}.txt")
        dataset_type_label_datas = dataset["label"]
        dataset_type_label_datas_str = []
        for values in dataset_type_label_datas:
            dataset_type_label_datas_str.append(" ".join([str(value) for value in values]))
        with open(dataset_type_label_file, "w") as f:
            dataset_type_label_datas_str = "\n".join(dataset_type_label_datas_str)
            f.write(dataset_type_label_datas_str)
        copyfile(absolute_image_path, f"{save_image_type_dir}/{file_name}")


def main():
    train_ratio = 0.8
    root = r"/media/wcirq/2EBD1711CB9A9F631/datasets/tianchi/all_dogs/Animal-Pose_Dataset/raw/animalpose_keypoint_new"
    save_root = r"/mnt/develop/PycharmProject/github/yolov7/data/Dog-Pose-keypoints"
    image_root = os.path.join(root, "images")
    label_path = os.path.join(root, "keypoints.json")
    datasets = read_label(label_path, image_root)
    np.random.shuffle(datasets)
    train_num = int(train_ratio * len(datasets))
    train_datasets = datasets[:train_num]
    valid_datasets = datasets[train_num:]
    make(save_root, train_datasets, "train")
    make(save_root, valid_datasets, "valid")
    print()


if __name__ == '__main__':
    main()