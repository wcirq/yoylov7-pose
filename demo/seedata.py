import cv2
import os
import numpy as np


def read_label(path, image=None):
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = line.split(" ")
            data = [float(d) for d in data]

            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
            ]
            img_h, img_w, _ = image.shape

            cx, cy, w, h = data[1:5]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            w *= img_w
            h *= img_h
            x2, y2 = x1+w, y1+h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            keypoints = np.array(data[5:]).reshape((-1, 3))
            for j, (x, y, v) in enumerate(keypoints):
                x = int(x*img_w)
                y = int(y*img_h)
                v = int(v)
                cv2.circle(image, (x, y), i+2, colors[v], -1)
                cv2.putText(image, f"{j}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("image", image)
        cv2.waitKey(0)
    return

def main():
    root = r"/mnt/develop/PycharmProject/github/yolov7/data/coco2017labels-keypoints"
    images_root = os.path.join(root, "images", "train2017")
    labels_root = os.path.join(root, "labels", "train2017")
    label_file_names = os.listdir(labels_root)
    for label_file_name in label_file_names:
        label_path = os.path.join(labels_root, label_file_name)
        image_path = os.path.join(images_root, f"{os.path.splitext(label_file_name)[0]}.jpg")

        image = cv2.imread(image_path)
        label = read_label(label_path, image)



if __name__ == '__main__':
    main()