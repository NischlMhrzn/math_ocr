import cv2
import matplotlib.pyplot as plt
import numpy as np


def pad_images(img, expected_size):
    pad_img = np.zeros((expected_size, expected_size, 3), dtype="uint8")
    h, w = img.shape[0], img.shape[1]
    ratio = 1
    if max(h, w) > expected_size:
        ratio = expected_size / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    pad_img[0 : img.shape[0], 0 : img.shape[1]] = img
    return pad_img, ratio


def get_bbox_crops(image, bboxes):
    crops = []
    for bbox in bboxes:
        crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        crops.append(crop)
    return crops


def get_equation_removed(image, bboxes):
    new_image = np.copy(image)
    for bbox in bboxes:
        new_image[bbox[1] : bbox[3], bbox[0] : bbox[2]] = np.full(
            new_image[bbox[1] : bbox[3], bbox[0] : bbox[2]].shape, fill_value=255
        )
    return new_image


def pad_bbox2_img(bboxes, ratio):
    new_bbox = []
    for bbox in bboxes:
        bbox = [int(i / ratio) for i in bbox]
        new_bbox.append(bbox)
    return new_bbox


if __name__ == "__main__":
    img_path = "/home/honey/Freelancing/math_ocr/test1.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bbox = [
        [138, 32, 231, 47],
        [96, 151, 309, 167],
        [93, 176, 310, 192],
        [92, 126, 318, 142],
        [166, 57, 239, 89],
        [93, 102, 310, 117],
        [256, 102, 310, 117],
        [253, 177, 311, 192],
        [199, 60, 243, 81],
    ]

    crops = get_bbox_crops(image, bbox)
    text = get_equation_removed(image, bbox)
    plt.imshow(text)
    plt.show()
    plt.imshow(image)
    plt.show()
    for i in crops:
        plt.imshow(i)
        plt.show()
    print("Hello")
