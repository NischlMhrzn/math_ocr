import cv2
from utils.detection.visualize import plot_bbox
import matplotlib.pyplot as plt


def overlap(bbox_1, bbox_2):
    if (
        (bbox_2[3] >= bbox_1[1])
        and (bbox_2[1] <= bbox_1[3])
        and ((bbox_2[2] >= bbox_1[0]) and (bbox_2[0] <= bbox_1[2]))
    ):
        return True
    else:
        return False


def remove_overlap(bboxes):
    bboxes = sorted(bboxes, key=lambda x: x[1] + x[0])
    combined = []
    removed = []
    for i in range(len(bboxes)):
        processed = bboxes[i]
        if bboxes[i] in removed:
            continue
        else:
            for j in range(i + 1, len(bboxes)):
                if bboxes[i] in removed:
                    continue
                if overlap(processed, bboxes[j]):
                    processed[0] = min(bboxes[j][0], processed[0])
                    processed[1] = min(bboxes[j][1], processed[1])
                    processed[2] = max(bboxes[j][2], processed[2])
                    processed[3] = max(bboxes[j][3], processed[3])
                    removed.append(bboxes[j])
        combined.append(processed)
    return combined


def adjust_pad(image, bboxes):
    adjusted_bboxes = []
    for bbox in bboxes:
        crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        _, binary_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
        top_row = binary_crop[0]
        bottom_row = binary_crop[-1]
        first_column = binary_crop[:, 0]
        last_column = binary_crop[:, -1]
        edges = [first_column, top_row, last_column, bottom_row]
        for index, edge in enumerate(edges):
            if check_black_pixels(edge):
                bbox = extend_crops(image, bbox, index)
        crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        adjusted_bboxes.append(bbox)
    return adjusted_bboxes


def extend_crops(image, bbox, index):
    if index < 2:
        bbox[index] = bbox[index] - 1
    else:
        bbox[index] = bbox[index] + 1
    extended_crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    gray_crop = cv2.cvtColor(extended_crop, cv2.COLOR_RGB2GRAY)
    _, binary_crop = cv2.threshold(gray_crop, 127, 255, cv2.THRESH_BINARY)
    if index == 0:
        edge = binary_crop[:, 0]
    elif index == 1:
        edge = binary_crop[0]
    elif index == 2:
        edge = binary_crop[:, -1]
    elif index == 3:
        edge = binary_crop[-1]
    if check_black_pixels(edge):
        bbox = extend_crops(image, bbox, index)
        return bbox
    else:
        return bbox


def check_black_pixels(pixels):
    assert pixels.dtype == "uint8", "The image must be interger dtype"
    if 0 in pixels:
        return True
    else:
        return False


def process_bboxes(image, bboxes):
    bboxes = remove_overlap(bboxes)
    bboxes = adjust_pad(image, bboxes)
    return bboxes


if __name__ == "__main__":
    bboxes = [
        [83, 177, 313, 193],
        [89, 129, 319, 142],
        [234, 126, 320, 141],
        [198, 102, 306, 117],
        [85, 102, 188, 118],
        [141, 126, 257, 141],
        [84, 146, 315, 161],
        [86, 161, 310, 176],
        [202, 151, 307, 166],
        [125, 31, 233, 46],
        [80, 177, 190, 193],
        [86, 126, 186, 142],
        [144, 151, 275, 165],
        [165, 59, 244, 94],
        [108, 151, 211, 167],
        [199, 177, 309, 193],
        [133, 101, 269, 117],
    ]
    image = cv2.imread("/home/honey/Freelancing/math_ocr/test1.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_bboxes = remove_overlap(bboxes)
    new_bboxes = adjust_pad(image, new_bboxes)
    for i in new_bboxes:
        plot_bbox(image, i)
