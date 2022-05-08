import cv2
from utils.detection.visualize import plot_bbox
from operator import itemgetter


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
    new_bboxes = remove_overlap(bboxes)
    for i in new_bboxes:
        plot_bbox(image, i)
