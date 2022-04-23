import cv2
import torch
import numpy as np


def draw_box(image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)


def detect(images, detections, thres=0.2):
    cls = 1
    boxes = []
    scores = []
    for k in range(len(images)):

        img_boxes = []
        img_scores = []
        for j in range(detections.size(2)):
            if detections[k, cls, j, 0] < thres:
                continue

            pt = detections[k, cls, j, 1:]
            coords = (pt[0], pt[1], pt[2], pt[3])
            img_boxes.append(coords)
            img_scores.append(detections[k, cls, j, 0])

            boxes.append(img_boxes)
            scores.append(img_scores)

    return boxes, scores


def FixImgCoordinates(images, boxes):
    new_boxes = []
    if isinstance(images, list):
        for i in range(len(images)):
            print(images[i].shape)
            bbs = []
            for o_box in boxes[i]:
                b = [None] * 4
                b[0] = int(o_box[0] * images[i].shape[0])
                b[1] = int(o_box[1] * images[i].shape[1])
                b[2] = int(o_box[2] * images[i].shape[0])
                b[3] = int(o_box[3] * images[i].shape[1])
                bbs.append(b)

            new_boxes.append(bbs)
    else:
        bbs = []
        for o_box in boxes[0]:
            b = [None] * 4
            b[0] = int(o_box[0] * images.shape[1])
            b[1] = int(o_box[1] * images.shape[0])
            b[2] = int(o_box[2] * images.shape[1])
            b[3] = int(o_box[3] * images.shape[0])
            bbs.append(b)

            # this could be
            # b[0] = int(o_box[0] * images.shape[0]) ==> b[0] = int(o_box[0] * images.shape[1])
            # b[1] = int(o_box[1] * images.shape[1]) ==> b[1] = int(o_box[1] * images.shape[0])
            # b[2] = int(o_box[2] * images.shape[0]) ==> b[2] = int(o_box[2] * images.shape[1])
            # b[3] = int(o_box[3] * images.shape[1]) ==> b[3] = int(o_box[3] * images.shape[0])

        new_boxes.append(bbs)

    return new_boxes


def DrawAllBoxes(images, boxes):
    for i in range(len(images)):
        draw_box(images[i], boxes[i])
